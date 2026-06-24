"""Read and write mmCIF-format structure files.

Parses the ``_atom_site`` loop from the first ``data_`` block in the file.
All other categories (symmetry, entity, struct_conn, etc.) are silently
skipped.

Multiline ``;`` text blocks inside an ``_atom_site`` loop are not supported
(they essentially never occur for coordinate data); a :exc:`ValueError` is
raised if one is encountered so the user knows to pre-process the file.

References
----------
PDBx/mmCIF dictionary:
    https://mmcif.wwpdb.org/dictionaries/mmcif_pdbx_v50.dic/Index/
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union

from ..structure import Structure


# ---------------------------------------------------------------------------
# Mapping from _atom_site.* field names → Structure atom-table column names
# ---------------------------------------------------------------------------

# Each entry: (_atom_site field, destination column, default if absent)
_FIELD_MAP = [
    ("group_PDB",           "record_name",    "ATOM"),
    ("id",                  "atom_serial",    pd.NA),
    ("type_symbol",         "element",        ""),
    ("label_atom_id",       "atom_name",      None),      # required
    ("label_alt_id",        "alt_loc",        ""),
    ("label_comp_id",       "res_name",       None),      # required
    ("label_asym_id",       "label_asym_id",  pd.NA),
    ("auth_asym_id",        "chain_id",       None),      # required; fallback below
    ("label_seq_id",        "label_seq_id",   pd.NA),
    ("auth_seq_id",         "res_seq",        None),      # required; fallback below
    ("label_entity_id",     "label_entity_id",pd.NA),
    ("pdbx_PDB_ins_code",   "icode",          ""),
    ("occupancy",           "occupancy",      1.0),
    ("B_iso_or_equiv",      "b_factor",       0.0),
    ("pdbx_formal_charge",  "charge",         ""),
    ("pdbx_PDB_model_num",  "model",          1),
    # Cartn_x/y/z are handled separately → coords array
]

_CIF_NULL = {".", "?"}


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

def _tokenize_cif_line(line: str) -> list:
    """Tokenize a single mmCIF data line.

    Handles:
    * Bare tokens (whitespace-separated)
    * Single-quoted strings ``'...'``
    * Double-quoted strings ``"..."``

    The closing quote must be followed by whitespace or end-of-string (CIF
    rule) to avoid splitting values that internally contain unmatched quotes.

    Parameters
    ----------
    line : str
        A single stripped line of CIF text (no newline).

    Returns
    -------
    list of str
        Tokens; ``.`` and ``?`` are returned as-is (callers map to nulls).
    """
    tokens = []
    i = 0
    n = len(line)

    while i < n:
        # Skip whitespace
        if line[i].isspace():
            i += 1
            continue

        # Inline comment (rare but valid)
        if line[i] == "#":
            break

        # Quoted string
        if line[i] in ("'", '"'):
            quote = line[i]
            i += 1
            start = i
            while i < n:
                if line[i] == quote:
                    # CIF rule: closing quote must be followed by whitespace/EOL
                    if i + 1 >= n or line[i + 1].isspace():
                        tokens.append(line[start:i])
                        i += 1  # skip the closing quote
                        break
                i += 1
            else:
                # Unterminated quote — append what we have
                tokens.append(line[start:i])
            continue

        # Bare token
        start = i
        while i < n and not line[i].isspace():
            i += 1
        tokens.append(line[start:i])

    return tokens


def _read_atom_site_loop(lines: list, start: int, filepath: Path):
    """Parse an ``_atom_site`` loop_ block starting at *start*.

    Parameters
    ----------
    lines : list of str
        All lines of the file, stripped of newlines.
    start : int
        Index of the ``loop_`` line that opens this block.
    filepath : Path
        Used only in error messages.

    Returns
    -------
    tuple (headers, rows)
        *headers* : list of ``_atom_site.*`` field names (in order).
        *rows* : list of lists of str tokens (one per atom).
    """
    i = start + 1  # first line after `loop_`
    n = len(lines)
    headers = []
    rows = []
    in_multiline = False

    # Phase 1: collect headers
    while i < n:
        line = lines[i].strip()
        if not line or line.startswith("#"):
            i += 1
            continue
        if line.startswith("_atom_site.") or line.startswith("_atom_site_"):
            # normalise to dot notation
            field = line.split()[0]
            if "." not in field:
                field = field.replace("_atom_site_", "_atom_site.", 1)
            headers.append(field[len("_atom_site."):])
            i += 1
        else:
            break  # first data line

    if not headers:
        return headers, rows

    # Phase 2: collect data rows
    n_cols = len(headers)
    current_row_tokens: list = []

    while i < n:
        line = lines[i]
        stripped = line.strip()
        i += 1

        # Multiline text block (;-delimited)
        if stripped.startswith(";"):
            raise ValueError(
                f"Multiline ';' text value in _atom_site loop is not supported "
                f"(line {i} in {filepath}).  Pre-process the file to remove "
                f"multiline values."
            )

        # End of loop: new category, new loop, new data block, or EOF
        if (
            stripped.startswith("loop_")
            or stripped.startswith("_")
            or stripped.startswith("data_")
            or stripped.startswith("save_")
            or stripped == "#"
        ):
            break

        if not stripped or stripped.startswith("#"):
            continue

        tokens = _tokenize_cif_line(stripped)
        current_row_tokens.extend(tokens)

        # Emit complete rows as we accumulate n_cols tokens
        while len(current_row_tokens) >= n_cols:
            rows.append(current_row_tokens[:n_cols])
            current_row_tokens = current_row_tokens[n_cols:]

    return headers, rows


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------

def read_mmcif(filepath: Union[str, Path]) -> Structure:
    """Parse an mmCIF-format file and return a :class:`~bioviper2.Structure`.

    Reads the ``_atom_site`` loop from the first ``data_`` block.  Both
    ``ATOM`` and ``HETATM`` (``group_PDB`` values) are included.

    Auth-facing identifiers (``auth_asym_id``, ``auth_seq_id``) are used as
    ``chain_id`` / ``res_seq`` to match PDB-style biological conventions.
    When auth fields are absent the corresponding ``label_*`` fields are used
    as a fallback.  Both label and auth ids are stored in the atom table for
    lossless mmCIF → mmCIF round-trips.

    Parameters
    ----------
    filepath : str or pathlib.Path
        Path to the ``.cif`` or ``.mmcif`` file.

    Returns
    -------
    Structure

    Raises
    ------
    ValueError
        If no ``_atom_site`` loop is found, or if the file contains an
        unsupported multiline ``;`` text block in the atom-site loop.
    """
    filepath = Path(filepath)

    with open(filepath) as fh:
        raw_lines = fh.readlines()
    lines = [l.rstrip("\n") for l in raw_lines]

    # Find the _atom_site loop_
    atom_site_loop_start = None
    in_data = False
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("data_"):
            if in_data:
                # second data block — stop
                break
            in_data = True
            continue
        if stripped == "loop_":
            # Peek ahead to see if this loop contains _atom_site fields
            for j in range(idx + 1, min(idx + 20, len(lines))):
                peek = lines[j].strip()
                if not peek or peek.startswith("#"):
                    continue
                if peek.startswith("_atom_site"):
                    atom_site_loop_start = idx
                    break
                if not peek.startswith("_"):
                    break
            if atom_site_loop_start is not None:
                break

    if atom_site_loop_start is None:
        raise ValueError(f"No _atom_site records found in {filepath}")

    headers, rows = _read_atom_site_loop(lines, atom_site_loop_start, filepath)

    if not rows:
        raise ValueError(f"No _atom_site records found in {filepath}")

    # Build a column index: field name → position in each row
    col_idx = {h: i for i, h in enumerate(headers)}

    def _get(row, field, default):
        """Extract a token from a row by field name, mapping CIF nulls."""
        if field not in col_idx:
            return default
        val = row[col_idx[field]]
        if val in _CIF_NULL:
            return default
        return val

    # Coordinate column indices (required)
    for coord_field in ("Cartn_x", "Cartn_y", "Cartn_z"):
        if coord_field not in col_idx:
            raise ValueError(
                f"_atom_site loop in {filepath} is missing required field "
                f"_atom_site.{coord_field}"
            )

    xs, ys, zs = [], [], []
    records: list[dict] = []

    for row in rows:
        try:
            x = float(row[col_idx["Cartn_x"]])
            y = float(row[col_idx["Cartn_y"]])
            z = float(row[col_idx["Cartn_z"]])
        except (ValueError, KeyError) as exc:
            raise ValueError(
                f"Cannot parse coordinates in _atom_site loop in {filepath}: {exc}"
            ) from exc

        xs.append(x)
        ys.append(y)
        zs.append(z)

        rec_name = _get(row, "group_PDB", "ATOM").strip()
        is_hetatm = rec_name == "HETATM"

        # chain_id: prefer auth_asym_id, fall back to label_asym_id
        chain_id = _get(row, "auth_asym_id", None)
        if chain_id is None:
            chain_id = _get(row, "label_asym_id", "")
        if chain_id is None:
            chain_id = ""

        # res_seq: prefer auth_seq_id, fall back to label_seq_id
        res_seq_raw = _get(row, "auth_seq_id", None)
        if res_seq_raw is None:
            res_seq_raw = _get(row, "label_seq_id", None)
        try:
            res_seq = int(res_seq_raw) if res_seq_raw is not None else pd.NA
        except ValueError:
            res_seq = pd.NA

        # atom_serial
        serial_raw = _get(row, "id", None)
        try:
            atom_serial = int(serial_raw) if serial_raw is not None else pd.NA
        except ValueError:
            atom_serial = pd.NA

        # label_seq_id (stored separately; may legitimately be "." for HETATM)
        lseq_raw = _get(row, "label_seq_id", None)
        try:
            label_seq_id = int(lseq_raw) if lseq_raw is not None else pd.NA
        except ValueError:
            label_seq_id = pd.NA

        # model number
        model_raw = _get(row, "pdbx_PDB_model_num", "1")
        try:
            model = int(model_raw)
        except ValueError:
            model = 1

        # icode: "." or "?" → ""
        icode_raw = row[col_idx["pdbx_PDB_ins_code"]] if "pdbx_PDB_ins_code" in col_idx else ""
        icode = "" if icode_raw in _CIF_NULL else icode_raw.strip()

        # alt_loc
        alt_raw = _get(row, "label_alt_id", "")
        alt_loc = "" if alt_raw in _CIF_NULL or alt_raw is None else alt_raw.strip()

        # occupancy / b_factor
        occ_raw = _get(row, "occupancy", None)
        try:
            occupancy = float(occ_raw) if occ_raw is not None else 1.0
        except ValueError:
            occupancy = 1.0

        bfac_raw = _get(row, "B_iso_or_equiv", None)
        try:
            b_factor = float(bfac_raw) if bfac_raw is not None else 0.0
        except ValueError:
            b_factor = 0.0

        charge_raw = _get(row, "pdbx_formal_charge", "")
        charge = "" if charge_raw in _CIF_NULL or charge_raw is None else charge_raw.strip()

        records.append({
            "record_name":     rec_name,
            "atom_serial":     atom_serial,
            "atom_name":       _get(row, "label_atom_id", "").strip(),
            "alt_loc":         alt_loc,
            "res_name":        _get(row, "label_comp_id", "").strip(),
            "chain_id":        chain_id,
            "res_seq":         res_seq,
            "icode":           icode,
            "occupancy":       occupancy,
            "b_factor":        b_factor,
            "element":         _get(row, "type_symbol", "").strip().upper(),
            "charge":          charge,
            "model":           model,
            "hetero":          is_hetatm,
            "label_asym_id":   _get(row, "label_asym_id", pd.NA),
            "label_seq_id":    label_seq_id,
            "label_entity_id": _get(row, "label_entity_id", pd.NA),
        })

    coords = np.array([xs, ys, zs], dtype=np.float64).T
    atoms = pd.DataFrame(records)
    atoms["atom_serial"] = atoms["atom_serial"].astype("Int64")
    atoms["res_seq"] = atoms["res_seq"].astype("Int64")
    atoms["label_seq_id"] = atoms["label_seq_id"].astype("Int64")

    return Structure(coords, atoms)


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

# Canonical _atom_site field order for output
_WRITE_FIELDS = [
    ("group_PDB",           "record_name"),
    ("id",                  "atom_serial"),
    ("type_symbol",         "element"),
    ("label_atom_id",       "atom_name"),
    ("label_alt_id",        "alt_loc"),
    ("label_comp_id",       "res_name"),
    ("label_asym_id",       "label_asym_id"),
    ("auth_asym_id",        "chain_id"),
    ("label_entity_id",     "label_entity_id"),
    ("label_seq_id",        "label_seq_id"),
    ("auth_seq_id",         "res_seq"),
    ("pdbx_PDB_ins_code",   "icode"),
    ("Cartn_x",             "__x__"),
    ("Cartn_y",             "__y__"),
    ("Cartn_z",             "__z__"),
    ("occupancy",           "occupancy"),
    ("B_iso_or_equiv",      "b_factor"),
    ("pdbx_formal_charge",  "charge"),
    ("pdbx_PDB_model_num",  "model"),
]


def _cif_val(val, is_numeric: bool = False) -> str:
    """Format a value for CIF output.

    Null/NA values → ``.``.
    Strings that contain whitespace are quoted.
    """
    if val is None or (not isinstance(val, (bool, float)) and pd.isna(val)):
        return "."
    if is_numeric:
        return str(val)
    s = str(val)
    if not s:
        return "."
    if any(c.isspace() for c in s):
        return f'"{s}"'
    return s


def write_mmcif(structure: Structure, filepath: Union[str, Path]) -> None:
    """Write a :class:`~bioviper2.Structure` to mmCIF format.

    Emits a minimal valid mmCIF file with a ``data_`` block and a single
    ``_atom_site`` loop containing all canonical fields.  The output is
    readable by PyMOL, PHENIX, Biotite, and other tools that consume
    PDBx/mmCIF.

    Parameters
    ----------
    structure : Structure
    filepath : str or pathlib.Path
        Output path; the stem is used as the ``data_`` block name.

    Notes
    -----
    Round-trip fidelity:

    * All per-atom annotation columns (including ``label_*`` and entity ids)
      round-trip exactly when the input was read from mmCIF.
    * Files originally read from PDB will have ``<NA>`` for ``label_asym_id``,
      ``label_seq_id``, and ``label_entity_id`` (written as ``.``).
    * Atom serials are written as stored; they are not renumbered.
    """
    filepath = Path(filepath)
    block_name = filepath.stem.replace(" ", "_") or "structure"

    df = structure.atoms
    coords = structure.coords

    # Build a per-row serial if atom_serial is all NA
    use_stored_serial = df["atom_serial"].notna().any()

    with open(filepath, "w") as fh:
        fh.write(f"data_{block_name}\n#\n")
        fh.write("loop_\n")
        for cif_field, _ in _WRITE_FIELDS:
            fh.write(f"_atom_site.{cif_field}\n")

        for i in range(len(df)):
            row = df.iloc[i]
            xyz = coords[i]

            serial = (
                int(row["atom_serial"]) if use_stored_serial and pd.notna(row["atom_serial"])
                else i + 1
            )
            label_asym = _cif_val(row["label_asym_id"])
            label_seq  = _cif_val(row["label_seq_id"])
            label_ent  = _cif_val(row["label_entity_id"])

            # alt_loc / icode / charge: empty string → "."
            alt_loc  = row["alt_loc"]  or "."
            icode    = row["icode"]    or "."
            charge   = row["charge"]   or "."

            res_seq_val = str(int(row["res_seq"])) if pd.notna(row["res_seq"]) else "."

            fh.write(
                f"{row['record_name']:<6s} {serial:5d} "
                f"{_cif_val(row['element']):<2s} "
                f"{_cif_val(row['atom_name']):<4s} "
                f"{alt_loc:<1s} "
                f"{_cif_val(row['res_name']):<3s} "
                f"{label_asym:<4s} "
                f"{_cif_val(row['chain_id']):<4s} "
                f"{label_ent:<4s} "
                f"{label_seq:<6s} "
                f"{res_seq_val:<6s} "
                f"{icode:<1s} "
                f"{xyz[0]:8.3f} {xyz[1]:8.3f} {xyz[2]:8.3f} "
                f"{row['occupancy']:6.2f} {row['b_factor']:6.2f} "
                f"{charge:<2s} "
                f"{int(row['model'])}\n"
            )

        fh.write("#\n")
