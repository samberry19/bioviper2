"""Read and write PDB-format structure files.

Parsed records: ATOM, HETATM, MODEL, ENDMDL, TER, END.
All other record types (HEADER, REMARK, SEQRES, HELIX, SHEET, CONECT,
ANISOU, CRYST1, …) are silently skipped.

PDB format uses fixed-width columns (1-indexed in the spec; 0-indexed Python
slices are used throughout this module).  Lines shorter than 80 characters
are padded with spaces before slicing so that element/charge slices at
cols 77-80 never raise :exc:`IndexError`.

Limitations
-----------
- Chain IDs are read from column 22 (single character).  Files using
  two-character chain IDs in non-standard positions are not supported.
- Atom serials > 99 999 and residue sequence numbers outside ±9 999 (hybrid-36
  encoding) are not decoded; the raw string is stored as ``<NA>`` for serials
  and raises :exc:`ValueError` for res_seq.
- Secondary-structure records, ANISOU, CONECT, and crystallographic metadata
  are not captured.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union

from ..structure import Structure


# ---------------------------------------------------------------------------
# Known 2-letter element symbols relevant to biomolecular structures.
# Used by _guess_element() to avoid confusing CA (alpha-carbon) with Ca
# (calcium) etc.
# ---------------------------------------------------------------------------
_TWO_LETTER_ELEMENTS = {
    "FE", "ZN", "MG", "CA", "MN", "CO", "NI", "CU", "MO",
    "NA", "CL", "BR", "SE", "SI", "HG", "PT", "AU", "AG",
    "AL", "LI", "RB", "CS", "BA", "SR", "AS", "BI", "CD",
    "CR", "GA", "GE", "IN", "IR", "OS", "PB", "PD", "RE",
    "RH", "RU", "SB", "SN", "TE", "TI", "TL", "V",  "W",
    "YB",
}


def _guess_element(raw_name: str, is_hetatm: bool) -> str:
    """Infer element symbol from a 4-character PDB atom name when the element
    column (cols 77-78) is blank.

    PDB convention: the atom name field is 4 characters (cols 13-16).  For
    standard amino-acid/nucleic-acid atoms the element symbol occupies cols
    14-15 (1 or 2 characters), leaving col 13 as a space.  Thus:
    - `` CA `` → the leading space ⇒ single-letter element  → ``C``
    - ``FE  `` → no leading space, 2-letter uppercase prefix → ``FE``
    - `` MG `` → leading space but ``MG`` is a known 2-letter element → ``MG``

    The raw_name passed here is the *un-stripped* 4-character field.
    """
    if len(raw_name) < 4:
        raw_name = raw_name.ljust(4)

    # If leading character is alpha → the element starts at col 0 (4-char name
    # flush-left, e.g. some HETATM).
    # If leading character is space → element starts at col 1.
    if raw_name[0] == " ":
        # Standard case: check 2-char candidate first (e.g. " MG ")
        two = raw_name[1:3].strip().upper()
        if two in _TWO_LETTER_ELEMENTS and is_hetatm:
            return two
        # Single-letter: first alphabetic character at position 1
        for ch in raw_name[1:]:
            if ch.isalpha():
                return ch.upper()
        return ""
    else:
        # Flush-left 4-char name: check first two chars as a 2-letter element
        two = raw_name[0:2].strip().upper()
        if two in _TWO_LETTER_ELEMENTS:
            return two
        # Otherwise first alphabetic character
        for ch in raw_name:
            if ch.isalpha():
                return ch.upper()
        return ""


def _format_atom_name(name: str, element: str) -> str:
    """Format an atom name into the 4-character PDB field (cols 13-16).

    PDB rules:
    - Atom names with a 1-character element symbol and fewer than 4 characters
      are right-padded to 3 characters and placed at columns 14-16 (i.e. one
      leading space): ``" CA "``, ``" N  "``, ``" OG1"``.
    - Atom names with a 2-character element symbol, or names that are 4
      characters long, start at column 13: ``"FE  "``, ``"HD11"``, ``"CA  "``
      (calcium HETATM).

    Examples
    --------
    >>> _format_atom_name("CA", "C")
    ' CA '
    >>> _format_atom_name("N", "N")
    ' N  '
    >>> _format_atom_name("FE", "FE")
    'FE  '
    >>> _format_atom_name("HD11", "H")
    'HD11'
    >>> _format_atom_name("OG1", "O")
    ' OG1'
    """
    name = name.strip()
    elem = element.strip().upper()

    if len(name) >= 4:
        return name[:4]

    if len(elem) >= 2:
        # 2-letter element: flush-left, right-padded to 4
        return name.ljust(4)[:4]

    # 1-letter element and name < 4 chars: one leading space, left-justified
    # in the remaining 3 characters
    return (" " + name).ljust(4)[:4]


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------

def read_pdb(filepath: Union[str, Path]) -> Structure:
    """Parse a PDB-format file and return a :class:`~bioviper2.Structure`.

    Both ``ATOM`` and ``HETATM`` records are read into the same atom table.
    ``MODEL``/``ENDMDL`` records populate the ``model`` column; files with no
    ``MODEL`` records assign ``model=1`` to all atoms.  ``TER`` records are
    ignored on read (they are re-inferred from chain boundaries on write).

    Parameters
    ----------
    filepath : str or pathlib.Path
        Path to the ``.pdb`` or ``.ent`` file.

    Returns
    -------
    Structure

    Raises
    ------
    ValueError
        If no ``ATOM`` or ``HETATM`` records are found, or if a record line
        cannot be parsed.

    Notes
    -----
    Element symbols are read from cols 77-78.  When that field is blank the
    element is inferred from the atom name via :func:`_guess_element`.
    Atom serials that overflow the 5-digit field (hybrid-36 encoding) are
    stored as ``<NA>`` rather than raising.
    """
    filepath = Path(filepath)

    xs, ys, zs = [], [], []
    records: list[dict] = []
    current_model = 1

    with open(filepath) as fh:
        for lineno, raw in enumerate(fh, 1):
            line = raw.rstrip("\n").ljust(80)
            rec = line[0:6].strip()

            if rec == "MODEL":
                try:
                    current_model = int(line[6:14].strip())
                except ValueError:
                    current_model += 1
                continue

            if rec in ("ENDMDL", "TER", "END"):
                continue

            if rec not in ("ATOM", "HETATM"):
                continue

            is_hetatm = rec == "HETATM"

            # Parse numeric fields with graceful fallback
            try:
                serial_raw = line[6:11].strip()
                try:
                    atom_serial = int(serial_raw)
                except ValueError:
                    atom_serial = pd.NA

                atom_name_raw = line[12:16]         # keep raw for element guess
                atom_name = atom_name_raw.strip()
                alt_loc = line[16:17].strip()
                res_name = line[17:20].strip()
                chain_id = line[21:22].strip()

                res_seq_raw = line[22:26].strip()
                res_seq = int(res_seq_raw) if res_seq_raw else pd.NA

                icode = line[26:27].strip()

                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])

                occ_raw = line[54:60].strip()
                occupancy = float(occ_raw) if occ_raw else 1.0

                bfac_raw = line[60:66].strip()
                b_factor = float(bfac_raw) if bfac_raw else 0.0

                elem_raw = line[76:78].strip()
                element = elem_raw.upper() if elem_raw else _guess_element(
                    atom_name_raw, is_hetatm
                )

                charge = line[78:80].strip()

            except (ValueError, IndexError) as exc:
                raise ValueError(
                    f"Malformed {'HETATM' if is_hetatm else 'ATOM'} record at "
                    f"line {lineno} in {filepath}: {exc!s}\n  {raw.rstrip()}"
                ) from exc

            xs.append(x)
            ys.append(y)
            zs.append(z)
            records.append({
                "record_name":  rec,
                "atom_serial":  atom_serial,
                "atom_name":    atom_name,
                "alt_loc":      alt_loc,
                "res_name":     res_name,
                "chain_id":     chain_id,
                "res_seq":      res_seq,
                "icode":        icode,
                "occupancy":    occupancy,
                "b_factor":     b_factor,
                "element":      element,
                "charge":       charge,
                "model":        current_model,
                "hetero":       is_hetatm,
            })

    if not records:
        raise ValueError(f"No atom records found in {filepath}")

    coords = np.array([xs, ys, zs], dtype=np.float64).T   # (n, 3)
    atoms = pd.DataFrame(records)
    # Apply nullable Int64 casts
    atoms["atom_serial"] = atoms["atom_serial"].astype("Int64")
    atoms["res_seq"] = atoms["res_seq"].astype("Int64")

    return Structure(coords, atoms)


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

def write_pdb(
    structure: Structure,
    filepath: Union[str, Path],
    *,
    renumber: bool = True,
) -> None:
    """Write a :class:`~bioviper2.Structure` to PDB format.

    Parameters
    ----------
    structure : Structure
    filepath : str or pathlib.Path
    renumber : bool, default True
        When ``True`` atoms are renumbered sequentially from 1 within each
        model.  When ``False`` the values in the ``atom_serial`` column are
        written as-is (may collide or exceed 99 999 across models).

    Notes
    -----
    Round-trip fidelity:

    * All per-atom fields in the schema (coordinates, b-factors/pLDDT,
      occupancy, element, alt_loc, icode, charge) round-trip faithfully.
    * TER serial numbers are re-assigned sequentially (original TER serials
      from the input file are not preserved).
    * Header/REMARK/CRYST1/ANISOU/CONECT records are not written.
    * Atom serials > 99 999 are not encoded in hybrid-36 notation; a
      :exc:`ValueError` is raised if ``renumber=False`` and any serial
      exceeds 99 999.
    """
    filepath = Path(filepath)
    df = structure.atoms
    coords = structure.coords
    multi_model = structure.n_models > 1

    with open(filepath, "w") as fh:
        serial_counter = 0  # runs across TER records too

        for model_num in structure.models:
            if multi_model:
                fh.write(f"MODEL     {model_num:4d}\n")

            model_mask = df["model"] == model_num
            model_idx = np.where(model_mask.values)[0]

            prev_chain = None
            for rel, i in enumerate(model_idx):
                row = df.iloc[i]
                xyz = coords[i]

                # Emit TER before first atom of a new chain (polymer only)
                if (
                    prev_chain is not None
                    and row["chain_id"] != prev_chain
                    and not _last_was_hetatm
                ):
                    serial_counter += 1
                    fh.write(
                        f"TER   {serial_counter:5d}      "
                        f"{prev_res_name:<3s} {prev_chain_id}{prev_res_seq:4d}"
                        f"{prev_icode:<1s}\n"
                    )

                serial_counter += 1
                if renumber:
                    serial = serial_counter
                else:
                    serial = int(row["atom_serial"]) if pd.notna(row["atom_serial"]) else serial_counter
                    if serial > 99_999:
                        raise ValueError(
                            f"atom_serial {serial} exceeds 99 999 and hybrid-36 "
                            "encoding is not supported (use renumber=True)"
                        )

                rec = "HETATM" if row["hetero"] else "ATOM  "
                name_field = _format_atom_name(row["atom_name"], row["element"])
                alt_loc = (row["alt_loc"] or " ")[:1]
                res_name = f"{row['res_name']:<3s}"
                chain = (row["chain_id"] or " ")[:1]
                res_seq_val = int(row["res_seq"]) if pd.notna(row["res_seq"]) else 0
                icode = (row["icode"] or " ")[:1]
                elem = f"{row['element']:>2s}" if row["element"] else "  "
                charge = f"{row['charge']:<2s}" if row["charge"] else "  "

                fh.write(
                    f"{rec}{serial:5d} {name_field}{alt_loc}{res_name} "
                    f"{chain}{res_seq_val:4d}{icode}   "
                    f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}"
                    f"{row['occupancy']:6.2f}{row['b_factor']:6.2f}          "
                    f"{elem}{charge}\n"
                )

                # Track for TER emission
                prev_chain = row["chain_id"]
                prev_chain_id = chain
                prev_res_name = res_name.strip()
                prev_res_seq = res_seq_val
                prev_icode = icode
                _last_was_hetatm = bool(row["hetero"])

            # Emit TER at end of chain if last record was ATOM
            if prev_chain is not None and not _last_was_hetatm:
                serial_counter += 1
                fh.write(
                    f"TER   {serial_counter:5d}      "
                    f"{prev_res_name:<3s} {prev_chain_id}{prev_res_seq:4d}"
                    f"{prev_icode:<1s}\n"
                )

            if multi_model:
                fh.write("ENDMDL\n")

        fh.write("END\n")
