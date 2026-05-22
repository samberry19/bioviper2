import re
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Union

from ..msa import MSA

# ---------------------------------------------------------------------------
# Header description parsing
# ---------------------------------------------------------------------------

# Matches word-boundary + identifier + '=', e.g. OS=, OX=, n=, TaxID=
_KV_PATTERN = re.compile(r'\b([A-Za-z][A-Za-z0-9_]*)=')


def _parse_header_description(desc: str) -> dict[str, str]:
    """Parse a FASTA header description into structured fields.

    Handles UniProt / UniRef style key=value annotations:
      'Some protein OS=Homo sapiens OX=9606 GN=PROT PE=1 SV=2'
      'Cluster n=10 Tax=Homo sapiens TaxID=9606 RepID=A0A000_HUMAN'

    Free text before the first key=value pair is stored under 'description'.
    If no key=value pairs are found the whole string is stored as 'description'.
    Values are left as strings; callers can cast as needed.
    """
    matches = list(_KV_PATTERN.finditer(desc))

    if not matches:
        return {"description": desc.strip()} if desc.strip() else {}

    result: dict[str, str] = {}

    # Free-text prefix before the first key=
    pre = desc[: matches[0].start()].strip()
    if pre:
        result["description"] = pre

    for i, m in enumerate(matches):
        key = m.group(1)
        val_start = m.end()
        val_end = matches[i + 1].start() if i + 1 < len(matches) else len(desc)
        result[key] = desc[val_start:val_end].strip()

    return result


def _metadata_row_to_header_desc(row: pd.Series) -> str:
    """Reconstruct a FASTA header description string from a metadata row.

    'description' is written as free text first; all other fields follow as
    KEY=VALUE pairs.  NaN / empty values are skipped.
    """
    parts: list[str] = []
    if "description" in row.index:
        val = row["description"]
        if pd.notna(val) and str(val).strip():
            parts.append(str(val).strip())
    for col in row.index:
        if col == "description":
            continue
        val = row[col]
        if pd.notna(val) and str(val).strip():
            parts.append(f"{col}={val}")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_fasta_records(filepath: Union[str, Path]) -> tuple[list[str], list[str], list[str]]:
    """Read a FASTA file and return (ids, raw_descriptions, sequences)."""
    ids: list[str] = []
    descriptions: list[str] = []
    seqs: list[str] = []

    current_id: Optional[str] = None
    current_desc: str = ""
    current_parts: list[str] = []

    with open(filepath) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    seqs.append("".join(current_parts))
                header = line[1:].strip()
                parts = header.split(None, 1)
                current_id = parts[0]
                current_desc = parts[1] if len(parts) > 1 else ""
                ids.append(current_id)
                descriptions.append(current_desc)
                current_parts = []
            elif current_id is not None:
                current_parts.append(line.strip())

        if current_id is not None:
            seqs.append("".join(current_parts))

    if not seqs:
        raise ValueError(f"No sequences found in {filepath}")

    return ids, descriptions, seqs


# ---------------------------------------------------------------------------
# Aligned sequences → MSA
# ---------------------------------------------------------------------------

def read_fasta(filepath: Union[str, Path]) -> MSA:
    """Parse a FASTA file containing a multiple sequence alignment.

    Sequence IDs are the first whitespace-delimited token on each header line.
    The remainder of each header is parsed for KEY=VALUE fields (UniProt /
    UniRef style); free text before the first key becomes 'description'.
    All parsed fields are stored as columns of MSA.metadata.

    Raises ValueError if sequences have unequal lengths.
    """
    ids, descriptions, seqs = _parse_fasta_records(filepath)

    lengths = {len(s) for s in seqs}
    if len(lengths) > 1:
        raise ValueError(
            f"Sequences have unequal lengths ({min(lengths)}–{max(lengths)}); "
            "file does not appear to be a multiple sequence alignment"
        )

    n, L = len(seqs), next(iter(lengths))
    array = np.empty((n, L), dtype="U1")
    for i, seq in enumerate(seqs):
        array[i] = list(seq)

    index = pd.Index(ids, name="id")
    parsed = [_parse_header_description(d) for d in descriptions]
    metadata = pd.DataFrame(parsed, index=index)

    return MSA(array, index=index, metadata=metadata if not metadata.empty else None)


def write_fasta(msa: MSA, filepath: Union[str, Path], line_width: int = 60) -> None:
    """Write an MSA to a FASTA file.

    All metadata columns are reconstructed into the header line: 'description'
    appears as free text, all other columns as KEY=VALUE pairs.
    Sequences are wrapped at *line_width* characters (default 60).
    """
    with open(filepath, "w") as fh:
        for i, seq_id in enumerate(msa.index):
            if msa.metadata is not None:
                desc = _metadata_row_to_header_desc(msa.metadata.loc[seq_id])
            else:
                desc = ""
            header = f">{seq_id}" + (f" {desc}" if desc else "")
            fh.write(header + "\n")
            seq = "".join(msa._array[i])
            for start in range(0, len(seq), line_width):
                fh.write(seq[start : start + line_width] + "\n")


# ---------------------------------------------------------------------------
# Unaligned sequences → DataFrame
# ---------------------------------------------------------------------------

def read_fasta_sequences(filepath: Union[str, Path]) -> pd.DataFrame:
    """Parse a FASTA file of unaligned sequences into a DataFrame.

    Returns a DataFrame indexed by sequence ID.  Columns are:
        sequence    : full sequence string
        length      : sequence length
        + one column per KEY=VALUE field found in any header line
          (e.g. OS, OX, GN, Tax, TaxID — NaN where absent)
        description : free-text prefix of the header, if present

    Sequences of unequal length are accepted; no alignment is assumed.
    """
    ids, descriptions, seqs = _parse_fasta_records(filepath)
    index = pd.Index(ids, name="id")

    parsed = [_parse_header_description(d) for d in descriptions]
    meta = pd.DataFrame(parsed, index=index)  # columns vary per file

    df = pd.DataFrame({"sequence": seqs, "length": [len(s) for s in seqs]}, index=index)
    if not meta.empty:
        df = pd.concat([df, meta], axis=1)
    return df


def write_fasta_sequences(
    df: pd.DataFrame,
    filepath: Union[str, Path],
    seq_col: str = "sequence",
    line_width: int = 60,
) -> None:
    """Write a sequence DataFrame to a FASTA file.

    Parameters
    ----------
    df        : DataFrame indexed by sequence ID; must contain *seq_col*.
    seq_col   : column holding the sequence string (default 'sequence').
    line_width: characters per sequence line (default 60); 0 = no wrapping.

    All columns except *seq_col* and 'length' are included in the header.
    'description' is written as free text; all other columns as KEY=VALUE.
    """
    skip = {seq_col, "length"}
    meta_cols = [c for c in df.columns if c not in skip]

    with open(filepath, "w") as fh:
        for seq_id, row in df.iterrows():
            desc = _metadata_row_to_header_desc(row[meta_cols]) if meta_cols else ""
            header = f">{seq_id}" + (f" {desc}" if desc else "")
            fh.write(header + "\n")
            seq = str(row[seq_col])
            if line_width > 0:
                for start in range(0, len(seq), line_width):
                    fh.write(seq[start : start + line_width] + "\n")
            else:
                fh.write(seq + "\n")
