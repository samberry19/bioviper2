import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Union

from ..msa import MSA


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _parse_fasta_records(filepath: Union[str, Path]) -> tuple[list[str], list[str], list[str]]:
    """Read a FASTA file and return (ids, descriptions, sequences) as parallel lists."""
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
    Any remaining header text is stored as 'description' in MSA.metadata.

    Raises
    ------
    ValueError
        If the file contains no sequences, or sequences of unequal length
        (which would indicate the file is not a proper alignment).
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
    metadata = pd.DataFrame({"description": descriptions}, index=index)
    return MSA(array, index=index, metadata=metadata)


def write_fasta(msa: MSA, filepath: Union[str, Path], line_width: int = 60) -> None:
    """Write an MSA to a FASTA file.

    The 'description' metadata column, if present, is appended to each header line.
    Sequences are wrapped at *line_width* characters (default 60).
    """
    has_desc = msa.metadata is not None and "description" in msa.metadata.columns
    with open(filepath, "w") as fh:
        for i, seq_id in enumerate(msa.index):
            desc = str(msa.metadata.loc[seq_id, "description"]) if has_desc else ""
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

    Returns a DataFrame indexed by sequence ID with columns:
        sequence    : the full sequence string
        length      : sequence length (convenience; equals len(sequence))
        description : remainder of the FASTA header line after the ID

    Sequences of unequal length are accepted; no alignment is assumed.
    """
    ids, descriptions, seqs = _parse_fasta_records(filepath)
    index = pd.Index(ids, name="id")
    return pd.DataFrame(
        {
            "sequence": seqs,
            "length": [len(s) for s in seqs],
            "description": descriptions,
        },
        index=index,
    )


def write_fasta_sequences(
    df: pd.DataFrame,
    filepath: Union[str, Path],
    seq_col: str = "sequence",
    desc_col: Optional[str] = "description",
    line_width: int = 60,
) -> None:
    """Write a sequence DataFrame to a FASTA file.

    Parameters
    ----------
    df       : DataFrame indexed by sequence ID; must contain *seq_col*.
    seq_col  : column holding the sequence string (default 'sequence').
    desc_col : column to use as the header description, or None to omit.
               Silently ignored if the column is absent.
    line_width: characters per sequence line (default 60); 0 = no wrapping.
    """
    has_desc = desc_col is not None and desc_col in df.columns
    with open(filepath, "w") as fh:
        for seq_id, row in df.iterrows():
            desc = str(row[desc_col]) if has_desc else ""
            header = f">{seq_id}" + (f" {desc}" if desc else "")
            fh.write(header + "\n")
            seq = str(row[seq_col])
            if line_width > 0:
                for start in range(0, len(seq), line_width):
                    fh.write(seq[start : start + line_width] + "\n")
            else:
                fh.write(seq + "\n")
