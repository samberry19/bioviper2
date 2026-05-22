import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union

from ..msa import MSA


def read_a3m(filepath: Union[str, Path]) -> MSA:
    """Parse an A3M-format file (used by mmseqs2, ColabFold, AlphaFold pipelines).

    A3M is a FASTA-like format where lowercase characters represent insertions
    relative to the query (first) sequence. This reader strips all lowercase
    characters from every sequence, yielding a proper alignment of length equal
    to the query length.

    Sequence IDs are the first whitespace token of each header; the rest is
    stored as 'description' in MSA.metadata.
    """
    ids: list[str] = []
    descriptions: list[str] = []
    raw_seqs: list[str] = []

    current_id: str | None = None
    current_desc: str = ""
    current_parts: list[str] = []

    with open(filepath) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    raw_seqs.append("".join(current_parts))
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
            raw_seqs.append("".join(current_parts))

    if not raw_seqs:
        raise ValueError(f"No sequences found in {filepath}")

    # Strip lowercase (insertions) from all sequences
    seqs = [_strip_insertions(s) for s in raw_seqs]

    lengths = {len(s) for s in seqs}
    if len(lengths) > 1:
        raise ValueError(
            f"After removing insertions, sequences have unequal lengths "
            f"({min(lengths)}–{max(lengths)}). The file may not be valid A3M."
        )

    n, L = len(seqs), next(iter(lengths))
    array = np.empty((n, L), dtype="U1")
    for i, seq in enumerate(seqs):
        array[i] = list(seq)

    index = pd.Index(ids, name="id")
    metadata = pd.DataFrame({"description": descriptions}, index=index)
    return MSA(array, index=index, metadata=metadata)


def _strip_insertions(seq: str) -> str:
    """Remove lowercase characters (insertions relative to query) from a sequence."""
    return "".join(c for c in seq if not c.islower())


def write_a3m(msa: MSA, filepath: Union[str, Path], line_width: int = 60) -> None:
    """Write an MSA to A3M format.

    Since insertion information is not stored in MSA, this writes sequences in
    uppercase (no lowercase insertions) — equivalent to writing FASTA. The first
    sequence is treated as the query.
    """
    from .fasta import write_fasta
    write_fasta(msa, filepath, line_width=line_width)
