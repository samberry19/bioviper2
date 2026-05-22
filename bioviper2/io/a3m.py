import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union

from ..msa import MSA
from .fasta import _parse_fasta_records, _parse_header_description


def read_a3m(filepath: Union[str, Path]) -> MSA:
    """Parse an A3M-format file (used by mmseqs2, ColabFold, AlphaFold pipelines).

    A3M is a FASTA-like format where lowercase characters represent insertions
    relative to the query (first) sequence.  This reader strips all lowercase
    characters from every sequence, yielding a proper alignment whose length
    equals that of the query.

    Header descriptions are parsed for KEY=VALUE fields (same as read_fasta).
    """
    ids, descriptions, raw_seqs = _parse_fasta_records(filepath)

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
    parsed = [_parse_header_description(d) for d in descriptions]
    metadata = pd.DataFrame(parsed, index=index)

    return MSA(array, index=index, metadata=metadata if not metadata.empty else None)


def _strip_insertions(seq: str) -> str:
    """Remove lowercase characters (insertions relative to query) from a sequence."""
    return "".join(c for c in seq if not c.islower())


def write_a3m(msa: MSA, filepath: Union[str, Path], line_width: int = 60) -> None:
    """Write an MSA to A3M format.

    Since insertion information is not stored in MSA, this writes sequences in
    uppercase (no lowercase insertions) — equivalent to writing FASTA.
    """
    from .fasta import write_fasta
    write_fasta(msa, filepath, line_width=line_width)
