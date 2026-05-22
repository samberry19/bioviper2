import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union

from ..msa import MSA

_BLOCK_WIDTH = 60
_MIN_ID_COL = 16  # minimum ID column width to match standard Clustal layout


def read_clustal(filepath: Union[str, Path]) -> MSA:
    """Parse a Clustal-format alignment file (.aln / .clw).

    Handles multi-block interleaved format. Conservation lines (lines whose
    first non-space character is *, :, or .) are skipped automatically.
    Sequence IDs are taken from the first whitespace-delimited token on each
    sequence line.
    """
    seq_parts: dict[str, list[str]] = {}
    seq_order: list[str] = []
    header_seen = False

    with open(filepath) as fh:
        for line in fh:
            line = line.rstrip("\n")

            if not header_seen:
                if line.upper().startswith("CLUSTAL"):
                    header_seen = True
                continue

            if not line.strip():
                continue

            stripped = line.lstrip()
            if stripped and stripped[0] in "*.:-":
                continue

            parts = line.split(None, 1)
            if len(parts) == 2:
                seq_id, seq = parts[0], parts[1].strip().split()[0]
                if seq_id not in seq_parts:
                    seq_parts[seq_id] = []
                    seq_order.append(seq_id)
                seq_parts[seq_id].append(seq)

    if not seq_order:
        raise ValueError(f"No sequences found in {filepath}")

    seqs = ["".join(seq_parts[sid]) for sid in seq_order]
    lengths = {len(s) for s in seqs}
    if len(lengths) > 1:
        raise ValueError(
            f"Sequences have unequal lengths ({min(lengths)}–{max(lengths)}) "
            "after concatenating blocks"
        )

    n, L = len(seqs), next(iter(lengths))
    array = np.empty((n, L), dtype="U1")
    for i, seq in enumerate(seqs):
        array[i] = list(seq)

    index = pd.Index(seq_order, name="id")
    return MSA(array, index=index, metadata=None)


def write_clustal(
    msa: MSA,
    filepath: Union[str, Path],
    line_width: int = _BLOCK_WIDTH,
) -> None:
    """Write an MSA to Clustal format.

    Conservation line uses '*' for fully conserved positions (excluding gap-only
    columns) and ' ' elsewhere. Metadata is not written — Clustal format has no
    standard mechanism for per-sequence metadata.
    """
    id_width = max(_MIN_ID_COL, max(len(str(sid)) for sid in msa.index) + 2)
    seqs = ["".join(msa._array[i]) for i in range(msa.n_seqs)]

    with open(filepath, "w") as fh:
        fh.write("CLUSTAL W multiple sequence alignment\n\n\n")

        for block_start in range(0, msa.n_positions, line_width):
            for seq_id, seq in zip(msa.index, seqs):
                chunk = seq[block_start : block_start + line_width]
                fh.write(f"{str(seq_id):<{id_width}}{chunk}\n")

            # Conservation line
            col_slice = msa._array[:, block_start : block_start + line_width]
            cons = _conservation_line(col_slice)
            fh.write(f"{' ' * id_width}{cons}\n\n")


def _conservation_line(col_array: np.ndarray) -> str:
    """Return a conservation string for a (n_seqs × block_width) character array."""
    result = []
    for col in col_array.T:
        chars = set(col)
        if len(chars) == 1 and "-" not in chars and "." not in chars:
            result.append("*")
        else:
            result.append(" ")
    return "".join(result)
