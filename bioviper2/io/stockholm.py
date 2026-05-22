import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union

from ..msa import MSA


def read_stockholm(filepath: Union[str, Path]) -> MSA:
    """Parse a Stockholm-format alignment file.

    Per-sequence annotations (#=GS lines) are stored in MSA.metadata.
    File-level annotations (#=GF) are ignored; column/residue annotations
    (#=GC, #=GR) are silently skipped.

    Handles both single-block and interleaved (multi-block) formats.
    """
    seq_parts: dict[str, list[str]] = {}
    seq_order: list[str] = []
    gs_meta: dict[str, dict[str, str]] = {}

    with open(filepath) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line or line.startswith("# STOCKHOLM") or line == "//":
                continue

            if line.startswith("#=GS"):
                parts = line.split(None, 3)
                if len(parts) >= 3:
                    seq_id, tag = parts[1], parts[2]
                    value = parts[3] if len(parts) > 3 else ""
                    gs_meta.setdefault(seq_id, {})[tag] = value
                continue

            if line.startswith("#"):
                continue

            parts = line.split(None, 1)
            if len(parts) == 2:
                seq_id, seq = parts
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

    if gs_meta:
        all_tags = sorted({tag for tags in gs_meta.values() for tag in tags})
        meta_df = pd.DataFrame(
            {tag: [gs_meta.get(sid, {}).get(tag, "") for sid in seq_order] for tag in all_tags},
            index=index,
        )
    else:
        meta_df = None

    return MSA(array, index=index, metadata=meta_df)


def write_stockholm(msa: MSA, filepath: Union[str, Path], line_width: int = 0) -> None:
    """Write an MSA to Stockholm format.

    Metadata columns are written as #=GS annotations. When *line_width* > 0
    the alignment is split into interleaved blocks of that width (default: no
    wrapping, one sequence per line).
    """
    id_width = max(len(str(sid)) for sid in msa.index)

    with open(filepath, "w") as fh:
        fh.write("# STOCKHOLM 1.0\n\n")

        if msa.metadata is not None:
            for seq_id in msa.index:
                for tag in msa.metadata.columns:
                    val = msa.metadata.loc[seq_id, tag]
                    if pd.notna(val) and str(val):
                        fh.write(f"#=GS {seq_id} {tag} {val}\n")
            fh.write("\n")

        seqs = ["".join(msa._array[i]) for i in range(msa.n_seqs)]

        if line_width > 0:
            for block_start in range(0, msa.n_positions, line_width):
                for seq_id, seq in zip(msa.index, seqs):
                    chunk = seq[block_start : block_start + line_width]
                    fh.write(f"{str(seq_id):<{id_width}}  {chunk}\n")
                fh.write("\n")
        else:
            for seq_id, seq in zip(msa.index, seqs):
                fh.write(f"{str(seq_id):<{id_width}}  {seq}\n")

        fh.write("//\n")
