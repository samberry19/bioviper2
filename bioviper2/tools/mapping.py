"""Alignment↔structure residue mapping.

Maps MSA alignment columns to row indices of a ``Structure``'s CA distance
matrix.  This is the core primitive needed for superposition-free structural
analyses (iRMSD, evolutionary structural analyses, etc.).
"""
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from ..msa import MSA
    from ..structure import Structure

# Gap characters recognised by default (mirrors MSA conventions)
_DEFAULT_GAP_CHARS = ("-", ".")


def map_alignment_to_structure(
    msa: "MSA",
    seq_id,
    structure: "Structure",
    *,
    chain=None,
    model=None,
    strategy: str = "auto",
    return_type: str = "array",
    gap_chars=_DEFAULT_GAP_CHARS,
    **align_kwargs,
):
    """Map each alignment column of *seq_id*'s row to a CA distance-matrix index.

    For each column ``c`` of the MSA, returns the **row index into
    ``structure.distance_matrix('ca').values``** that the aligned residue in
    that column corresponds to.  Columns that are gaps in this row, or whose
    residue is not modelled in the structure, receive the sentinel value
    **``-1``**.

    This is the reusable primitive for structure-guided MSA analyses.  iRMSD,
    contact-map comparisons, and similar metrics all reduce to "index a CA
    distance matrix by aligned columns", which is exactly what this function
    provides.

    Parameters
    ----------
    msa : MSA
        Multiple sequence alignment.
    seq_id :
        Row identifier in *msa* (must exist in ``msa.index``).
    structure : Structure
        The protein structure whose CA distances are to be indexed.
    chain : str or None, optional
        If given, restrict the structure to this chain before mapping.  Use
        when the alignment row represents a single chain of a multi-chain
        structure.
    model : int or None, optional
        If given, restrict the structure to this model number.  Required when
        *structure* contains more than one model and *chain* alone does not
        reduce it to a single model.
    strategy : {'auto', 'positional', 'align'}, default 'auto'
        How to match the ungapped alignment row to the structure residues:

        ``'auto'``
            If the ungapped MSA row sequence exactly equals the structure's
            one-letter CA sequence, use fast positional matching.  Otherwise
            fall back to pairwise global sequence alignment.  This is the
            correct default for the vast majority of real use cases.
        ``'positional'``
            Assume the k-th ungapped character in this MSA row is the k-th CA
            residue of the structure.  Requires equal lengths; raises
            :exc:`ValueError` otherwise.  Fast, but breaks if the structure has
            missing or extra residues relative to the sequence.
        ``'align'``
            Always run a global pairwise sequence alignment (Needleman-Wunsch)
            between the ungapped MSA row and the structure's CA sequence.
            Robust to missing residues, fragments, and point mismatches.
    return_type : {'array', 'series'}, default 'array'
        ``'array'`` → ``numpy.ndarray``, dtype ``int64``, length
        ``msa.n_positions``.
        ``'series'`` → ``pandas.Series`` indexed by ``msa.column_index``,
        named *seq_id*.
    gap_chars : tuple of str, default ``('-', '.')``
        Characters treated as gaps in the MSA row.
    **align_kwargs
        Forwarded to :func:`bioviper2.align` when ``strategy`` requires a
        pairwise alignment.  Useful for overriding ``substitution_matrix``,
        ``gap_open``, or ``gap_extend``.

    Returns
    -------
    numpy.ndarray or pandas.Series
        Integer array of length ``msa.n_positions``.  Entry ``c`` is the
        row index in ``structure.distance_matrix('ca').values`` for the
        residue at alignment column ``c``, or ``-1`` if the column is a gap
        in this row or the residue is absent from the structure.

    Raises
    ------
    KeyError
        If *seq_id* is not in ``msa.index``.
    ValueError
        If the structure has multiple models and *model* is not specified.
        If ``strategy='positional'`` and the sequence lengths differ.

    Examples
    --------
    >>> msa = bv.read("alignment.fasta")
    >>> s   = bv.read_pdb("structure.pdb")
    >>> mapping = bv.map_alignment_to_structure(msa, "seq1", s)
    >>> dm = s.distance_matrix("ca")
    >>> # Distance between columns 10 and 20 of the alignment:
    >>> dm.values[mapping[10], mapping[20]]   # only if both >= 0
    """
    from .align import align as _align

    gap_set = set(gap_chars)

    # ------------------------------------------------------------------
    # 1. Locate the row in the MSA
    # ------------------------------------------------------------------
    try:
        r = msa.index.get_loc(seq_id)
    except KeyError:
        raise KeyError(
            f"seq_id {seq_id!r} is not in the MSA index. "
            f"Available IDs: {list(msa.index[:5])}{'...' if len(msa.index) > 5 else ''}"
        ) from None

    row = msa._array[r]          # 1-D array of single chars, length n_positions
    L = len(row)

    # ------------------------------------------------------------------
    # 2. Non-gap columns of this row
    # ------------------------------------------------------------------
    cols = np.where(~np.isin(row, list(gap_set)))[0]  # alignment cols with residues
    aln_seq = "".join(row[cols])                        # ungapped sequence

    # ------------------------------------------------------------------
    # 3. Resolve the structure view (chain / model)
    # ------------------------------------------------------------------
    view = structure
    if chain is not None:
        view = view.select(chain=chain)
    if model is not None:
        view = view.select(model=model)
    elif view.n_models > 1:
        raise ValueError(
            f"Structure has {view.n_models} models. Pass model=<int> or "
            "pre-select a model with .select(model=<int>)."
        )

    # struct_seq is 1:1 with CA-matrix row indices 0..n_ca-1
    struct_seq = view.sequence(ca_only=True)
    n_ca = len(struct_seq)

    # ------------------------------------------------------------------
    # 4. Build residue → CA-row mapping
    # ------------------------------------------------------------------
    out = np.full(L, -1, dtype=np.int64)

    if len(aln_seq) == 0 or n_ca == 0:
        # Nothing to map
        pass
    else:
        actual_strategy = strategy
        if strategy == "auto":
            actual_strategy = "positional" if aln_seq == struct_seq else "align"

        if actual_strategy == "positional":
            if len(aln_seq) != n_ca:
                raise ValueError(
                    f"strategy='positional' requires equal lengths, but the "
                    f"ungapped alignment row has {len(aln_seq)} residues while "
                    f"the structure has {n_ca} CA residues.  Use "
                    f"strategy='align' to handle missing/extra residues."
                )
            # k-th ungapped residue → CA row k
            for k, c in enumerate(cols):
                out[c] = k

        else:  # "align"
            result_msa = _align(
                aln_seq, struct_seq,
                mode="global",
                seq1_id="query",
                seq2_id="struct",
                **align_kwargs,
            )
            arr = result_msa._array  # shape (2, L_aln); row 0 = aln_seq, row 1 = struct
            aln_gap = "-"            # align() inserts '-' as gap

            i = 0   # index into aln_seq ungapped residues → maps to cols[i]
            j = 0   # index into struct CA rows
            for ci in range(arr.shape[1]):
                c0 = arr[0, ci]
                c1 = arr[1, ci]
                q_is_gap = (c0 == aln_gap)
                s_is_gap = (c1 == aln_gap)

                if not q_is_gap and not s_is_gap:
                    # Both ungapped → residue i of aln_seq ↔ CA row j
                    out[cols[i]] = j
                    i += 1
                    j += 1
                elif not q_is_gap:
                    # Gap in struct: residue i of aln_seq has no structural counterpart
                    i += 1
                elif not s_is_gap:
                    # Gap in aln: extra residue in struct (not in alignment)
                    j += 1
                # Both gaps: shouldn't happen; skip

    # ------------------------------------------------------------------
    # 5. Return
    # ------------------------------------------------------------------
    if return_type == "series":
        return pd.Series(out, index=msa.column_index, name=seq_id)
    return out
