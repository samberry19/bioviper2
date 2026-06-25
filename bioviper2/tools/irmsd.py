"""Superposition-free iRMSD metric for comparing alignments by structural agreement.

Implements the APDB-family iRMSD (Armougom, Moretti, Keduas & Notredame 2006;
T-Coffee).  Lower = better structural agreement.  No superposition is required:
intramolecular Cα–Cα distances are rotation/translation invariant.

Reference:
    Armougom F, Moretti S, Keduas V, Notredame C (2006)
    "The iRMSD: a local measure of sequence alignment accuracy using structural
    information."  Bioinformatics 22(14):e35–e39.
    https://doi.org/10.1093/bioinformatics/btl218
"""
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .mapping import map_alignment_to_structure

if TYPE_CHECKING:
    from ..msa import MSA
    from ..structure import Structure


class IRMSDResult:
    """Result of :func:`irmsd`.

    Attributes
    ----------
    global_ : float
        Global iRMSD: ``sqrt(ΣS / N)`` pooled over all structure pairs and all
        evaluated residue-pair comparisons.  ``numpy.nan`` when no pairs could
        be evaluated.
    per_column : pandas.Series
        Per-alignment-column local iRMSD, indexed by ``msa.column_index``.
        Each column's score is ``sqrt(S_c / N_c)`` where the sum runs over all
        structure pairs and all partner columns within the neighbourhood radius
        of that column.  ``numpy.nan`` at gap-only columns or columns with no
        evaluated pairs.
    pairwise : pandas.DataFrame
        Symmetric matrix of per-pair iRMSD values, shape ``(n_structs,
        n_structs)``, indexed and columned by the sequence IDs of the
        structures used.  Diagonal entries are ``numpy.nan``.
    n_pairs_evaluated : int
        Total number of (column_i, column_j) × structure-pair comparisons
        included in the global score.
    seq_ids : list
        The sequence IDs for which both an MSA row and a structure were
        available.
    """

    def __init__(
        self,
        global_: float,
        per_column: pd.Series,
        pairwise: pd.DataFrame,
        n_pairs_evaluated: int,
        seq_ids: list,
    ):
        self.global_ = global_
        self.per_column = per_column
        self.pairwise = pairwise
        self.n_pairs_evaluated = n_pairs_evaluated
        self.seq_ids = seq_ids

    def __repr__(self) -> str:
        n = len(self.seq_ids)
        g = f"{self.global_:.4f}" if not np.isnan(self.global_) else "nan"
        return (
            f"IRMSDResult(global={g}, {n} structure{'s' if n != 1 else ''}, "
            f"{self.n_pairs_evaluated} pairs evaluated)"
        )


def irmsd(
    msa: "MSA",
    structures: dict,
    *,
    radius: float = 10.0,
    mode: str = "ca",
    strategy: str = "auto",
    chains: dict = None,
    models: dict = None,
    gap_chars=("-", "."),
    **align_kwargs,
) -> IRMSDResult:
    """Compute the superposition-free iRMSD metric.

    For each pair of structures associated with aligned sequences, compare
    intramolecular Cα–Cα distances between residue pairs that the alignment
    makes equivalent.  Only pairs within a neighbourhood sphere of radius
    *radius* are counted (the single parameter of the metric, default 10 Å).

    The global iRMSD is ``sqrt(ΣS / N)`` where the sum pools squared distance
    deviations across **all** structure pairs and **all** evaluated residue
    pairs — it is NOT the mean of the per-pair iRMSD scores.  Lower = better
    alignment–structure agreement.

    Parameters
    ----------
    msa : MSA
        Multiple sequence alignment.  Rows whose sequence ID does not appear
        in *structures* are silently ignored.
    structures : dict
        Mapping ``{seq_id: Structure}`` associating each aligned sequence with
        its protein structure.  Entries whose seq_id does not appear in the MSA
        trigger a warning and are ignored.
    radius : float, default 10.0
        Neighbourhood sphere radius in Ångstroms.  Only Cα pairs whose mean
        distance (averaged over the two structures) is ≤ *radius* contribute to
        the score.  The T-Coffee default is 10 Å.
    mode : str, default 'ca'
        Distance-matrix mode passed to :meth:`Structure.distance_matrix`.
        ``'ca'`` is the standard choice for iRMSD.
    strategy : {'auto', 'positional', 'align'}, default 'auto'
        Alignment↔structure residue mapping strategy.  See
        :func:`map_alignment_to_structure` for details.
    chains : dict or None, optional
        ``{seq_id: chain_id}`` — restrict each structure to the given chain
        before mapping.  Useful for multi-chain structures where only one chain
        corresponds to the aligned sequence.
    models : dict or None, optional
        ``{seq_id: model_int}`` — restrict each structure to the given model.
        Required for structures with more than one model; raises
        :exc:`ValueError` if a multi-model structure has no entry here.
    gap_chars : tuple of str, default ``('-', '.')``
        Characters treated as gaps in the MSA.
    **align_kwargs
        Forwarded to :func:`bioviper2.align` when *strategy* requires a
        pairwise alignment.

    Returns
    -------
    IRMSDResult
        Object with attributes ``global_``, ``per_column``, ``pairwise``,
        ``n_pairs_evaluated``, and ``seq_ids``.  See :class:`IRMSDResult`.

    Raises
    ------
    ValueError
        If a structure in *structures* has multiple models and the corresponding
        entry is absent from *models*.

    Examples
    --------
    >>> msa = bv.read("alignment.fasta")
    >>> structures = {
    ...     "seq1": bv.read_pdb("s1.pdb"),
    ...     "seq2": bv.read_pdb("s2.pdb"),
    ... }
    >>> result = bv.irmsd(msa, structures)
    >>> result.global_
    1.23
    >>> result.per_column.plot()           # local structural support per column
    >>> result.pairwise                    # seq_id × seq_id DataFrame
    """
    # ------------------------------------------------------------------
    # Determine which seq_ids to use (intersection of MSA rows ∩ structures)
    # ------------------------------------------------------------------
    msa_id_set = set(msa.index)
    struct_id_set = set(structures.keys())

    ignored_in_struct = struct_id_set - msa_id_set
    if ignored_in_struct:
        warnings.warn(
            f"The following seq_ids are in `structures` but not in the MSA "
            f"and will be ignored: {sorted(str(x) for x in ignored_in_struct)}"
        )

    # Preserve MSA row order for deterministic results
    common_ids = [sid for sid in msa.index if sid in struct_id_set]

    if len(common_ids) < 2:
        warnings.warn(
            "Fewer than 2 structures overlap with the MSA; returning NaN result."
        )
        dummy_pc = pd.Series(
            np.full(msa.n_positions, np.nan),
            index=msa.column_index,
            name="per_column_irmsd",
        )
        dummy_pw = pd.DataFrame(
            np.full((len(common_ids), len(common_ids)), np.nan),
            index=common_ids, columns=common_ids,
        )
        return IRMSDResult(
            global_=np.nan,
            per_column=dummy_pc,
            pairwise=dummy_pw,
            n_pairs_evaluated=0,
            seq_ids=common_ids,
        )

    # ------------------------------------------------------------------
    # Precompute filtered structure views and CA distance matrices
    # ------------------------------------------------------------------
    views = {}
    dms = {}
    for sid in common_ids:
        chain = None if chains is None else chains.get(sid)
        model_val = None if models is None else models.get(sid)
        struct = structures[sid]

        if chain is not None:
            struct = struct.select(chain=chain)
        if model_val is not None:
            struct = struct.select(model=model_val)
        elif struct.n_models > 1:
            raise ValueError(
                f"Structure for '{sid}' has {struct.n_models} models. "
                f"Pass models={{'{sid}': <model_int>}} to select one, or "
                "pre-select with .select(model=<int>)."
            )
        views[sid] = struct
        dms[sid] = struct.distance_matrix(mode).values

    # ------------------------------------------------------------------
    # Precompute alignment↔structure mappings
    # ------------------------------------------------------------------
    maps = {}
    for sid in common_ids:
        maps[sid] = map_alignment_to_structure(
            msa, sid, views[sid],
            strategy=strategy,
            gap_chars=gap_chars,
            **align_kwargs,
        )

    # ------------------------------------------------------------------
    # Accumulate over all unordered structure pairs
    # ------------------------------------------------------------------
    n_pos = msa.n_positions
    total_S = 0.0
    total_N = 0
    col_sq_global = np.zeros(n_pos, dtype=np.float64)
    col_N_global = np.zeros(n_pos, dtype=np.int64)

    n_structs = len(common_ids)
    pw_S = np.zeros((n_structs, n_structs), dtype=np.float64)
    pw_N = np.zeros((n_structs, n_structs), dtype=np.int64)

    for idx_a in range(n_structs):
        A = common_ids[idx_a]
        mA = maps[A]
        dmA = dms[A]

        for idx_b in range(idx_a + 1, n_structs):
            B = common_ids[idx_b]
            mB = maps[B]
            dmB = dms[B]

            # Valid alignment columns: non-sentinel in both maps
            V_arr = np.where((mA >= 0) & (mB >= 0))[0]
            if len(V_arr) < 2:
                continue

            ia = mA[V_arr]   # CA-row indices into dmA
            ib = mB[V_arr]   # CA-row indices into dmB

            DA = dmA[np.ix_(ia, ia)]
            DB = dmB[np.ix_(ib, ib)]
            sq = (DA - DB) ** 2
            mean_D = 0.5 * (DA + DB)

            # Neighbourhood: pairs within *radius* (diagonal = self-pair, excluded)
            in_radius = mean_D <= radius
            np.fill_diagonal(in_radius, False)

            # ── global accumulation (upper triangle avoids double-counting) ──
            mask_upper = np.triu(in_radius, k=1)
            pair_S = float((sq * mask_upper).sum())
            pair_N = int(mask_upper.sum())
            total_S += pair_S
            total_N += pair_N
            pw_S[idx_a, idx_b] = pair_S
            pw_S[idx_b, idx_a] = pair_S
            pw_N[idx_a, idx_b] = pair_N
            pw_N[idx_b, idx_a] = pair_N

            # ── per-column accumulation (full matrix; attribute to both ends) ──
            col_sq_pair = (sq * in_radius).sum(axis=1)   # shape: (n_V,)
            col_N_pair = in_radius.sum(axis=1).astype(np.int64)
            np.add.at(col_sq_global, V_arr, col_sq_pair)
            np.add.at(col_N_global, V_arr, col_N_pair)

    # ------------------------------------------------------------------
    # Compute final outputs
    # ------------------------------------------------------------------
    if total_N == 0:
        warnings.warn(
            "No residue pairs could be evaluated (all column pairs were either "
            "unaligned, unmodelled, or beyond the radius); returning NaN."
        )
        global_ = np.nan
    else:
        global_ = float(np.sqrt(total_S / total_N))

    # Per-column
    pc_values = np.where(
        col_N_global > 0,
        np.sqrt(col_sq_global / np.where(col_N_global > 0, col_N_global, 1.0)),
        np.nan,
    )
    per_column = pd.Series(
        pc_values, index=msa.column_index, name="per_column_irmsd"
    )

    # Pairwise DataFrame (symmetric, NaN on diagonal)
    pw_matrix = np.where(
        pw_N > 0,
        np.sqrt(pw_S / np.where(pw_N > 0, pw_N, 1)),
        np.nan,
    )
    np.fill_diagonal(pw_matrix, np.nan)
    pairwise = pd.DataFrame(pw_matrix, index=common_ids, columns=common_ids)

    return IRMSDResult(
        global_=global_,
        per_column=per_column,
        pairwise=pairwise,
        n_pairs_evaluated=total_N,
        seq_ids=common_ids,
    )
