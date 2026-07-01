"""Pairwise sequence alignment returning a 2-sequence MSA.

Implements global (Needleman-Wunsch) and local (Smith-Waterman) alignment
with affine gap penalties in pure numpy/Python — no external dependencies.

Typical usage::

    msa = bioviper2.align("ACDEFGHIKL", "ACDF-GHIKL")
    msa = bioviper2.align(seq_df.loc["prot1", "sequence"],
                          seq_df.loc["prot2", "sequence"],
                          substitution_matrix="blosum62")
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd

from ..msa import MSA

# ---------------------------------------------------------------------------
# Substitution matrices  (defined in bioviper2.matrices, imported here)
# ---------------------------------------------------------------------------

from ..matrices import _MATRICES

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_NEG_INF = -1e9


def _to_str(seq, hint_id: Optional[str]) -> tuple[str, str]:
    """Return (sequence_string, id) from various input types."""
    if isinstance(seq, pd.Series):
        seq_id = hint_id or (str(seq.name) if seq.name is not None else "seq")
        return "".join(seq.astype(str)), seq_id
    if isinstance(seq, str):
        return seq, hint_id or "seq"
    raise TypeError(f"seq must be str or pd.Series, got {type(seq).__name__}")


def _score_matrix(s1: str, s2: str, sub: dict) -> np.ndarray:
    """Precompute (n, m) float32 substitution score matrix."""
    n, m = len(s1), len(s2)
    mat = np.empty((n, m), dtype=np.float32)
    default = -4
    for i, a in enumerate(s1):
        for j, b in enumerate(s2):
            mat[i, j] = sub.get((a, b), default)
    return mat


def _nw_dp(s1: str, s2: str, scores: np.ndarray,
           gap_open: float, gap_extend: float):
    """Needleman-Wunsch DP with affine gap penalties.

    gap_open   : penalty charged when starting a new gap (< 0)
    gap_extend : penalty per gap character (< 0)

    A gap of length k costs: gap_open + k * gap_extend.

    Returns (M, Ix, Iy, tb_M, tb_Ix, tb_Iy) all (n+1, m+1) arrays.
    tb_* encode the predecessor state: 0=M, 1=Ix, 2=Iy.
    """
    n, m = len(s1), len(s2)
    M  = np.full((n + 1, m + 1), _NEG_INF, dtype=np.float64)
    Ix = np.full((n + 1, m + 1), _NEG_INF, dtype=np.float64)
    Iy = np.full((n + 1, m + 1), _NEG_INF, dtype=np.float64)
    tb_M  = np.zeros((n + 1, m + 1), dtype=np.int8)
    tb_Ix = np.zeros((n + 1, m + 1), dtype=np.int8)
    tb_Iy = np.zeros((n + 1, m + 1), dtype=np.int8)

    # Initialise borders
    M[0, 0] = 0.0
    for i in range(1, n + 1):
        Ix[i, 0] = gap_open + i * gap_extend
    for j in range(1, m + 1):
        Iy[0, j] = gap_open + j * gap_extend

    go = gap_open
    ge = gap_extend

    for i in range(1, n + 1):
        s_row = scores[i - 1]   # (m,) score row for seq1[i-1]
        prev_M  = M[i - 1]
        prev_Ix = Ix[i - 1]
        prev_Iy = Iy[i - 1]

        # --- M[i, 1:] = max(prev_M[:-1], prev_Ix[:-1], prev_Iy[:-1]) + s ---
        pm  = prev_M[:-1]
        pix = prev_Ix[:-1]
        piy = prev_Iy[:-1]
        best_prev = np.maximum(np.maximum(pm, pix), piy)
        M[i, 1:] = best_prev + s_row
        # traceback: which predecessor was best?
        tb_M[i, 1:] = np.where(pm >= np.maximum(pix, piy), 0,
                                np.where(pix >= piy, 1, 2)).astype(np.int8)

        # --- Ix[i, 1:] = max(prev_M[1:] + go, prev_Ix[1:]) + ge -----------
        ix_open = prev_M[1:] + go
        ix_ext  = prev_Ix[1:]
        open_better = ix_open >= ix_ext
        Ix[i, 1:] = np.where(open_better, ix_open, ix_ext) + ge
        tb_Ix[i, 1:] = np.where(open_better, np.int8(0), np.int8(1))

        # --- Iy[i, j] (sequential j scan: depends on Iy[i, j-1]) ----------
        Iy_row = Iy[i]
        M_row  = M[i]
        for j in range(1, m + 1):
            o = M_row[j - 1] + go + ge
            e = Iy_row[j - 1] + ge
            if o >= e:
                Iy_row[j] = o
                tb_Iy[i, j] = 0
            else:
                Iy_row[j] = e
                tb_Iy[i, j] = 2

    return M, Ix, Iy, tb_M, tb_Ix, tb_Iy


def _sw_dp(s1: str, s2: str, scores: np.ndarray,
           gap_open: float, gap_extend: float):
    """Smith-Waterman DP with affine gap penalties.

    Differences from NW: scores floored at 0, borders initialised to 0.
    """
    n, m = len(s1), len(s2)
    M  = np.zeros((n + 1, m + 1), dtype=np.float64)
    Ix = np.zeros((n + 1, m + 1), dtype=np.float64)
    Iy = np.zeros((n + 1, m + 1), dtype=np.float64)
    tb_M  = np.zeros((n + 1, m + 1), dtype=np.int8)
    tb_Ix = np.zeros((n + 1, m + 1), dtype=np.int8)
    tb_Iy = np.zeros((n + 1, m + 1), dtype=np.int8)

    go, ge = gap_open, gap_extend

    for i in range(1, n + 1):
        s_row   = scores[i - 1]
        prev_M  = M[i - 1]
        prev_Ix = Ix[i - 1]
        prev_Iy = Iy[i - 1]

        # M
        pm  = prev_M[:-1]
        pix = prev_Ix[:-1]
        piy = prev_Iy[:-1]
        best_prev = np.maximum(np.maximum(pm, pix), piy)
        raw = best_prev + s_row
        M[i, 1:] = np.maximum(raw, 0.0)
        tb_M[i, 1:] = np.where(pm >= np.maximum(pix, piy), 0,
                                np.where(pix >= piy, 1, 2)).astype(np.int8)

        # Ix
        ix_open = prev_M[1:] + go
        ix_ext  = prev_Ix[1:]
        open_better = ix_open >= ix_ext
        raw_ix = np.where(open_better, ix_open, ix_ext) + ge
        Ix[i, 1:] = np.maximum(raw_ix, 0.0)
        tb_Ix[i, 1:] = np.where(open_better, np.int8(0), np.int8(1))

        # Iy
        Iy_row = Iy[i]
        M_row  = M[i]
        for j in range(1, m + 1):
            o = M_row[j - 1] + go + ge
            e = Iy_row[j - 1] + ge
            best = max(o, e, 0.0)
            Iy_row[j] = best
            if best == 0.0:
                tb_Iy[i, j] = 0  # sentinel: traceback stops here
            elif o >= e:
                tb_Iy[i, j] = 0
            else:
                tb_Iy[i, j] = 2

    return M, Ix, Iy, tb_M, tb_Ix, tb_Iy


def _traceback(s1: str, s2: str,
               M, Ix, Iy, tb_M, tb_Ix, tb_Iy,
               i: int, j: int, start_state: int,
               local: bool) -> tuple[str, str, int, int]:
    """Walk the traceback matrices and return (aligned_s1, aligned_s2, i_start, j_start).

    For global alignment start_state is the best of {M, Ix, Iy} at (n, m).
    For local alignment, traceback stops when a score of 0 is reached.
    Returns the alignment strings (reversed during construction) and the
    (i, j) coordinates where the traceback ended (useful for local mode).
    """
    a1: list[str] = []
    a2: list[str] = []
    state = start_state

    while i > 0 or j > 0:
        if state == 0:  # M state: diagonal move
            # Boundary guards: if one sequence is exhausted, force to the
            # correct gap state rather than letting i/j go negative.
            if i <= 0:
                state = 2; continue   # drain remaining s2 as gaps in s1
            if j <= 0:
                state = 1; continue   # drain remaining s1 as gaps in s2
            # local: stop if we've reached the 0-origin
            if local and M[i, j] <= 0:
                break
            a1.append(s1[i - 1])
            a2.append(s2[j - 1])
            prev = int(tb_M[i, j])
            i -= 1
            j -= 1
            state = prev

        elif state == 1:  # Ix state: gap in seq2
            if i <= 0:
                state = 2; continue   # s1 exhausted — drain s2 instead
            if local and Ix[i, j] <= 0:
                break
            a1.append(s1[i - 1])
            a2.append("-")
            prev = int(tb_Ix[i, j])
            i -= 1
            state = 0 if prev == 0 else 1

        else:  # Iy state: gap in seq1
            if j <= 0:
                state = 1; continue   # s2 exhausted — drain s1 instead
            if local and Iy[i, j] <= 0:
                break
            a1.append("-")
            a2.append(s2[j - 1])
            prev = int(tb_Iy[i, j])
            j -= 1
            state = 0 if prev == 0 else 2

    a1.reverse()
    a2.reverse()
    return "".join(a1), "".join(a2), i, j


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def align(
    seq1: Union[str, "pd.Series"],
    seq2: Union[str, "pd.Series"],
    *,
    mode: str = "global",
    substitution_matrix: Union[str, dict] = "blosum62",
    gap_open: float = -10.0,
    gap_extend: float = -1.0,
    seq1_id: Optional[str] = None,
    seq2_id: Optional[str] = None,
) -> MSA:
    """Pairwise sequence alignment returning a 2-sequence MSA.

    Parameters
    ----------
    seq1, seq2:
        Sequences to align.  Each can be a plain ``str`` or a ``pd.Series``
        of single characters (as returned by ``msa.iloc[i]`` or
        ``msa.loc["id"]``).  When a Series is passed its ``.name`` is used as
        the sequence ID unless overridden by ``seq1_id`` / ``seq2_id``.
    mode:
        ``'global'`` (default) — Needleman-Wunsch; aligns the full length of
        both sequences, best for roughly equal-length, similar sequences.

        ``'local'`` — Smith-Waterman; finds the highest-scoring local
        subsequence alignment, best for finding a shared domain within longer
        or dissimilar sequences.
    substitution_matrix:
        Name of a built-in matrix (``'blosum62'`` or ``'nuc'``) or a custom
        ``dict`` mapping ``(char_a, char_b)`` pairs to scores.
        Default: ``'blosum62'``.
    gap_open:
        Penalty for opening a new gap (must be ≤ 0).  A gap of length k costs
        ``gap_open + k * gap_extend``.  Default: -10.
    gap_extend:
        Per-residue extension penalty (must be ≤ 0).  Default: -1.
    seq1_id, seq2_id:
        Sequence IDs to use in the returned MSA index.  If omitted, inferred
        from the Series name, or ``'seq1'``/``'seq2'``.

    Returns
    -------
    MSA
        Two-sequence MSA with gaps inserted.  Alignment score and percent
        identity are stored in ``MSA.metadata``.

    Raises
    ------
    ValueError
        If ``mode`` is unrecognised or gap penalties are positive.
    """
    if mode not in ("global", "local"):
        raise ValueError(f"mode must be 'global' or 'local', got {mode!r}")
    if gap_open > 0 or gap_extend > 0:
        raise ValueError("gap_open and gap_extend must be ≤ 0")

    s1, id1 = _to_str(seq1, seq1_id)
    s2, id2 = _to_str(seq2, seq2_id)

    if id1 == id2:
        id2 = id2 + "_2"

    # Resolve substitution matrix
    if isinstance(substitution_matrix, str):
        sub = _MATRICES.get(substitution_matrix.lower())
        if sub is None:
            raise ValueError(
                f"Unknown substitution_matrix {substitution_matrix!r}. "
                f"Built-ins: {sorted(_MATRICES)}."
            )
    else:
        sub = substitution_matrix

    # Strip gaps from input sequences (align the raw residues)
    s1_raw = s1.replace("-", "").replace(".", "")
    s2_raw = s2.replace("-", "").replace(".", "")

    scores = _score_matrix(s1_raw, s2_raw, sub)

    if mode == "global":
        M, Ix, Iy, tb_M, tb_Ix, tb_Iy = _nw_dp(
            s1_raw, s2_raw, scores, gap_open, gap_extend
        )
        n, m = len(s1_raw), len(s2_raw)
        # Start from the best terminal state
        terminal = [M[n, m], Ix[n, m], Iy[n, m]]
        start_state = int(np.argmax(terminal))
        score = terminal[start_state]
        a1, a2, _, _ = _traceback(
            s1_raw, s2_raw, M, Ix, Iy, tb_M, tb_Ix, tb_Iy,
            n, m, start_state, local=False,
        )
    else:  # local
        M, Ix, Iy, tb_M, tb_Ix, tb_Iy = _sw_dp(
            s1_raw, s2_raw, scores, gap_open, gap_extend
        )
        # Find max-score cell across all three matrices
        max_M  = M.max();  pos_M  = np.unravel_index(M.argmax(),  M.shape)
        max_Ix = Ix.max(); pos_Ix = np.unravel_index(Ix.argmax(), Ix.shape)
        max_Iy = Iy.max(); pos_Iy = np.unravel_index(Iy.argmax(), Iy.shape)
        candidates = [(max_M, 0, pos_M), (max_Ix, 1, pos_Ix), (max_Iy, 2, pos_Iy)]
        score, start_state, (si, sj) = max(candidates, key=lambda x: x[0])
        a1, a2, _, _ = _traceback(
            s1_raw, s2_raw, M, Ix, Iy, tb_M, tb_Ix, tb_Iy,
            si, sj, start_state, local=True,
        )

    # Build MSA array
    L = len(a1)
    arr = np.empty((2, L), dtype="U1")
    arr[0] = list(a1)
    arr[1] = list(a2)
    index = pd.Index([id1, id2], name="id")

    # Compute percent identity over aligned (non-double-gap) columns
    matches = sum(c1 == c2 and c1 != "-" for c1, c2 in zip(a1, a2))
    aligned_cols = sum(c1 != "-" or c2 != "-" for c1, c2 in zip(a1, a2))
    pct_id = 100.0 * matches / aligned_cols if aligned_cols else 0.0

    metadata = pd.DataFrame(
        {"score": [float(score), float(score)],
         "pct_identity": [pct_id, pct_id],
         "mode": [mode, mode]},
        index=index,
    )

    return MSA(arr, index=index, metadata=metadata)
