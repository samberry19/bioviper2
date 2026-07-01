import warnings
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Union

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Canonical 20 standard amino acids in alphabetical order.
STANDARD_AMINO_ACIDS: str = "ACDEFGHIKLMNPQRSTVWY"


# ---------------------------------------------------------------------------
# PairwiseFrequencies — returned by MSA.pairwise_frequencies()
# ---------------------------------------------------------------------------

class PairwiseFrequencies:
    """Joint and single-site character frequencies from an MSA.

    Attributes
    ----------
    alphabet : list[str]
        Non-gap characters present in the alignment (sorted).

    Access patterns
    ---------------
    pf[i, j]   → DataFrame (A × A) of joint frequencies for positions i, j
    pf.site    → DataFrame (L × A) of single-site frequencies
    pf.array   → raw (L, L, A, A) float32 numpy array
    """

    def __init__(
        self,
        f_ij: np.ndarray,
        f_i: np.ndarray,
        alphabet: list,
        positions: pd.RangeIndex,
    ):
        self._f_ij = f_ij          # (L, L, A, A) float32
        self._f_i = f_i            # (L, A) float32
        self.alphabet = alphabet
        self._positions = positions

    def __getitem__(self, key) -> pd.DataFrame:
        """Return the (A × A) joint frequency DataFrame for position pair (i, j)."""
        i, j = key
        return pd.DataFrame(self._f_ij[i, j], index=self.alphabet, columns=self.alphabet)

    @property
    def site(self) -> pd.DataFrame:
        """Single-site frequencies: DataFrame of shape (L × A)."""
        return pd.DataFrame(self._f_i, index=self._positions, columns=self.alphabet)

    @property
    def array(self) -> np.ndarray:
        """Raw (L, L, A, A) pairwise frequency array (float32)."""
        return self._f_ij

    def __repr__(self) -> str:
        L = self._f_ij.shape[0]
        return (
            f"PairwiseFrequencies({L} positions × "
            f"{len(self.alphabet)}-char alphabet)"
        )


def _label_slice_positions(index: pd.Index, key: slice) -> Tuple[int, int]:
    """Convert a label slice (inclusive on both ends, pandas semantics) to
    a half-open integer range [start, stop) suitable for numpy slicing."""
    if key.step is not None:
        raise NotImplementedError("Step slices are not supported in .loc")

    def _get_pos(label, default: int) -> int:
        pos = index.get_loc(label)
        if not isinstance(pos, (int, np.integer)):
            raise ValueError(
                f"Label '{label}' is not unique — label slices require a unique index"
            )
        return int(pos)

    start = _get_pos(key.start, 0) if key.start is not None else 0
    stop = (_get_pos(key.stop, len(index) - 1) + 1) if key.stop is not None else len(index)
    return start, stop


class _IlocIndexer:
    """Integer-location based indexer: msa.iloc[rows, cols].

    Mirrors pandas .iloc semantics:
      - scalar row + scalar col  → single character (str)
      - scalar row               → Series indexed by position
      - scalar col               → Series indexed by sequence ID
      - slice/array for both     → DataFrame
    """

    def __init__(self, msa: "MSA"):
        self._msa = msa

    def __getitem__(self, key) -> Union[str, pd.Series, "MSA"]:
        if isinstance(key, tuple):
            row_key, col_key = key
        else:
            row_key, col_key = key, slice(None)

        scalar_row = isinstance(row_key, (int, np.integer))
        scalar_col = isinstance(col_key, (int, np.integer))

        arr = self._msa._array

        if scalar_row:
            row_data = arr[int(row_key)]  # 1D
            if scalar_col:
                return row_data[int(col_key)].item()
            return pd.Series(
                row_data[col_key],
                index=self._msa._columns[col_key],
                name=self._msa._index[int(row_key)],
            )

        row_data = arr[row_key]  # 2D

        if scalar_col:
            return pd.Series(
                row_data[:, int(col_key)],
                index=self._msa._index[row_key],
                name=self._msa._columns[int(col_key)],
            )

        return self._msa._make_subset(
            row_data[:, col_key],
            row_idx=self._msa._index[row_key],
            col_idx=self._msa._columns[col_key],
            row_int=row_key,
        )


def _resolve_col_loc(col_index: pd.Index, col_key):
    """Translate a .loc column key to (numpy_indexer, col_labels, is_scalar).

    Performs label-based lookup so that both integer keys on a RangeIndex
    (backward-compatible) and named column labels on a custom Index work.
    """
    if isinstance(col_key, slice):
        if col_key == slice(None):
            return slice(None), col_index, False
        start, stop = _label_slice_positions(col_index, col_key)
        col_np = slice(start, stop)
        return col_np, col_index[col_np], False

    if isinstance(col_key, (list, np.ndarray)):
        col_pos = col_index.get_indexer(col_key)
        missing = [k for k, p in zip(col_key, col_pos) if p == -1]
        if missing:
            raise KeyError(f"Column labels not found: {missing}")
        return col_pos, col_index[col_pos], False

    # scalar label
    c_pos = col_index.get_loc(col_key)
    if not isinstance(c_pos, (int, np.integer)):
        raise ValueError(f"Column label {col_key!r} is not unique")
    return int(c_pos), None, True


class _LocIndexer:
    """Label-based indexer: msa.loc[seq_ids, col_labels].

    Both rows (sequence IDs) and columns (position labels) use label-based
    lookup.  When column_index is the default RangeIndex, integer column keys
    behave identically to before (RangeIndex.get_loc(n) == n).

    Supported row key forms:
      - str                → single sequence (returns Series)
      - 'id1':'id3' slice  → inclusive label slice (returns DataFrame)
      - list / array       → explicit list of IDs (returns DataFrame)

    Supported column key forms (same shapes):
      - scalar label       → column Series (or single char with scalar row)
      - label slice        → subset (inclusive, label-based)
      - list / array       → explicit list of column labels
    """

    def __init__(self, msa: "MSA"):
        self._msa = msa

    def __getitem__(self, key) -> Union[str, pd.Series, "MSA"]:
        if isinstance(key, tuple):
            row_key, col_key = key
        else:
            row_key, col_key = key, slice(None)

        col_np, col_labels, scalar_col = _resolve_col_loc(
            self._msa._columns, col_key
        )

        # ---- single row label --------------------------------------------
        if isinstance(row_key, str):
            row_pos = int(self._msa._index.get_loc(row_key))
            row_data = self._msa._array[row_pos]  # 1D
            if scalar_col:
                return row_data[col_np].item()
            return pd.Series(
                row_data[col_np],
                index=col_labels,
                name=row_key,
            )

        # ---- label slice -------------------------------------------------
        if isinstance(row_key, slice):
            start, stop = _label_slice_positions(self._msa._index, row_key)
            row_np = slice(start, stop)
            row_idx = self._msa._index[row_np]

        # ---- array-like of labels ----------------------------------------
        else:
            int_pos = self._msa._index.get_indexer(row_key)
            missing = np.asarray(row_key)[int_pos == -1]
            if len(missing):
                raise KeyError(f"Sequence IDs not found: {missing.tolist()}")
            row_np = int_pos
            row_idx = self._msa._index[row_np]

        row_data = self._msa._array[row_np]  # 2D

        if scalar_col:
            return pd.Series(
                row_data[:, col_np],
                index=row_idx,
                name=self._msa._columns[col_np],
            )

        return self._msa._make_subset(
            row_data[:, col_np],
            row_idx=row_idx,
            col_idx=col_labels,
            row_int=row_np,
        )


class MSA:
    """Multiple Sequence Alignment backed by a numpy 'U1' array.

    Memory layout: (n_seqs × n_positions) array of single Unicode characters.
    Indexing via .loc / .iloc slices the numpy array and returns a labeled
    pandas DataFrame — so you only pay the object-dtype cost for the subset,
    not the full alignment.

    Metadata (organism, accession, etc.) lives in a separate DataFrame that
    shares the same index as the sequences.
    """

    def __init__(
        self,
        array: np.ndarray,
        index: pd.Index,
        metadata: Optional[pd.DataFrame] = None,
    ):
        if array.ndim != 2:
            raise ValueError("array must be 2D (n_seqs × n_positions)")
        self._array = np.asarray(array, dtype="U1")
        self._index = pd.Index(index)
        self._columns = pd.RangeIndex(self._array.shape[1])

        if metadata is not None:
            if len(metadata) != len(self._index):
                raise ValueError("metadata row count must match number of sequences")
            self.metadata = metadata.copy()
            self.metadata.index = self._index
        else:
            self.metadata = None

        self._weights: Optional[np.ndarray] = None  # None → uniform 1.0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def shape(self) -> Tuple[int, int]:
        return self._array.shape

    @property
    def n_seqs(self) -> int:
        return self._array.shape[0]

    @property
    def n_positions(self) -> int:
        return self._array.shape[1]

    @property
    def index(self) -> pd.Index:
        return self._index

    @index.setter
    def index(self, new_index) -> None:
        idx = pd.Index(new_index)
        if len(idx) != self.n_seqs:
            raise ValueError(
                f"index length {len(idx)} does not match n_seqs {self.n_seqs}"
            )
        self._index = idx
        if self.metadata is not None:
            self.metadata.index = idx

    @property
    def column_index(self) -> pd.Index:
        """Column labels for each alignment position.

        By default this is a :class:`~pandas.RangeIndex` (0, 1, 2, …).  Assign
        any sequence of labels of length ``n_positions`` to name the columns
        according to a standardised numbering scheme (e.g. Kabat, IMGT,
        Ballesteros–Weinstein)::

            msa.column_index = kabat_numbers   # list, array, or pd.Index

        Once named, columns can be selected by label with ``.loc``::

            msa.loc[:, "H50"]          # all sequences at position H50
            msa.loc["seq1", "H1":"H5"] # residues H1–H5 for seq1 (inclusive)
            msa.loc[:, ["H1", "H50"]]  # two named columns

        ``.iloc`` continues to use integer positions regardless.
        """
        return self._columns

    @column_index.setter
    def column_index(self, names) -> None:
        idx = pd.Index(names)
        if len(idx) != self.n_positions:
            raise ValueError(
                f"column_index length {len(idx)} does not match "
                f"n_positions {self.n_positions}"
            )
        self._columns = idx

    @property
    def weights(self) -> pd.Series:
        """Sequence weights as a Series indexed by sequence ID.

        Returns uniform 1.0 for all sequences until :meth:`compute_weights`
        (or a manual assignment) has been called.
        """
        vals = np.ones(self.n_seqs, dtype=np.float64) if self._weights is None else self._weights.astype(np.float64)
        return pd.Series(vals, index=self._index, name="weight")

    @weights.setter
    def weights(self, values) -> None:
        arr = np.asarray(values, dtype=np.float64)
        if arr.shape != (self.n_seqs,):
            raise ValueError(f"weights must have length {self.n_seqs}, got {arr.shape}")
        self._weights = arr

    @property
    def n_eff(self) -> float:
        """Effective number of sequences (sum of weights)."""
        return float(self.weights.sum())

    def _effective_weights(self) -> np.ndarray:
        """Return float32 weight vector of length n_seqs."""
        if self._weights is None:
            return np.ones(self.n_seqs, dtype=np.float32)
        return self._weights.astype(np.float32)

    # ------------------------------------------------------------------
    # Indexers
    # ------------------------------------------------------------------

    @property
    def iloc(self) -> _IlocIndexer:
        return _IlocIndexer(self)

    @property
    def loc(self) -> _LocIndexer:
        return _LocIndexer(self)

    def _make_subset(
        self,
        array: np.ndarray,
        row_idx: pd.Index,
        col_idx: pd.Index,
        row_int,
    ) -> "MSA":
        """Build a new MSA from an array slice, preserving metadata/weights."""
        new_msa = MSA(array, index=row_idx)
        new_msa._columns = col_idx
        if self.metadata is not None:
            meta = self.metadata.iloc[row_int].copy()
            meta.index = row_idx
            new_msa.metadata = meta
        if self._weights is not None:
            new_msa._weights = self._weights[row_int].copy()
        return new_msa

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    def to_dataframe(self) -> pd.DataFrame:
        """Convert full alignment to an object-dtype DataFrame.
        For large MSAs this is expensive — prefer .iloc / .loc for subsets."""
        return pd.DataFrame(self._array, index=self._index, columns=self._columns)

    def select_uppercase_columns(self) -> "MSA":
        """Return a new MSA containing only columns that have at least one uppercase character.

        In HMM-based alignments (e.g. from hmmalign / hmmbuild), match-state
        columns carry uppercase residues while insert-state columns carry
        lowercase residues.  This method retains only the match-state columns,
        discarding pure-lowercase (insert) columns.

        Sequence weights and metadata are preserved unchanged (they are
        per-sequence, not per-column).
        """
        is_upper = np.char.isupper(self._array)   # (n, L) bool; False for gaps/lowercase
        keep = is_upper.any(axis=0)               # (L,) — True for match-state columns

        new_msa = MSA(self._array[:, keep], index=self._index, metadata=self.metadata)
        if self._weights is not None:
            new_msa._weights = self._weights.copy()
        if not isinstance(self._columns, pd.RangeIndex):
            new_msa._columns = self._columns[keep]
        return new_msa

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def _gap_mask(self, gap_chars) -> np.ndarray:
        """Boolean array (n_seqs × n_positions), True where character is a gap."""
        return np.isin(self._array, list(gap_chars))

    def coverage(self, gap_chars=("-", "."), per: str = "position") -> pd.Series:
        """Alignment coverage, either per-position (depth) or per-sequence (occupancy).

        Parameters
        ----------
        gap_chars : characters treated as gaps.
        per       : ``'position'`` (default) or ``'sequence'``.

            ``'position'``
                **Depth** — for each alignment column, the fraction of sequences
                that carry a non-gap character.  Returns a Series indexed by
                position with values in [0, 1].

            ``'sequence'``
                **Occupancy** — for each sequence, the fraction of alignment
                columns that are non-gap in that sequence.  Returns a Series
                indexed by sequence ID with values in [0, 1].

                Note: this is *not* the same as the fraction of the original
                (unaligned) sequence that is covered by the alignment; computing
                that requires knowing the unaligned sequence lengths, which are
                not stored in the MSA.

        Raises
        ------
        ValueError if *per* is not ``'position'`` or ``'sequence'``.
        """
        if per not in ("position", "sequence"):
            raise ValueError(f"per must be 'position' or 'sequence', got {per!r}")

        is_gap = self._gap_mask(gap_chars)

        if per == "position":
            depth = (~is_gap).sum(axis=0).astype(float) / self.n_seqs
            return pd.Series(depth, index=self._columns, name="coverage")
        else:
            occupancy = (~is_gap).sum(axis=1).astype(float) / self.n_positions
            return pd.Series(occupancy, index=self._index, name="occupancy")

    def conservation(self, gap_chars=("-", ".")) -> pd.Series:
        """Fraction of non-gap sequences carrying the modal character at each position.

        Returns a Series indexed by position with values in [0, 1].
        NaN is returned for gap-only columns. Higher values mean more conserved.
        """
        is_gap = self._gap_mask(gap_chars)
        n_non_gap = (~is_gap).sum(axis=0).astype(float)
        gap_set = set(gap_chars)
        non_gap_chars = [c for c in np.unique(self._array) if c not in gap_set]

        if not non_gap_chars:
            return pd.Series(np.nan, index=self._columns, name="conservation")

        counts = np.stack([(self._array == c).sum(axis=0) for c in non_gap_chars])
        max_count = counts.max(axis=0).astype(float)

        with np.errstate(invalid="ignore"):
            result = np.where(n_non_gap > 0, max_count / n_non_gap, np.nan)

        return pd.Series(result, index=self._columns, name="conservation")

    def entropy(self, gap_chars=("-", ".")) -> pd.Series:
        """Shannon entropy (bits) at each position.

        Returns a Series indexed by position with values in [0, log2(alphabet_size)].
        NaN for gap-only columns. Lower values mean more conserved.
        """
        is_gap = self._gap_mask(gap_chars)
        n_non_gap = (~is_gap).sum(axis=0).astype(float)
        gap_set = set(gap_chars)
        non_gap_chars = [c for c in np.unique(self._array) if c not in gap_set]

        if not non_gap_chars:
            return pd.Series(np.nan, index=self._columns, name="entropy")

        counts = np.stack([(self._array == c).sum(axis=0) for c in non_gap_chars], dtype=float)

        with np.errstate(invalid="ignore", divide="ignore"):
            freqs = np.where(n_non_gap > 0, counts / n_non_gap, 0.0)
            log_freqs = np.where(freqs > 0, np.log2(freqs), 0.0)

        H = np.maximum(0.0, -(freqs * log_freqs).sum(axis=0))
        H = np.where(n_non_gap > 0, H, np.nan)

        return pd.Series(H, index=self._columns, name="entropy")

    def site_frequencies(
        self,
        gap_chars=("-", "."),
        pseudocount: float = 0.0,
    ) -> pd.DataFrame:
        """Character frequency at each alignment position.

        Returns a DataFrame of shape (n_positions × alphabet_size).
        Index: alignment positions. Columns: non-gap characters (sorted).
        Gap-only columns yield NaN; with a pseudocount they get a valid
        distribution even when N_eff = 0.

        Parameters
        ----------
        gap_chars   : characters treated as gaps; excluded from the alphabet
                      and from the per-position sequence count.
        pseudocount : uniform pseudocount λ.  Each character count gets λ/A
                      added before normalisation; the effective N increases by λ.
                      A value of 1 is a common choice (equivalent to a flat
                      Dirichlet prior).
        """
        gap_list = list(gap_chars)
        gap_set = set(gap_chars)
        is_gap = np.isin(self._array, gap_list)

        w = self._effective_weights()                          # (n,) float32
        not_gap = (~is_gap).astype(np.float32)                 # (n, L)
        n_eff = (w @ not_gap).astype(float)                   # (L,) weighted

        alphabet = sorted(c for c in np.unique(self._array) if c not in gap_set)
        A = len(alphabet)

        counts = np.stack(
            [w @ (self._array == c).astype(np.float32) for c in alphabet], axis=1
        ).astype(float)                                        # (L, A) weighted

        if pseudocount > 0:
            counts += pseudocount / A
            n_eff = n_eff + pseudocount

        with np.errstate(invalid="ignore"):
            freqs = np.where(n_eff[:, None] > 0, counts / n_eff[:, None], np.nan)

        return pd.DataFrame(freqs, index=self._columns, columns=alphabet)

    def pairwise_frequencies(
        self,
        gap_chars=("-", "."),
        pseudocount: float = 0.0,
    ) -> PairwiseFrequencies:
        """Joint character frequencies for every pair of alignment positions.

        Returns a :class:`PairwiseFrequencies` object.  Access patterns:

        .. code-block:: python

            pf = msa.pairwise_frequencies(pseudocount=1)
            pf[i, j]   # DataFrame (A × A) for positions i and j
            pf.site    # DataFrame (L × A) single-site marginals
            pf.array   # raw (L, L, A, A) float32 numpy array

        Parameters
        ----------
        gap_chars   : characters excluded from the alphabet and from N_eff.
        pseudocount : uniform pseudocount λ.
                      f_i(a)    = (count_i(a)   + λ/A)   / (N_i  + λ)
                      f_ij(a,b) = (count_ij(a,b) + λ/A²) / (N_ij + λ)
                      where N_i / N_ij are non-gap sequence counts.

        Notes
        -----
        The result tensor is O(L² × A²) in memory (float32).  For long
        alignments consider subsetting positions first.
        """
        n, L = self.shape
        gap_list = list(gap_chars)
        gap_set = set(gap_chars)
        is_gap = np.isin(self._array, gap_list)

        alphabet = sorted(c for c in np.unique(self._array) if c not in gap_set)
        A = len(alphabet)

        mem_gb = L * L * A * A * 4 / 1024 ** 3
        if mem_gb > 2:
            warnings.warn(
                f"Pairwise frequency tensor requires ~{mem_gb:.1f} GB. "
                "Consider subsetting positions first.",
                stacklevel=2,
            )

        # Denominators
        w = self._effective_weights()             # (n,) float32
        not_gap = (~is_gap).astype(np.float32)   # (n, L)
        wn = not_gap * w[:, None]                # (n, L) weighted not-gap
        n_i = wn.sum(axis=0)                     # (L,)
        n_ij = wn.T @ not_gap                    # (L, L)  — BLAS, symmetric

        # Accumulate counts
        # count_ij(a,b)[i,j] = sum_s w_s * A_a[s,i] * A_b[s,j]
        #   = ((A_a * w[:,None]).T @ A_b)[i, j]
        # Exploit symmetry: count_ij(b,a) = count_ij(a,b).T
        counts_ij = np.zeros((L, L, A, A), dtype=np.float32)
        counts_i = np.zeros((L, A), dtype=np.float32)

        for ka, a in enumerate(alphabet):
            A_a = (self._array == a).astype(np.float32)   # (n, L)
            counts_i[:, ka] = w @ A_a                      # weighted marginal

            for kb in range(ka + 1):
                A_b = (self._array == alphabet[kb]).astype(np.float32)
                C = (A_a * w[:, None]).T @ A_b             # (L, L) weighted
                counts_ij[:, :, ka, kb] = C
                if ka != kb:
                    counts_ij[:, :, kb, ka] = C.T

        # Apply pseudocount
        if pseudocount > 0:
            counts_ij += pseudocount / (A * A)
            counts_i += pseudocount / A
            n_ij = n_ij + pseudocount
            n_i = n_i + pseudocount

        # Normalise
        with np.errstate(invalid="ignore"):
            f_ij = np.where(
                n_ij[:, :, None, None] > 0,
                counts_ij / n_ij[:, :, None, None],
                np.nan,
            ).astype(np.float32)
            f_i = np.where(
                n_i[:, None] > 0, counts_i / n_i[:, None], np.nan
            ).astype(np.float32)

        return PairwiseFrequencies(f_ij, f_i, alphabet, self._columns)

    def pairwise_identity(
        self,
        gap_chars=("-", "."),
        mem_limit_mb: int = 512,
    ) -> pd.DataFrame:
        """Pairwise sequence identity matrix.

        For each pair (i, j):
            identity = (positions where both seqs have the same non-gap character)
                     / (positions where both seqs have any non-gap character)

        Algorithm: for each character c in the alphabet, form a binary (n × L)
        indicator matrix A_c and accumulate A_c @ A_c.T via BLAS. This is
        O(|alphabet| × n² × L) in flops but the n² term is handled by fast
        matrix multiplication rather than an explicit pair loop.

        Positions are processed in chunks so peak memory stays near
        *mem_limit_mb* regardless of alignment length.

        Returns
        -------
        pd.DataFrame of shape (n_seqs, n_seqs), float32, values in [0, 1].
        Diagonal is 1.0. NaN where a pair shares no non-gap positions.
        """
        n, L = self.shape
        gap_list = list(gap_chars)
        gap_set = set(gap_chars)

        # Warn early if the result matrix itself will be large
        result_gb = n * n * 4 / 1024 ** 3
        if result_gb > 2:
            import warnings
            warnings.warn(
                f"Result matrix will occupy ~{result_gb:.1f} GB. "
                "Consider subsetting first.",
                stacklevel=2,
            )

        # Per-chunk memory budget:
        # chunk  (U1, 4 B/char) + is_gap (bool, 1 B) + not_gap (f32, 4 B) + A_c (f32, 4 B)
        # ≈ 13 bytes × n × l_chunk
        bytes_per_col = 13 * n
        l_chunk = max(1, (mem_limit_mb * 1024 * 1024) // bytes_per_col)
        l_chunk = min(l_chunk, L)

        # Unique non-gap characters (alphabet); computed once over the full array
        unique_chars = [c for c in np.unique(self._array) if c not in gap_set]

        numer = np.zeros((n, n), dtype=np.float32)
        denom = np.zeros((n, n), dtype=np.float32)

        for l_start in range(0, L, l_chunk):
            chunk = self._array[:, l_start : l_start + l_chunk]  # (n, chunk_len) U1

            not_gap = (~np.isin(chunk, gap_list)).astype(np.float32)  # (n, chunk_len)
            denom += not_gap @ not_gap.T

            for c in unique_chars:
                A = (chunk == c).astype(np.float32)  # (n, chunk_len)
                numer += A @ A.T

        with np.errstate(invalid="ignore"):
            identity = np.where(denom > 0, numer / denom, np.nan)
        np.fill_diagonal(identity, 1.0)

        return pd.DataFrame(identity, index=self._index, columns=self._index)

    def pairwise_distance(
        self,
        matrix="blosum62",
        gap_chars=("-", "."),
        mem_limit_mb: int = 512,
    ) -> pd.DataFrame:
        """Pairwise sequence distance using a substitution matrix.

        For each pair (i, j):

        .. code-block:: text

            distance = 1 - S_ij / sqrt(S_ii|j * S_jj|i)

        where S_ij is the sum of substitution scores over columns where *both*
        sequences are non-gap, and S_ii|j (resp. S_jj|i) is the sum of
        *self-scores* of sequence i (resp. j) restricted to those same shared
        non-gap columns.  Dividing numerator and denominator by the shared
        non-gap count cancels, so the metric is length-independent.

        The cross-score is computed via the eigendecomposition of *S* —
        one BLAS matmul per eigenvalue, the same count as
        :meth:`pairwise_identity` has per unique character.

        Parameters
        ----------
        matrix : str or dict or (alphabet, ndarray) tuple
            Substitution matrix.  ``"blosum62"`` (default).  Accepts any
            format understood by :func:`bioviper2.matrices.as_array`:

            - ``str``  — built-in name: ``"blosum62"`` or ``"nuc"``.
            - ``dict`` — flat ``(char_a, char_b) -> score`` mapping;
              symmetric, or with only one ordering (the reverse is inferred).
            - ``tuple[list[str], np.ndarray]`` — ``(alphabet, S)`` where
              *S* is a square symmetric numeric array.

        gap_chars : characters treated as gaps (excluded from scoring).
            Characters present in the alignment but absent from the matrix
            alphabet are also treated as gaps (a warning is emitted).
        mem_limit_mb : approximate peak memory budget for column chunks, MB.

        Returns
        -------
        pd.DataFrame of shape (n_seqs, n_seqs), float32, indexed by sequence
        ID.  Diagonal is 0.0.  NaN where a pair shares no non-gap positions.
        Values can exceed 1.0 for pairs more dissimilar than random background
        (BLOSUM62 cross-scores go negative).  Symmetric by construction.
        """
        from .matrices import as_array

        n, L = self.shape
        gap_set = set(gap_chars)

        result_gb = n * n * 4 / 1024 ** 3
        if result_gb > 2:
            warnings.warn(
                f"Result matrix will occupy ~{result_gb:.1f} GB. "
                "Consider subsetting first.",
                stacklevel=2,
            )

        # Resolve substitution matrix -> sorted alphabet + (A, A) float32 array
        alphabet, S = as_array(matrix, gap_chars=tuple(gap_chars))
        A = len(alphabet)
        char_to_idx: dict = {c: i for i, c in enumerate(alphabet)}

        # Warn about MSA characters that are absent from the matrix alphabet
        all_chars = {c for c in np.unique(self._array) if c not in gap_set}
        unknown = all_chars - set(alphabet)
        if unknown:
            warnings.warn(
                f"Characters {sorted(unknown)!r} are not in the substitution "
                f"matrix and will be treated as gaps.",
                stacklevel=2,
            )

        # Eigendecompose S = V diag(lam) V^T (eigh: real symmetric, stable)
        lam, V = np.linalg.eigh(S.astype(np.float64))
        lam = lam.astype(np.float32)  # (A,) ascending
        V   = V.astype(np.float32)    # (A, A), V[:, k] = k-th eigenvector

        # Diagonal self-scores S[a, a] for each residue in the alphabet
        self_scores = np.diag(S)  # (A,) float32

        # Column chunk size (same budget model as pairwise_identity)
        bytes_per_col = 13 * n
        l_chunk = max(1, (mem_limit_mb * 1024 * 1024) // bytes_per_col)
        l_chunk = min(l_chunk, L)

        num    = np.zeros((n, n), dtype=np.float32)  # Σ_l S[res_i(l), res_j(l)]
        si_acc = np.zeros((n, n), dtype=np.float32)  # Σ_l S[res_i(l), res_i(l)] on shared cols
        denom  = np.zeros((n, n), dtype=np.float32)  # count of shared non-gap cols

        for l_start in range(0, L, l_chunk):
            chunk = self._array[:, l_start : l_start + l_chunk]  # (n, cl) U1
            cl = chunk.shape[1]

            # Residue codes: -1 for gap or out-of-alphabet characters
            codes = np.full((n, cl), -1, dtype=np.int16)
            for c, ci in char_to_idx.items():
                codes[chunk == c] = ci

            valid = codes >= 0                  # (n, cl) bool: in-matrix non-gap
            safe  = np.where(valid, codes, 0)   # (n, cl) int16, dummy 0 at gaps

            in_mat = valid.astype(np.float32)   # 1.0 where valid, 0 elsewhere
            denom += in_mat @ in_mat.T

            # Self-scores: S[a, a] for each residue; 0 at gaps/unknowns
            ss = np.where(valid, self_scores[safe], 0.0).astype(np.float32)
            # si_acc[i,j] = Σ_l ss[i,l] * in_mat[j,l]
            #   = self-scores of row-seq i on cols where col-seq j is valid
            si_acc += ss @ in_mat.T

            # Cross-score via eigendecomposition:
            # num[i,j] += Σ_k lam_k (Y_k @ Y_k^T)[i,j]
            # Y_k[i,l] = V[res_i(l), k] * in_mat[i,l]  (0 at gaps)
            for k in range(A):
                Yk = np.where(valid, V[safe, k], 0.0).astype(np.float32)
                num += lam[k] * (Yk @ Yk.T)

        # norm_sim[i,j] = num[i,j] / sqrt(si_acc[i,j] * si_acc[j,i])
        # si_acc[j,i] = sj_acc[i,j] (transposed), so si_acc.T gives the j self-scores
        with np.errstate(invalid="ignore", divide="ignore"):
            norm_denom = np.sqrt(si_acc * si_acc.T)
            sim = np.where(
                (denom > 0) & (norm_denom > 0),
                num / norm_denom,
                np.nan,
            )

        distance = (1.0 - sim).astype(np.float32)
        np.fill_diagonal(distance, 0.0)

        return pd.DataFrame(distance, index=self._index, columns=self._index)

    def compute_weights(
        self,
        threshold: float = 0.8,
        gap_chars=("-", "."),
        mem_limit_mb: int = 512,
    ) -> pd.Series:
        """Compute and store similarity-threshold sequence weights.

        For each sequence i, k_i = number of sequences (including i itself)
        whose pairwise identity with i is >= *threshold*.  The weight is then
        w_i = 1/k_i.  Sequences with no comparable partners (NaN identity)
        are treated as singletons (k_i = 1, w_i = 1).

        After calling this method, :attr:`weights` and :attr:`n_eff` reflect
        the new values, and both :meth:`site_frequencies` and
        :meth:`pairwise_frequencies` will use them automatically.

        Parameters
        ----------
        threshold   : identity threshold (default 0.8).
        gap_chars   : forwarded to :meth:`pairwise_identity`.
        mem_limit_mb: forwarded to :meth:`pairwise_identity`.

        Returns
        -------
        pd.Series of weights (also stored as ``self.weights``).
        """
        pid = self.pairwise_identity(gap_chars=gap_chars, mem_limit_mb=mem_limit_mb)
        pid_arr = pid.to_numpy(dtype=np.float32, na_value=0.0)
        k = (pid_arr >= threshold).sum(axis=1).astype(np.float64)
        k = np.maximum(k, 1.0)
        self._weights = 1.0 / k
        return self.weights

    def to_sequences(self) -> pd.Series:
        """Return each sequence as a string, indexed by sequence ID."""
        return pd.Series(
            ["".join(row) for row in self._array],
            index=self._index,
            name="sequence",
        )

    def to_unaligned_sequences(self, gap_chars=("-", ".")) -> pd.Series:
        """Return each sequence with gap characters removed, indexed by sequence ID.

        Strips all characters in *gap_chars* from every row, yielding the
        original unaligned sequences.  The result is a Series of strings with
        varying lengths — i.e. the same format returned by
        :func:`~bioviper2.read_fasta_sequences`.

        Parameters
        ----------
        gap_chars : characters treated as gaps (default ``('-', '.')``)

        Examples
        --------
        Round-trip to a FASTA file::

            seqs = msa.to_unaligned_sequences()
            bioviper2.write_sequences(seqs.to_frame(), "ungapped.fasta")

        Preserve metadata alongside sequences::

            df = msa.to_unaligned_sequences().to_frame()
            if msa.metadata is not None:
                df = df.join(msa.metadata)
        """
        gap_set = set(gap_chars)
        return pd.Series(
            ["".join(c for c in row if c not in gap_set) for row in self._array],
            index=self._index,
            name="sequence",
        )

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __getitem__(self, key) -> Union["MSA", pd.Series, str]:
        """Row-selection shorthand delegating to .loc.

        msa["seqA"]           → pd.Series (single sequence)
        msa[["seqA", "seqB"]] → MSA subset
        msa["seqA":"seqC"]    → MSA subset (inclusive slice of sequence IDs)
        msa[0:5]              → use msa.iloc[0:5] for integer row slices
        """
        return self.loc[key]

    def __len__(self) -> int:
        return self.n_seqs

    def __repr__(self) -> str:
        meta_info = (
            f", metadata={list(self.metadata.columns)}"
            if self.metadata is not None
            else ""
        )
        weight_info = f", N_eff={self.n_eff:.1f}" if self._weights is not None else ""
        col_info = (
            f", column_index={self._columns.dtype.name}"
            if not isinstance(self._columns, pd.RangeIndex)
            else ""
        )
        return (
            f"MSA({self.n_seqs} sequences × {self.n_positions} positions"
            f"{col_info}{meta_info}{weight_info})"
        )


# ---------------------------------------------------------------------------
# Module-level utilities
# ---------------------------------------------------------------------------


def one_hot_encode_msa(
    msa,
    alphabet="standard",
    include_gap: bool = False,
    gap_chars=("-", "."),
    as_array: bool = False,
    dtype=np.float32,
):
    """One-hot encode every position of an MSA.

    Unlike ``pd.get_dummies()``, this function uses a **fixed alphabet** by
    default (the 20 standard amino acids), so encodings from different MSAs
    that share the same alignment columns are directly comparable — no missing
    columns for residues that happen to be absent from a particular MSA.

    Parameters
    ----------
    msa : MSA
        The alignment to encode.
    alphabet : {"standard", "present"} or sequence of str, default ``"standard"``
        Which symbols to encode:

        - ``"standard"`` — the 20 canonical amino acids in alphabetical order
          (``ACDEFGHIKLMNPQRSTVWY``, see :data:`STANDARD_AMINO_ACIDS`).
          Residues not in this set (e.g. ``X``, ``B``, gaps) produce an
          all-zero vector at that position.
        - ``"present"`` — only the non-gap characters that appear in *this*
          MSA (sorted), equivalent to the ``pd.get_dummies`` behaviour.  Not
          comparable to encodings of other MSAs unless their alphabets match.
        - explicit sequence of single characters — e.g. ``["A", "C", "G", "T"]``
          for a nucleotide MSA.  Must contain no duplicates.

    include_gap : bool, default ``False``
        If ``True``, append a gap category (labelled ``"-"``) after the
        residue columns.  All characters in *gap_chars* map to this column;
        non-gap residues outside the alphabet do **not** (they stay all-zero).
    gap_chars : tuple of str, default ``("-", ".")``
        Characters treated as gaps.  Relevant both for ``alphabet="present"``
        (they are excluded from the inferred alphabet) and for ``include_gap``.
    as_array : bool, default ``False``
        If ``True``, return a ``(array, alphabet_list)`` tuple instead of a
        DataFrame.  *array* has shape ``(n_seqs, n_positions, A)`` where *A*
        is the alphabet size (including the gap column if *include_gap*).
    dtype : numpy dtype, default ``np.float32``
        Element type of the output array / DataFrame values.

    Returns
    -------
    pd.DataFrame
        Shape ``(n_seqs, n_positions × A)``.  Row index = sequence IDs.
        Columns = ``pd.MultiIndex`` with levels ``("position", "residue")``,
        where *position* comes from ``msa.column_index`` (integers by default).
        Column order is **position-major, residue-minor**, so
        ``df.to_numpy().reshape(n_seqs, n_positions, A)`` recovers the 3-D
        array.
    or (np.ndarray, list[str])
        When ``as_array=True``: a tuple of the ``(n, L, A)`` encoding array
        and the flat alphabet list (including ``"-"`` at the end if
        ``include_gap=True``).

    Notes
    -----
    Memory scales as ``O(n_seqs × n_positions × A)``.  For very long
    alignments against large alphabets, consider subsetting positions first.

    Examples
    --------
    >>> df = bv.one_hot_encode_msa(msa)              # 20-AA, no gaps
    >>> df = bv.one_hot_encode_msa(msa, include_gap=True)   # 21 channels
    >>> df = bv.one_hot_encode_msa(msa, alphabet="present") # get_dummies style
    >>> arr, alph = bv.one_hot_encode_msa(msa, as_array=True)  # (n, L, 20) array
    """
    if not isinstance(msa, MSA):
        raise TypeError(
            f"msa must be an MSA instance; got {type(msa).__name__}."
        )

    gap_set = set(gap_chars)

    # ---- Resolve alphabet --------------------------------------------------
    if isinstance(alphabet, str):
        if alphabet == "standard":
            symbols = list(STANDARD_AMINO_ACIDS)
        elif alphabet == "present":
            symbols = sorted(
                c for c in np.unique(msa._array) if c not in gap_set
            )
        else:
            raise ValueError(
                f"alphabet string must be 'standard' or 'present'; "
                f"got {alphabet!r}."
            )
    else:
        # Explicit sequence of characters
        symbols = list(alphabet)
        if any(len(c) != 1 for c in symbols):
            raise ValueError(
                "Each entry in a custom alphabet must be a single character."
            )
        if len(symbols) != len(set(symbols)):
            raise ValueError("Custom alphabet contains duplicate characters.")

    if include_gap:
        full_alphabet = symbols + ["-"]
    else:
        full_alphabet = symbols

    A = len(full_alphabet)
    n, L = msa.shape

    # ---- Build (n, L, A) encoding -----------------------------------------
    arr = msa._array          # (n, L) U1
    encoding = np.zeros((n, L, A), dtype=dtype)

    for k, sym in enumerate(symbols):
        encoding[:, :, k] = (arr == sym).astype(dtype)

    if include_gap:
        encoding[:, :, -1] = np.isin(arr, list(gap_chars)).astype(dtype)

    # ---- Return ------------------------------------------------------------
    if as_array:
        return encoding, full_alphabet

    # MultiIndex columns: position-major, residue-minor
    col_mi = pd.MultiIndex.from_product(
        [msa._columns, full_alphabet],
        names=["position", "residue"],
    )
    flat = encoding.reshape(n, L * A)
    return pd.DataFrame(flat, index=msa._index, columns=col_mi)
