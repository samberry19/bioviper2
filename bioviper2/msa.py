import numpy as np
import pandas as pd
from typing import Optional, Tuple, Union


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

    def __getitem__(self, key) -> Union[str, pd.Series, pd.DataFrame]:
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
                name=int(col_key),
            )

        return pd.DataFrame(
            row_data[:, col_key],
            index=self._msa._index[row_key],
            columns=self._msa._columns[col_key],
        )


class _LocIndexer:
    """Label-based row indexer: msa.loc[seq_ids, cols].

    Row keys are sequence ID labels; column keys are integer positions.
    Supported row key forms:
      - str                → single sequence (returns Series)
      - 'id1':'id3' slice  → inclusive label slice (returns DataFrame)
      - list / array       → explicit list of IDs (returns DataFrame)
    Column key can be an int (returns Series), slice, or list.
    """

    def __init__(self, msa: "MSA"):
        self._msa = msa

    def __getitem__(self, key) -> Union[str, pd.Series, pd.DataFrame]:
        if isinstance(key, tuple):
            row_key, col_key = key
        else:
            row_key, col_key = key, slice(None)

        scalar_col = isinstance(col_key, (int, np.integer))

        # ---- single label ------------------------------------------------
        if isinstance(row_key, str):
            row_pos = int(self._msa._index.get_loc(row_key))
            row_data = self._msa._array[row_pos]  # 1D
            if scalar_col:
                return row_data[int(col_key)].item()
            return pd.Series(
                row_data[col_key],
                index=self._msa._columns[col_key],
                name=row_key,
            )

        # ---- label slice -------------------------------------------------
        if isinstance(row_key, slice):
            start, stop = _label_slice_positions(self._msa._index, row_key)
            row_pos = slice(start, stop)
            row_idx = self._msa._index[row_pos]

        # ---- array-like of labels ----------------------------------------
        else:
            int_pos = self._msa._index.get_indexer(row_key)
            missing = np.asarray(row_key)[int_pos == -1]
            if len(missing):
                raise KeyError(f"Sequence IDs not found: {missing.tolist()}")
            row_pos = int_pos
            row_idx = self._msa._index[row_pos]

        row_data = self._msa._array[row_pos]  # 2D

        if scalar_col:
            return pd.Series(
                row_data[:, int(col_key)],
                index=row_idx,
                name=int(col_key),
            )

        return pd.DataFrame(
            row_data[:, col_key],
            index=row_idx,
            columns=self._msa._columns[col_key],
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

    # ------------------------------------------------------------------
    # Indexers
    # ------------------------------------------------------------------

    @property
    def iloc(self) -> _IlocIndexer:
        return _IlocIndexer(self)

    @property
    def loc(self) -> _LocIndexer:
        return _LocIndexer(self)

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    def to_dataframe(self) -> pd.DataFrame:
        """Convert full alignment to an object-dtype DataFrame.
        For large MSAs this is expensive — prefer .iloc / .loc for subsets."""
        return pd.DataFrame(self._array, index=self._index, columns=self._columns)

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def _gap_mask(self, gap_chars) -> np.ndarray:
        """Boolean array (n_seqs × n_positions), True where character is a gap."""
        return np.isin(self._array, list(gap_chars))

    def coverage(self, gap_chars=("-", ".")) -> pd.Series:
        """Fraction of non-gap positions in each sequence.

        Returns a Series indexed by sequence ID with values in [0, 1].
        """
        n_non_gap = (~self._gap_mask(gap_chars)).sum(axis=1).astype(float)
        return pd.Series(n_non_gap / self.n_positions, index=self._index, name="coverage")

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

    def to_sequences(self) -> pd.Series:
        """Return each sequence as a string, indexed by sequence ID."""
        return pd.Series(
            ["".join(row) for row in self._array],
            index=self._index,
            name="sequence",
        )

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.n_seqs

    def __repr__(self) -> str:
        meta_info = (
            f", metadata={list(self.metadata.columns)}"
            if self.metadata is not None
            else ""
        )
        return f"MSA({self.n_seqs} sequences × {self.n_positions} positions{meta_info})"
