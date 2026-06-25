import numpy as np
import pandas as pd
from typing import Optional, Union


# ---------------------------------------------------------------------------
# Amino-acid three-letter → one-letter map
# ---------------------------------------------------------------------------

# Standard 20 amino acids plus common modified/non-standard residues.
# Unknown residues map to 'X'.
_AA3TO1: dict = {
    # Standard
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    # Common non-standard
    "MSE": "M",  # selenomethionine
    "SEC": "U",  # selenocysteine
    "PYL": "O",  # pyrrolysine
    # His protonation states (CHARMM / AMBER)
    "HSD": "H", "HSE": "H", "HSP": "H", "HIE": "H", "HID": "H", "HIP": "H",
    # Phosphorylated
    "SEP": "S", "TPO": "T", "PTR": "Y",
    # Other common modified residues
    "MLY": "K",  # N-methyl-lysine
    "CSO": "C",  # S-hydroxycysteine
    "CME": "C",  # S,S-(2-hydroxyethyl)thiocysteine
    "OCS": "C",  # cysteinesulfenic acid
    "KCX": "K",  # lysine NZ-carboxylic acid
    "LLP": "K",  # lysine-pyridoxal-5'-phosphate
}
_UNKNOWN_AA = "X"


def _res_name_to_one(res_name: str) -> str:
    """Convert a three-letter residue name to a one-letter code.

    Unknown residues return ``'X'``.
    """
    return _AA3TO1.get(str(res_name).upper().strip(), _UNKNOWN_AA)


# ---------------------------------------------------------------------------
# _SCHEMA — fixed ordered column defaults
# ---------------------------------------------------------------------------

# Mapping from select() kwarg name → atom-table column name
_SELECT_KWARG_TO_COL = {
    "chain":    "chain_id",
    "resi":     "res_seq",
    "atom":     "atom_name",
    "res_name": "res_name",
    "element":  "element",
    "model":    "model",
}

_REQUIRED_COLS = ("atom_name", "res_name", "chain_id", "res_seq")

_OPTIONAL_DEFAULTS: dict = {
    "record_name":    "ATOM",
    "atom_serial":    pd.NA,       # Int64 nullable
    "alt_loc":        "",
    "icode":          "",
    "occupancy":      1.0,
    "b_factor":       0.0,
    "element":        "",
    "charge":         "",
    "model":          1,
    "hetero":         False,
    "label_asym_id":  pd.NA,
    "label_seq_id":   pd.NA,       # Int64 nullable
    "label_entity_id": pd.NA,
}

# Canonical column order in the atoms DataFrame
_COL_ORDER = [
    "record_name", "atom_serial", "atom_name", "alt_loc",
    "res_name", "chain_id", "res_seq", "icode",
    "occupancy", "b_factor", "element", "charge",
    "model", "hetero",
    "label_asym_id", "label_seq_id", "label_entity_id",
]


def _fill_defaults(df: pd.DataFrame) -> pd.DataFrame:
    """Add any missing optional columns to *df* with their default values.
    Returns a new DataFrame with columns in canonical order.
    """
    df = df.copy()
    for col, default in _OPTIONAL_DEFAULTS.items():
        if col not in df.columns:
            df[col] = default
    # Cast nullable integer columns
    for col in ("atom_serial", "res_seq", "label_seq_id"):
        if col in df.columns:
            df[col] = df[col].astype("Int64")
    # Reorder to canonical order (keep any extra columns at the end)
    ordered = [c for c in _COL_ORDER if c in df.columns]
    extra = [c for c in df.columns if c not in _COL_ORDER]
    return df[ordered + extra]


# ---------------------------------------------------------------------------
# Module-level helpers shared by Structure.select and DistanceMatrix.select
# ---------------------------------------------------------------------------

def _selection_mask(
    df: pd.DataFrame,
    *,
    chain=None,
    resi=None,
    atom=None,
    res_name=None,
    element=None,
    model=None,
    hetero=None,
) -> pd.Series:
    """Return a boolean Series (same index as *df*) matching all supplied criteria.

    Criteria are combined with AND.  A ``None`` criterion is a no-op.  If a
    non-``None`` criterion references a column absent from *df*, a
    :exc:`ValueError` is raised with a clear message (used to reject e.g.
    ``atom=`` on a residue-level :class:`DistanceMatrix`).
    """
    mask = pd.Series(True, index=df.index)

    kwargs = {
        "chain": ("chain_id", chain),
        "resi":  ("res_seq",  resi),
        "atom":  ("atom_name", atom),
        "res_name": ("res_name", res_name),
        "element":  ("element",  element),
        "model":    ("model",    model),
    }

    for kwarg_name, (col, val) in kwargs.items():
        if val is None:
            continue
        if col not in df.columns:
            raise ValueError(
                f"Cannot filter by {kwarg_name!r}: column '{col}' is not present "
                f"in this label table.  Available columns: {list(df.columns)}"
            )
        col_data = df[col]
        if isinstance(val, (list, np.ndarray, range)):
            mask &= col_data.isin(list(val))
        else:
            mask &= col_data == val

    if hetero is not None:
        if "hetero" not in df.columns:
            raise ValueError(
                "Cannot filter by 'hetero': column 'hetero' is not present "
                f"in this label table.  Available columns: {list(df.columns)}"
            )
        mask &= df["hetero"] == bool(hetero)

    return mask


def _pairwise_distances(coords: np.ndarray) -> np.ndarray:
    """Compute the pairwise Euclidean distance matrix for *coords*.

    Parameters
    ----------
    coords : ndarray, shape (n, 3)

    Returns
    -------
    ndarray, shape (n, n), dtype float64
        Symmetric matrix with zero diagonal.
    """
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]  # (n, n, 3)
    return np.sqrt((diff ** 2).sum(axis=-1))


# ---------------------------------------------------------------------------
# _AtomIndexer — lightweight single-axis .iloc for Structure
# ---------------------------------------------------------------------------

class _AtomIndexer:
    """Integer-position indexer returning a sub-:class:`Structure`.

    Accessed via ``structure.iloc[rows]``. Row selection may be an integer,
    slice, list, or boolean array — anything accepted by numpy/pandas integer
    positional indexing. Column access is intentionally *not* supported here;
    use ``structure.atoms[col]`` directly.

    Notes
    -----
    This is intentionally lighter than :class:`MSA`'s two-axis
    ``_IlocIndexer`` because there is only one meaningful selection axis
    (atoms); the annotation columns have fixed names and are accessed through
    the ``atoms`` DataFrame directly.
    """

    def __init__(self, structure: "Structure"):
        self._s = structure

    def __getitem__(self, rows) -> "Structure":
        new_coords = self._s._coords[rows]
        new_atoms = self._s.atoms.iloc[rows].reset_index(drop=True)
        if new_coords.ndim == 1:
            # scalar row → single-atom Structure (coords must stay 2-D)
            new_coords = new_coords[np.newaxis, :]
            new_atoms = new_atoms.to_frame().T.reset_index(drop=True)
        return Structure(new_coords, new_atoms, validate=False)


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------

class Structure:
    """Protein structure backed by a numpy coordinate array and a pandas atom table.

    Memory layout: an ``(n_atoms, 3)`` float64 array of xyz coordinates, plus a
    parallel :class:`~pandas.DataFrame` (``atoms``) sharing a positional
    :class:`~pandas.RangeIndex`.  Every row ``i`` of ``coords`` corresponds
    exactly to row ``i`` of ``atoms``, so coordinate and annotation access
    are always aligned.

    Annotation columns follow a fixed schema:

    ``record_name``, ``atom_serial``, ``atom_name``, ``alt_loc``,
    ``res_name``, ``chain_id``, ``res_seq``, ``icode``,
    ``occupancy``, ``b_factor``, ``element``, ``charge``,
    ``model``, ``hetero``, ``label_asym_id``, ``label_seq_id``,
    ``label_entity_id``.

    The ``b_factor`` column doubles as the per-residue confidence score
    (pLDDT) for AlphaFold models; access it via the :attr:`plddt` alias.

    Selection uses author-facing identifiers (``chain_id``, ``res_seq``)
    to match PDB/biological conventions. The mmCIF ``label_*`` columns are
    present but are ``<NA>`` when a file is read from PDB format.

    Parameters
    ----------
    coords : array-like, shape (n_atoms, 3)
        xyz coordinates; coerced to float64.
    atoms : pandas.DataFrame
        Per-atom annotations. Must contain the required columns
        ``atom_name``, ``res_name``, ``chain_id``, ``res_seq``.
        Missing optional columns are filled with schema defaults.
    validate : bool, default True
        When False the shape/required-column checks are skipped (used
        internally by :meth:`select` and :meth:`iloc` for speed).

    Examples
    --------
    >>> import bioviper2 as bv
    >>> s = bv.read_structure("1abc.pdb")
    >>> s
    Structure(1234 atoms, 2 chains [A, B], 156 residues, 1 model)
    >>> s.select(chain="A", atom="CA").coords.shape
    (78, 3)
    >>> s.atoms["b_factor"]          # raw B-factors / pLDDT scores
    >>> s.select(hetero=False)       # protein + nucleic only, no waters/ligands
    """

    def __init__(
        self,
        coords,
        atoms: pd.DataFrame,
        *,
        validate: bool = True,
    ):
        self._coords = np.asarray(coords, dtype=np.float64)

        if validate:
            if self._coords.ndim != 2 or self._coords.shape[1] != 3:
                raise ValueError(
                    f"coords must be shape (n_atoms, 3), got {self._coords.shape}"
                )
            if len(atoms) != self._coords.shape[0]:
                raise ValueError(
                    f"atoms row count ({len(atoms)}) must match number of atoms "
                    f"({self._coords.shape[0]})"
                )
            missing = [c for c in _REQUIRED_COLS if c not in atoms.columns]
            if missing:
                raise ValueError(
                    f"atoms DataFrame is missing required columns: {missing}"
                )

        self._index = pd.RangeIndex(self._coords.shape[0])
        self.atoms = _fill_defaults(atoms)
        self.atoms.index = self._index

    # ------------------------------------------------------------------
    # Properties — coordinates
    # ------------------------------------------------------------------

    @property
    def coords(self) -> np.ndarray:
        """xyz coordinate array, shape ``(n_atoms, 3)``, dtype float64."""
        return self._coords

    # ------------------------------------------------------------------
    # Properties — sizes
    # ------------------------------------------------------------------

    @property
    def shape(self) -> tuple:
        """Shape of the coordinate array: ``(n_atoms, 3)``."""
        return self._coords.shape

    @property
    def n_atoms(self) -> int:
        """Total number of atoms."""
        return self._coords.shape[0]

    @property
    def n_models(self) -> int:
        """Number of distinct models (NMR ensembles, multi-model files)."""
        return int(self.atoms["model"].nunique())

    @property
    def models(self) -> np.ndarray:
        """Sorted array of unique model numbers."""
        return np.sort(self.atoms["model"].unique())

    @property
    def chains(self) -> np.ndarray:
        """Unique chain identifiers in order of first appearance."""
        seen = {}
        for c in self.atoms["chain_id"]:
            if c not in seen:
                seen[c] = None
        return np.array(list(seen))

    @property
    def n_chains(self) -> int:
        """Number of distinct chains."""
        return len(self.chains)

    @property
    def residues(self) -> pd.DataFrame:
        """Residue table: one row per unique ``(model, chain_id, res_seq, icode)``.

        Columns: ``model``, ``chain_id``, ``res_seq``, ``icode``, ``res_name``.
        Rows are in file order (order of first appearance). Index is a fresh
        :class:`~pandas.RangeIndex`.
        """
        key_cols = ["model", "chain_id", "res_seq", "icode"]
        subset = self.atoms[key_cols + ["res_name"]]
        # drop_duplicates preserves first-occurrence order
        return subset.drop_duplicates(subset=key_cols).reset_index(drop=True)

    @property
    def n_residues(self) -> int:
        """Number of distinct residues (unique model/chain/resSeq/icode tuples)."""
        return len(self.residues)

    @property
    def b_factors(self) -> pd.Series:
        """B-factor / temperature-factor column as a Series."""
        return self.atoms["b_factor"]

    @property
    def plddt(self) -> pd.Series:
        """AlphaFold pLDDT confidence scores (stored in the b_factor column)."""
        return self.atoms["b_factor"]

    # ------------------------------------------------------------------
    # Indexer
    # ------------------------------------------------------------------

    @property
    def iloc(self) -> _AtomIndexer:
        """Integer-position indexer returning sub-:class:`Structure` objects.

        Usage::

            s.iloc[0]          # first atom as a 1-atom Structure
            s.iloc[:100]       # first 100 atoms
            s.iloc[[0, 5, 9]]  # atoms at positions 0, 5, 9
        """
        return _AtomIndexer(self)

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def select(
        self,
        *,
        chain=None,
        resi=None,
        atom=None,
        res_name=None,
        element=None,
        model=None,
        hetero=None,
    ) -> "Structure":
        """Return a sub-:class:`Structure` matching all supplied criteria (AND).

        Each keyword argument accepts a scalar value or a list/array of values.
        Omitting an argument (or passing ``None``) imposes no filter on that
        field.  The returned :class:`Structure` has a fresh
        :class:`~pandas.RangeIndex` and its ``coords``/``atoms`` arrays are
        positionally aligned.  Returns a zero-atom :class:`Structure` when no
        atoms match — it does not raise.

        Parameters
        ----------
        chain : str or list of str, optional
            Filter on ``chain_id`` (author chain id).
        resi : int, list of int, or range, optional
            Filter on ``res_seq`` (author residue sequence number).
        atom : str or list of str, optional
            Filter on ``atom_name``.
        res_name : str or list of str, optional
            Filter on ``res_name`` (three-letter residue code).
        element : str or list of str, optional
            Filter on ``element`` symbol.
        model : int or list of int, optional
            Filter on ``model`` number.
        hetero : bool, optional
            ``True`` → keep only HETATM atoms; ``False`` → keep only ATOM atoms.

        Examples
        --------
        >>> ca_a = s.select(chain="A", atom="CA")
        >>> backbone = s.select(atom=["N", "CA", "C", "O"], hetero=False)
        >>> waters = s.select(res_name="HOH")
        >>> model1 = s.select(model=1)
        """
        mask = _selection_mask(
            self.atoms,
            chain=chain, resi=resi, atom=atom,
            res_name=res_name, element=element,
            model=model, hetero=hetero,
        )
        idx = mask.values
        new_coords = self._coords[idx]
        new_atoms = self.atoms[idx].reset_index(drop=True)
        return Structure(new_coords, new_atoms, validate=False)

    # ------------------------------------------------------------------
    # Sequence
    # ------------------------------------------------------------------

    def sequence(self, *, ca_only: bool = False) -> str:
        """Return the one-letter residue sequence in :attr:`residues` order.

        All chains are concatenated (no separator).  Unknown three-letter
        residue names are mapped to ``'X'``.

        Parameters
        ----------
        ca_only : bool, default False
            When ``True``, only residues that have a Cα atom are included.
            The resulting string is **byte-for-byte aligned with the row order
            of** ``distance_matrix("ca")`` — the k-th character corresponds to
            row k of the CA distance matrix.

        Returns
        -------
        str
            One character per residue.

        Examples
        --------
        >>> s.sequence()                 # all residues, all chains
        'MAKVFGR...'
        >>> s.select(chain="A").sequence()
        'MAKVFGR'
        >>> s.sequence(ca_only=True)     # aligned with distance_matrix("ca")
        'MAKVFGR'
        """
        if ca_only:
            # Replicate exactly what distance_matrix("ca") does so the order
            # is guaranteed identical.
            ca_atoms = self.select(atom="CA")
            res_names = ca_atoms.atoms["res_name"]
        else:
            res_names = self.residues["res_name"]
        return "".join(_res_name_to_one(rn) for rn in res_names)

    def sequence_residues(self, *, ca_only: bool = False) -> pd.DataFrame:
        """Return a residue table with an additional ``one_letter`` column.

        The result is :attr:`residues` (or its CA-filtered version) augmented
        with a ``'one_letter'`` column, so each character in :meth:`sequence`
        has a corresponding row carrying its ``(chain_id, res_seq, icode)``
        back-mapping.

        Parameters
        ----------
        ca_only : bool, default False
            When ``True``, restrict to CA-bearing residues (parallel to
            ``sequence(ca_only=True)``).

        Returns
        -------
        pandas.DataFrame
            Columns: ``model``, ``chain_id``, ``res_seq``, ``icode``,
            ``res_name``, ``one_letter``.  Fresh :class:`~pandas.RangeIndex`.

        Examples
        --------
        >>> sr = s.select(chain="A").sequence_residues()
        >>> sr[["res_seq", "res_name", "one_letter"]]
        """
        if ca_only:
            ca_atoms = self.select(atom="CA")
            res_table = ca_atoms.atoms[
                ["model", "chain_id", "res_seq", "icode", "res_name"]
            ].reset_index(drop=True).copy()
        else:
            res_table = self.residues.copy()
        res_table["one_letter"] = [
            _res_name_to_one(rn) for rn in res_table["res_name"]
        ]
        return res_table

    # ------------------------------------------------------------------
    # Distance matrix
    # ------------------------------------------------------------------

    def distance_matrix(
        self,
        mode: str = "ca",
        *,
        model: Optional[int] = None,
    ) -> "DistanceMatrix":
        """Compute a pairwise distance matrix and return a labeled :class:`DistanceMatrix`.

        Parameters
        ----------
        mode : str
            How to define the representative point(s) per entity:

            ``"ca"``
                Cα–Cα distances. One row/column per residue that has a Cα atom.
                Returns a residue-level :class:`DistanceMatrix`.
            ``"cb"``
                Cβ–Cβ distances, with Cα substituted for glycine (and for any
                residue that lacks a Cβ).  Returns a residue-level matrix.
            ``"all_atom"`` (alias ``"all"``)
                All-atom pairwise distances. One row/column per atom; includes
                HETATM atoms unless the caller pre-filters with
                ``.select(hetero=False)``.  Returns an atom-level matrix.
                *O(n²)* memory — suitable for typical PDB structures.
            ``"min"``
                Residue×residue minimum inter-atomic distance (the distance
                between the closest atom pair of each residue pair).  Returns
                a residue-level matrix.  Zero on the diagonal (self-pairs).

        model : int, optional
            Select a single model before computing.  *Required* when the
            structure contains more than one model; omitting it raises a
            :exc:`ValueError` so that cross-model distances are never computed
            silently.

        Returns
        -------
        DistanceMatrix

        Examples
        --------
        >>> dm = s.distance_matrix("ca")
        >>> dm.shape
        (4, 4)
        >>> dm.select(chain="A").values
        array([[0.  , 6.93],
               [6.93, 0.  ]])
        >>> dm.to_dataframe().loc[("A", 1, ""), ("A", 2, "")]
        6.928...
        >>> s.distance_matrix("all_atom").select(chain="A", atom="CA")
        DistanceMatrix(all_atom, 2×2 atoms)
        """
        mode = mode.lower()
        if mode == "all":
            mode = "all_atom"
        _valid = {"ca", "cb", "all_atom", "min"}
        if mode not in _valid:
            raise ValueError(
                f"Unknown distance_matrix mode {mode!r}. "
                f"Supported modes: {sorted(_valid)}"
            )

        # ── multi-model guard ──────────────────────────────────────────────
        if self.n_models > 1:
            if model is None:
                raise ValueError(
                    f"This structure has {self.n_models} models. "
                    "Pass model=<int> or call .select(model=<int>) first to "
                    "avoid computing cross-model distances."
                )
            s = self.select(model=model)
        else:
            s = self

        # ── residue key helper ─────────────────────────────────────────────
        _rkey_cols = ["model", "chain_id", "res_seq", "icode"]

        # ── residue-label columns (for residue-level matrices) ─────────────
        _res_label_cols = ["model", "chain_id", "res_seq", "icode", "res_name"]

        # ── CA / CB modes ──────────────────────────────────────────────────
        if mode in ("ca", "cb"):
            if mode == "ca":
                rep = s.select(atom="CA")
            else:  # cb
                # For each residue, prefer CB; fall back to CA (GLY, etc.)
                ca_cb = s.select(atom=["CA", "CB"])
                if len(ca_cb) == 0:
                    rep = ca_cb
                else:
                    df = ca_cb.atoms.copy()
                    df["_xyz_i"] = np.arange(len(df))
                    # Build a residue-key column for grouping
                    df["_rkey"] = (
                        df["chain_id"].astype(str) + "|"
                        + df["res_seq"].astype(str) + "|"
                        + df["icode"].astype(str) + "|"
                        + df["model"].astype(str)
                    )
                    # Within each residue, rank CB first (0), CA second (1)
                    df["_pref"] = df["atom_name"].map({"CB": 0, "CA": 1}).fillna(2).astype(int)
                    df = df.sort_values(["_rkey", "_pref"])
                    keep_idx = df.groupby("_rkey", sort=False)["_xyz_i"].first().values
                    rep = Structure(
                        ca_cb.coords[keep_idx],
                        ca_cb.atoms.iloc[keep_idx].reset_index(drop=True),
                        validate=False,
                    )

            coords_rep = rep.coords
            matrix = _pairwise_distances(coords_rep)
            labels = rep.atoms[_res_label_cols].reset_index(drop=True)
            return DistanceMatrix(matrix, labels, level="residue", mode=mode)

        # ── all_atom mode ──────────────────────────────────────────────────
        if mode == "all_atom":
            matrix = _pairwise_distances(s.coords)
            labels = s.atoms.reset_index(drop=True)
            return DistanceMatrix(matrix, labels, level="atom", mode=mode)

        # ── min mode ──────────────────────────────────────────────────────
        # mode == "min"
        # Build a per-atom residue code (compact integer) preserving file order
        df = s.atoms.copy()
        rkey_str = (
            df["chain_id"].astype(str) + "|"
            + df["res_seq"].astype(str) + "|"
            + df["icode"].astype(str) + "|"
            + df["model"].astype(str)
        )
        # Factorize in file order (first occurrence defines the code)
        seen: dict = {}
        codes = []
        for k in rkey_str:
            if k not in seen:
                seen[k] = len(seen)
            codes.append(seen[k])
        codes_arr = np.array(codes, dtype=np.int32)

        D_atom = _pairwise_distances(s.coords)   # (n_atoms, n_atoms)
        n_res = len(seen)

        # Reduce to (n_res, n_res) by taking min over atom groups
        D_res = np.full((n_res, n_res), np.inf, dtype=np.float64)
        np.minimum.at(D_res, (codes_arr[:, np.newaxis], codes_arr[np.newaxis, :]), D_atom)
        # Self-pairs along diagonal should be exactly 0
        np.fill_diagonal(D_res, 0.0)

        # Build residue labels (first-appearance order)
        first_idx = []
        seen2: dict = {}
        for i, k in enumerate(rkey_str):
            if k not in seen2:
                seen2[k] = i
        first_idx = [seen2[k] for k in seen]   # same order as `seen`
        labels = s.atoms.iloc[first_idx][_res_label_cols].reset_index(drop=True)

        return DistanceMatrix(D_res, labels, level="residue", mode=mode)

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    def to_dataframe(self) -> pd.DataFrame:
        """Return a single flat DataFrame with coordinate columns appended.

        The result contains all columns of :attr:`atoms` plus ``x``, ``y``,
        ``z`` columns derived from :attr:`coords`.  This is the "expensive
        full view" analogous to :meth:`MSA.to_dataframe`.

        Returns
        -------
        pandas.DataFrame
            Shape ``(n_atoms, n_annotation_cols + 3)``.
        """
        xyz = pd.DataFrame(
            self._coords, columns=["x", "y", "z"], index=self._index
        )
        return pd.concat([self.atoms, xyz], axis=1)

    def copy(self) -> "Structure":
        """Return a deep copy of this :class:`Structure`."""
        return Structure(self._coords.copy(), self.atoms.copy(), validate=False)

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.n_atoms

    def __repr__(self) -> str:
        chain_list = self.chains.tolist()
        if len(chain_list) <= 6:
            chains_str = "[" + ", ".join(str(c) for c in chain_list) + "]"
        else:
            shown = ", ".join(str(c) for c in chain_list[:5])
            chains_str = f"[{shown}, …+{len(chain_list)-5}]"

        s = (
            f"Structure({self.n_atoms} atoms, "
            f"{self.n_chains} chain{'s' if self.n_chains != 1 else ''} {chains_str}, "
            f"{self.n_residues} residue{'s' if self.n_residues != 1 else ''}"
        )
        if self.n_models > 1:
            s += f", {self.n_models} models"
        else:
            s += ", 1 model"
        return s + ")"


# ---------------------------------------------------------------------------
# DistanceMatrix
# ---------------------------------------------------------------------------

# MultiIndex key columns for each level
_RESIDUE_INDEX_COLS = ["chain_id", "res_seq", "icode"]
_ATOM_INDEX_COLS    = ["chain_id", "res_seq", "icode", "atom_name"]

# Columns that are valid at each level for .select() filtering
_RESIDUE_VALID_KWARGS = {"chain", "resi", "res_name", "model"}
_ATOM_VALID_KWARGS    = {"chain", "resi", "atom", "res_name", "element", "model", "hetero"}


class DistanceMatrix:
    """Labeled pairwise distance matrix derived from a :class:`Structure`.

    Both axes share a single label table (a subset of the atom table, or its
    residue reduction) so the matrix is always navigable by meaningful
    biological identifiers rather than arbitrary integer positions.

    Instances are created by :meth:`Structure.distance_matrix` — do not
    construct directly.

    Attributes
    ----------
    labels : pandas.DataFrame
        One row per row/column of the matrix.  For residue-level matrices
        the columns are ``model``, ``chain_id``, ``res_seq``, ``icode``,
        ``res_name``.  For atom-level matrices all atom-table annotation
        columns are present.

    Examples
    --------
    >>> dm = s.distance_matrix("ca")
    >>> dm.shape
    (4, 4)
    >>> dm.values[0, 1]         # raw distance between first two residues
    6.928...
    >>> dm.to_dataframe().loc[("A", 1, ""), ("A", 2, "")]
    6.928...
    >>> dm.select(chain="A")    # sub-matrix for chain A only
    DistanceMatrix(ca, 2×2 residues)
    """

    def __init__(
        self,
        matrix: np.ndarray,
        labels: pd.DataFrame,
        *,
        level: str,
        mode: str,
    ):
        self._matrix = np.asarray(matrix, dtype=np.float64)
        self.labels = labels.reset_index(drop=True)
        self._level = level   # "residue" or "atom"
        self._mode = mode

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def values(self) -> np.ndarray:
        """The raw ``(n, n)`` distance matrix as a numpy array."""
        return self._matrix

    @property
    def array(self) -> np.ndarray:
        """Alias for :attr:`values`."""
        return self._matrix

    @property
    def shape(self) -> tuple:
        """Shape of the distance matrix: ``(n, n)``."""
        return self._matrix.shape

    @property
    def level(self) -> str:
        """``"residue"`` or ``"atom"``."""
        return self._level

    @property
    def mode(self) -> str:
        """The mode string used to compute this matrix (``"ca"``, ``"cb"``, etc.)."""
        return self._mode

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    def to_dataframe(self) -> pd.DataFrame:
        """Return the distance matrix as a pandas DataFrame with a MultiIndex.

        The row and column index is a :class:`~pandas.MultiIndex` built from
        the label table:

        * Residue-level matrices (``ca``, ``cb``, ``min``): index levels
          ``(chain_id, res_seq, icode)``.
        * Atom-level matrices (``all_atom``): index levels
          ``(chain_id, res_seq, icode, atom_name)``.

        This lets you access specific cells by biological identifier::

            dm.to_dataframe().loc[("A", 50, ""), ("A", 60, "")]

        Returns
        -------
        pandas.DataFrame
            Square DataFrame with MultiIndex on both axes; ``float64`` values.
        """
        if self._level == "residue":
            idx_cols = _RESIDUE_INDEX_COLS
        else:
            idx_cols = _ATOM_INDEX_COLS

        mi = pd.MultiIndex.from_frame(self.labels[idx_cols])
        return pd.DataFrame(self._matrix, index=mi, columns=mi)

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def select(
        self,
        *,
        chain=None,
        resi=None,
        atom=None,
        res_name=None,
        element=None,
        model=None,
        hetero=None,
    ) -> "DistanceMatrix":
        """Return a sub-:class:`DistanceMatrix` matching all supplied criteria.

        Works identically to :meth:`Structure.select`: each keyword argument
        accepts a scalar or list; criteria combine as AND; ``None`` = no filter.
        Both axes are filtered symmetrically — the result is always square.

        For residue-level matrices (``ca``, ``cb``, ``min``) the ``atom`` and
        ``element`` keyword arguments are rejected with a :exc:`ValueError`
        because those columns are not present in the residue label table.

        Returns a 0×0 :class:`DistanceMatrix` when no rows match.

        Parameters
        ----------
        chain : str or list of str, optional
        resi : int, list of int, or range, optional
        atom : str or list of str, optional
            Atom-level matrices only.
        res_name : str or list of str, optional
        element : str or list of str, optional
            Atom-level matrices only.
        model : int or list of int, optional
        hetero : bool, optional
            Atom-level matrices only.

        Examples
        --------
        >>> dm.select(chain="A")
        >>> dm.select(chain="A", resi=range(1, 11))   # residues 1-10 of chain A
        >>> dm_atoms.select(chain="A", atom="CA")     # all-atom matrix sub-block
        """
        mask = _selection_mask(
            self.labels,
            chain=chain, resi=resi, atom=atom,
            res_name=res_name, element=element,
            model=model, hetero=hetero,
        )
        idx = mask.values
        new_matrix = self._matrix[np.ix_(idx, idx)]
        new_labels = self.labels[idx].reset_index(drop=True)
        return DistanceMatrix(new_matrix, new_labels, level=self._level, mode=self._mode)

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self._matrix.shape[0]

    def __repr__(self) -> str:
        n = self._matrix.shape[0]
        unit = "residues" if self._level == "residue" else "atoms"
        return f"DistanceMatrix({self._mode}, {n}×{n} {unit})"
