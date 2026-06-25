"""Tests for map_alignment_to_structure()."""
import math
import numpy as np
import pandas as pd
import pytest

import bioviper2 as bv
from bioviper2 import Structure, MSA
from bioviper2.tools.mapping import map_alignment_to_structure


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_msa(rows: dict) -> MSA:
    """Build a tiny MSA from a dict {seq_id: aligned_string}."""
    ids = list(rows.keys())
    seqs = list(rows.values())
    n_pos = len(seqs[0])
    arr = np.array([list(s) for s in seqs], dtype="U1")
    return MSA(arr, index=pd.Index(ids, name="id"))


def _three_ca_structure():
    """ALA, GLY, SER — one CA each at x=0,5,9."""
    coords = np.array([
        [0., 0., 0.],   # ALA CA
        [5., 0., 0.],   # GLY CA
        [9., 0., 0.],   # SER CA
    ], dtype=float)
    atoms = pd.DataFrame({
        "atom_name": ["CA", "CA", "CA"],
        "res_name":  ["ALA", "GLY", "SER"],
        "chain_id":  ["A", "A", "A"],
        "res_seq":   [1, 2, 3],
    })
    return Structure(coords, atoms)


def _four_ca_structure():
    """ALA, GLY, LEU, SER — one CA each."""
    coords = np.zeros((4, 3), dtype=float)
    atoms = pd.DataFrame({
        "atom_name": ["CA"]*4,
        "res_name":  ["ALA", "GLY", "LEU", "SER"],
        "chain_id":  ["A"]*4,
        "res_seq":   [1, 2, 3, 4],
    })
    return Structure(coords, atoms)


def _missing_internal_structure():
    """ALA, LEU, SER — GLY is missing internally (structure has residues 1,3,4)."""
    coords = np.zeros((3, 3), dtype=float)
    atoms = pd.DataFrame({
        "atom_name": ["CA", "CA", "CA"],
        "res_name":  ["ALA", "LEU", "SER"],
        "chain_id":  ["A", "A", "A"],
        "res_seq":   [1, 3, 4],
    })
    return Structure(coords, atoms)


# ---------------------------------------------------------------------------
# Identical row == structure sequence → positional via auto
# ---------------------------------------------------------------------------

class TestAutoPositional:
    def test_maps_all_columns(self):
        s = _three_ca_structure()
        msa = _make_msa({"a": "AGS"})
        m = map_alignment_to_structure(msa, "a", s)
        np.testing.assert_array_equal(m, [0, 1, 2])

    def test_no_gap_columns(self):
        s = _three_ca_structure()
        msa = _make_msa({"a": "AGS"})
        m = map_alignment_to_structure(msa, "a", s)
        assert (m >= 0).all()

    def test_uses_positional_path(self):
        """When sequences are identical, auto should use positional (not align)."""
        s = _three_ca_structure()
        msa = _make_msa({"a": "AGS"})
        # If strategy auto → positional: result identical to explicit positional
        m_auto = map_alignment_to_structure(msa, "a", s, strategy="auto")
        m_pos  = map_alignment_to_structure(msa, "a", s, strategy="positional")
        np.testing.assert_array_equal(m_auto, m_pos)


# ---------------------------------------------------------------------------
# Gapped rows
# ---------------------------------------------------------------------------

class TestGappedRow:
    def test_gap_column_gets_minus_one(self):
        s = _three_ca_structure()
        # Alignment row has a gap at position 1: A-GS → 4 positions
        msa = _make_msa({"a": "A-GS"})
        m = map_alignment_to_structure(msa, "a", s)
        assert m[0] == 0   # A → CA row 0
        assert m[1] == -1  # gap
        assert m[2] == 1   # G → CA row 1
        assert m[3] == 2   # S → CA row 2

    def test_leading_gap(self):
        s = _three_ca_structure()
        msa = _make_msa({"a": "-AGS"})
        m = map_alignment_to_structure(msa, "a", s)
        assert m[0] == -1
        assert m[1] == 0
        assert m[2] == 1
        assert m[3] == 2

    def test_trailing_gap(self):
        s = _three_ca_structure()
        msa = _make_msa({"a": "AGS-"})
        m = map_alignment_to_structure(msa, "a", s)
        assert m[0] == 0
        assert m[1] == 1
        assert m[2] == 2
        assert m[3] == -1

    def test_dot_gap_also_recognised(self):
        s = _three_ca_structure()
        msa = _make_msa({"a": "A.GS"})
        m = map_alignment_to_structure(msa, "a", s)
        assert m[1] == -1

    def test_length(self):
        s = _three_ca_structure()
        msa = _make_msa({"a": "A-GS"})
        m = map_alignment_to_structure(msa, "a", s)
        assert len(m) == 4  # == msa.n_positions


# ---------------------------------------------------------------------------
# Missing structural residue → align falls back
# ---------------------------------------------------------------------------

class TestMissingResidue:
    def test_missing_internal_residue_gets_minus_one(self):
        # MSA row = "AGLS" (4 residues), structure has ALA,LEU,SER (GLY missing).
        # align will match: A→A, G→gap in struct, L→L, S→S
        s = _missing_internal_structure()  # ALA, LEU, SER
        msa = _make_msa({"a": "AGLS"})
        m = map_alignment_to_structure(msa, "a", s, strategy="align")
        assert m[0] == 0    # A → ALA (CA row 0)
        assert m[1] == -1   # G → not in structure
        assert m[2] == 1    # L → LEU (CA row 1)
        assert m[3] == 2    # S → SER (CA row 2)

    def test_auto_falls_back_to_align_when_seqs_differ(self):
        s = _missing_internal_structure()  # seq "ALS"
        msa = _make_msa({"a": "AGLS"})    # ungapped: "AGLS" ≠ "ALS"
        m_auto  = map_alignment_to_structure(msa, "a", s, strategy="auto")
        m_align = map_alignment_to_structure(msa, "a", s, strategy="align")
        np.testing.assert_array_equal(m_auto, m_align)


# ---------------------------------------------------------------------------
# Explicit strategies
# ---------------------------------------------------------------------------

class TestStrategies:
    def test_positional_requires_equal_length(self):
        s = _three_ca_structure()    # 3 CA residues
        msa = _make_msa({"a": "AGLS"})  # 4-position row, 4 ungapped
        with pytest.raises(ValueError, match="equal length"):
            map_alignment_to_structure(msa, "a", s, strategy="positional")

    def test_positional_equal_length_works(self):
        s = _three_ca_structure()
        msa = _make_msa({"a": "AGS"})
        m = map_alignment_to_structure(msa, "a", s, strategy="positional")
        np.testing.assert_array_equal(m, [0, 1, 2])

    def test_align_strategy_always_runs(self):
        s = _three_ca_structure()
        msa = _make_msa({"a": "AGS"})
        m = map_alignment_to_structure(msa, "a", s, strategy="align")
        # Even when identical, align path should give the same correct mapping
        np.testing.assert_array_equal(m, [0, 1, 2])


# ---------------------------------------------------------------------------
# return_type
# ---------------------------------------------------------------------------

class TestReturnType:
    def test_default_is_ndarray(self):
        s = _three_ca_structure()
        msa = _make_msa({"a": "AGS"})
        m = map_alignment_to_structure(msa, "a", s)
        assert isinstance(m, np.ndarray)
        assert m.dtype == np.int64

    def test_series_return_type(self):
        s = _three_ca_structure()
        msa = _make_msa({"a": "AGS"})
        m = map_alignment_to_structure(msa, "a", s, return_type="series")
        assert isinstance(m, pd.Series)
        assert m.name == "a"

    def test_series_uses_column_index(self):
        s = _three_ca_structure()
        msa = _make_msa({"a": "AGS"})
        msa.column_index = pd.Index(["H1", "H2", "H3"])
        m = map_alignment_to_structure(msa, "a", s, return_type="series")
        assert list(m.index) == ["H1", "H2", "H3"]

    def test_series_values_match_array(self):
        s = _three_ca_structure()
        msa = _make_msa({"a": "AGS"})
        m_arr = map_alignment_to_structure(msa, "a", s)
        m_ser = map_alignment_to_structure(msa, "a", s, return_type="series")
        np.testing.assert_array_equal(m_arr, m_ser.values)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrors:
    def test_bad_seq_id_raises_keyerror(self):
        s = _three_ca_structure()
        msa = _make_msa({"a": "AGS"})
        with pytest.raises(KeyError):
            map_alignment_to_structure(msa, "z", s)

    def test_multimodel_without_model_raises(self, multimodel_pdb):
        s = bv.read_pdb(multimodel_pdb)
        msa = _make_msa({"a": "A"})
        with pytest.raises(ValueError, match="model"):
            map_alignment_to_structure(msa, "a", s)

    def test_multimodel_with_model_ok(self, multimodel_pdb):
        s = bv.read_pdb(multimodel_pdb)
        # multimodel.pdb: 2 models of ALA A 1 → sequence "A"
        msa = _make_msa({"a": "A"})
        m = map_alignment_to_structure(msa, "a", s, model=1)
        assert m[0] == 0

    def test_chain_kwarg_filters_structure(self, mini_pdb):
        # mini.pdb: chain A = "AG" (2 CA residues), chain B = "LS"
        s = bv.read_pdb(mini_pdb)
        msa = _make_msa({"a": "AG"})
        m = map_alignment_to_structure(msa, "a", s, chain="A")
        np.testing.assert_array_equal(m, [0, 1])


# ---------------------------------------------------------------------------
# Integration with real fixture
# ---------------------------------------------------------------------------

class TestWithMiniPdb:
    def test_full_alignment_maps_correctly(self, mini_pdb):
        s = bv.read_pdb(mini_pdb)
        # Full structure: sequence "AGLS" across both chains
        msa = _make_msa({"a": "AGLS"})
        m = map_alignment_to_structure(msa, "a", s)
        # All 4 columns should map; none should be -1
        assert (m >= 0).all()
        assert len(m) == 4

    def test_gapped_alignment_mini(self, mini_pdb):
        s = bv.read_pdb(mini_pdb)
        # Only chain A columns (positions 0,1); chain B is gap
        msa = _make_msa({"a": "AG--"})
        m = map_alignment_to_structure(msa, "a", s, chain="A")
        assert m[0] == 0
        assert m[1] == 1
        assert m[2] == -1
        assert m[3] == -1

    def test_exports(self):
        assert hasattr(bv, "map_alignment_to_structure")
