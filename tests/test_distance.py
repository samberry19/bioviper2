"""Tests for Structure.distance_matrix() and the DistanceMatrix class."""
import math
import numpy as np
import pandas as pd
import pytest

import bioviper2 as bv
from bioviper2 import DistanceMatrix, Structure

# Expected CA distance within each chain of mini.pdb:
# CA of res1 = (2,3,4), CA of res2 = (6,7,8) → √((4²)·3) = √48 ≈ 6.9282
_SQRT48 = math.sqrt(48)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gly_structure():
    """Small in-memory structure with ALA (has CB), GLY (no CB), SER (has CB)."""
    # ALA: CA at (1,0,0), CB at (2,0,0)
    # GLY: CA at (3,0,0),            ← no CB, should fall back to CA
    # SER: CA at (4,0,0), CB at (5,0,0)
    coords = np.array([
        [1., 0., 0.],  # ALA CA
        [2., 0., 0.],  # ALA CB
        [3., 0., 0.],  # GLY CA
        [4., 0., 0.],  # SER CA
        [5., 0., 0.],  # SER CB
    ], dtype=float)
    atoms = pd.DataFrame({
        "atom_name": ["CA", "CB", "CA", "CA", "CB"],
        "res_name":  ["ALA", "ALA", "GLY", "SER", "SER"],
        "chain_id":  ["A"] * 5,
        "res_seq":   [1, 1, 2, 3, 3],
    })
    return Structure(coords, atoms)


def _min_structure():
    """Two residues with 2 atoms each; minimum inter-residue distance is 1.0."""
    # Residue 1: atoms at x=0 and x=1
    # Residue 2: atoms at x=2 and x=3
    # min distance(res1, res2) = |1-2| = 1.0
    coords = np.array([
        [0., 0., 0.],   # res1 atom1
        [1., 0., 0.],   # res1 atom2
        [2., 0., 0.],   # res2 atom1
        [3., 0., 0.],   # res2 atom2
    ], dtype=float)
    atoms = pd.DataFrame({
        "atom_name": ["N", "CA", "N", "CA"],
        "res_name":  ["ALA", "ALA", "GLY", "GLY"],
        "chain_id":  ["A"] * 4,
        "res_seq":   [1, 1, 2, 2],
    })
    return Structure(coords, atoms)


# ---------------------------------------------------------------------------
# CA mode
# ---------------------------------------------------------------------------

class TestCA:
    def test_shape(self, mini_pdb):
        dm = bv.read_pdb(mini_pdb).distance_matrix("ca")
        assert dm.shape == (4, 4)

    def test_symmetric(self, mini_pdb):
        dm = bv.read_pdb(mini_pdb).distance_matrix("ca")
        np.testing.assert_allclose(dm.values, dm.values.T)

    def test_zero_diagonal(self, mini_pdb):
        dm = bv.read_pdb(mini_pdb).distance_matrix("ca")
        np.testing.assert_allclose(np.diag(dm.values), 0.0)

    def test_known_value(self, mini_pdb):
        dm = bv.read_pdb(mini_pdb).distance_matrix("ca")
        # Row 0 = chain A res1, row 1 = chain A res2
        assert dm.values[0, 1] == pytest.approx(_SQRT48, rel=1e-4)

    def test_level(self, mini_pdb):
        dm = bv.read_pdb(mini_pdb).distance_matrix("ca")
        assert dm.level == "residue"

    def test_mode(self, mini_pdb):
        dm = bv.read_pdb(mini_pdb).distance_matrix("ca")
        assert dm.mode == "ca"

    def test_repr(self, mini_pdb):
        dm = bv.read_pdb(mini_pdb).distance_matrix("ca")
        assert "ca" in repr(dm)
        assert "4×4" in repr(dm)
        assert "residues" in repr(dm)

    def test_len(self, mini_pdb):
        dm = bv.read_pdb(mini_pdb).distance_matrix("ca")
        assert len(dm) == 4

    def test_labels_columns(self, mini_pdb):
        dm = bv.read_pdb(mini_pdb).distance_matrix("ca")
        for col in ("chain_id", "res_seq", "icode", "res_name"):
            assert col in dm.labels.columns

    def test_labels_real_res_seq(self, mini_pdb):
        # res_seq should be author numbers (1 and 2), not 0-3
        dm = bv.read_pdb(mini_pdb).distance_matrix("ca")
        assert set(dm.labels["res_seq"].astype(int)) == {1, 2}

    def test_default_mode_is_ca(self, mini_pdb):
        s = bv.read_pdb(mini_pdb)
        dm_default = s.distance_matrix()
        dm_ca = s.distance_matrix("ca")
        np.testing.assert_array_equal(dm_default.values, dm_ca.values)


# ---------------------------------------------------------------------------
# CA-mode to_dataframe
# ---------------------------------------------------------------------------

class TestToDataFrame:
    def test_multiindex_on_both_axes(self, mini_pdb):
        dm = bv.read_pdb(mini_pdb).distance_matrix("ca")
        df = dm.to_dataframe()
        assert isinstance(df.index, pd.MultiIndex)
        assert isinstance(df.columns, pd.MultiIndex)

    def test_index_names(self, mini_pdb):
        dm = bv.read_pdb(mini_pdb).distance_matrix("ca")
        df = dm.to_dataframe()
        assert list(df.index.names) == ["chain_id", "res_seq", "icode"]
        assert list(df.columns.names) == ["chain_id", "res_seq", "icode"]

    def test_loc_by_real_residue_number(self, mini_pdb):
        dm = bv.read_pdb(mini_pdb).distance_matrix("ca")
        df = dm.to_dataframe()
        val = df.loc[("A", 1, ""), ("A", 2, "")]
        assert val == pytest.approx(_SQRT48, rel=1e-4)

    def test_loc_self_distance_zero(self, mini_pdb):
        dm = bv.read_pdb(mini_pdb).distance_matrix("ca")
        df = dm.to_dataframe()
        assert df.loc[("A", 1, ""), ("A", 1, "")] == pytest.approx(0.0)

    def test_all_atom_index_names(self, mini_pdb):
        dm = bv.read_pdb(mini_pdb).distance_matrix("all_atom")
        df = dm.to_dataframe()
        assert list(df.index.names) == ["chain_id", "res_seq", "icode", "atom_name"]


# ---------------------------------------------------------------------------
# CA-mode select
# ---------------------------------------------------------------------------

class TestCASelect:
    def test_select_chain(self, mini_pdb):
        dm = bv.read_pdb(mini_pdb).distance_matrix("ca")
        sub = dm.select(chain="A")
        assert sub.shape == (2, 2)

    def test_select_chain_list(self, mini_pdb):
        dm = bv.read_pdb(mini_pdb).distance_matrix("ca")
        sub = dm.select(chain=["A", "B"])
        assert sub.shape == (4, 4)

    def test_select_resi(self, mini_pdb):
        dm = bv.read_pdb(mini_pdb).distance_matrix("ca")
        sub = dm.select(resi=1)
        assert sub.shape == (2, 2)   # chain A res1 AND chain B res1

    def test_select_chain_and_resi(self, mini_pdb):
        dm = bv.read_pdb(mini_pdb).distance_matrix("ca")
        sub = dm.select(chain="A", resi=1)
        assert sub.shape == (1, 1)
        np.testing.assert_allclose(sub.values[0, 0], 0.0)

    def test_select_preserves_symmetry(self, mini_pdb):
        dm = bv.read_pdb(mini_pdb).distance_matrix("ca")
        sub = dm.select(chain="A")
        np.testing.assert_allclose(sub.values, sub.values.T)

    def test_select_fresh_labels_index(self, mini_pdb):
        dm = bv.read_pdb(mini_pdb).distance_matrix("ca")
        sub = dm.select(chain="A")
        assert list(sub.labels.index) == [0, 1]

    def test_select_by_real_res_seq(self, mini_pdb):
        # res_seq=2 is the SECOND residue, but its AUTHOR number is 2, not 1
        dm = bv.read_pdb(mini_pdb).distance_matrix("ca")
        sub = dm.select(resi=2)
        assert (sub.labels["res_seq"].astype(int) == 2).all()

    def test_select_empty_no_raise(self, mini_pdb):
        dm = bv.read_pdb(mini_pdb).distance_matrix("ca")
        sub = dm.select(chain="Z")
        assert sub.shape == (0, 0)

    def test_select_atom_raises_on_residue_level(self, mini_pdb):
        dm = bv.read_pdb(mini_pdb).distance_matrix("ca")
        with pytest.raises(ValueError, match="atom"):
            dm.select(atom="CA")

    def test_select_element_raises_on_residue_level(self, mini_pdb):
        dm = bv.read_pdb(mini_pdb).distance_matrix("ca")
        with pytest.raises(ValueError, match="element"):
            dm.select(element="C")


# ---------------------------------------------------------------------------
# CB mode
# ---------------------------------------------------------------------------

class TestCB:
    def test_shape(self, mini_pdb):
        # mini.pdb has no CB atoms (backbone only) so falls back to CA entirely
        dm = bv.read_pdb(mini_pdb).distance_matrix("cb")
        assert dm.shape == (4, 4)
        dm_ca = bv.read_pdb(mini_pdb).distance_matrix("ca")
        np.testing.assert_allclose(dm.values, dm_ca.values)

    def test_gly_fallback_to_ca(self):
        s = _gly_structure()
        dm = s.distance_matrix("cb")
        assert dm.shape == (3, 3)   # ALA, GLY, SER
        # ALA→CB(2,0,0), GLY→CA(3,0,0), SER→CB(5,0,0)
        # ALA-GLY: |2-3| = 1.0; ALA-SER: |2-5| = 3.0; GLY-SER: |3-5| = 2.0
        np.testing.assert_allclose(dm.values[0, 1], 1.0, atol=1e-6)  # ALA-GLY
        np.testing.assert_allclose(dm.values[0, 2], 3.0, atol=1e-6)  # ALA-SER
        np.testing.assert_allclose(dm.values[1, 2], 2.0, atol=1e-6)  # GLY-SER

    def test_cb_uses_cb_not_ca_for_non_gly(self):
        s = _gly_structure()
        dm = s.distance_matrix("cb")
        dm_ca = s.distance_matrix("ca")
        # ALA has CB at x=2, CA at x=1 — CB matrix row 0 should differ from CA
        assert not np.allclose(dm.values[0], dm_ca.values[0])

    def test_level_residue(self):
        assert _gly_structure().distance_matrix("cb").level == "residue"


# ---------------------------------------------------------------------------
# All-atom mode
# ---------------------------------------------------------------------------

class TestAllAtom:
    def test_shape(self, mini_pdb):
        dm = bv.read_pdb(mini_pdb).distance_matrix("all_atom")
        assert dm.shape == (16, 16)

    def test_alias_all(self, mini_pdb):
        dm1 = bv.read_pdb(mini_pdb).distance_matrix("all_atom")
        dm2 = bv.read_pdb(mini_pdb).distance_matrix("all")
        np.testing.assert_array_equal(dm1.values, dm2.values)

    def test_level_atom(self, mini_pdb):
        assert bv.read_pdb(mini_pdb).distance_matrix("all_atom").level == "atom"

    def test_select_ca_sub_block_matches_ca_matrix(self, mini_pdb):
        s = bv.read_pdb(mini_pdb)
        dm_all = s.distance_matrix("all_atom")
        dm_ca = s.distance_matrix("ca")
        sub = dm_all.select(atom="CA")
        np.testing.assert_allclose(sub.values, dm_ca.values, atol=1e-10)

    def test_select_resi_all_atoms(self, mini_pdb):
        dm = bv.read_pdb(mini_pdb).distance_matrix("all_atom")
        # chain A res 1 has 4 atoms (N, CA, C, O)
        sub = dm.select(chain="A", resi=1)
        assert sub.shape == (4, 4)

    def test_repr_says_atoms(self, mini_pdb):
        dm = bv.read_pdb(mini_pdb).distance_matrix("all_atom")
        assert "atoms" in repr(dm)

    def test_atom_level_select_by_atom_name(self, mini_pdb):
        dm = bv.read_pdb(mini_pdb).distance_matrix("all_atom")
        sub = dm.select(atom="N")
        assert sub.shape == (4, 4)   # one N per residue × 4 residues

    def test_atom_level_hetero_filter(self, hetatm_pdb):
        s = bv.read_pdb(hetatm_pdb)
        dm_all = s.distance_matrix("all_atom")
        dm_prot = s.distance_matrix("all_atom")   # same structure
        sub_prot = dm_all.select(hetero=False)
        # hetatm.pdb has 2 ATOM + 3 HETATM = 5 atoms; selecting hetero=False → 2
        assert sub_prot.shape == (2, 2)


# ---------------------------------------------------------------------------
# Min mode
# ---------------------------------------------------------------------------

class TestMin:
    def test_shape(self, mini_pdb):
        dm = bv.read_pdb(mini_pdb).distance_matrix("min")
        assert dm.shape == (4, 4)

    def test_symmetric(self, mini_pdb):
        dm = bv.read_pdb(mini_pdb).distance_matrix("min")
        np.testing.assert_allclose(dm.values, dm.values.T)

    def test_zero_diagonal(self, mini_pdb):
        dm = bv.read_pdb(mini_pdb).distance_matrix("min")
        np.testing.assert_allclose(np.diag(dm.values), 0.0)

    def test_min_leq_ca(self, mini_pdb):
        # The minimum inter-atom distance ≤ CA–CA distance
        dm_min = bv.read_pdb(mini_pdb).distance_matrix("min")
        dm_ca  = bv.read_pdb(mini_pdb).distance_matrix("ca")
        assert np.all(dm_min.values <= dm_ca.values + 1e-9)

    def test_known_min_distance(self):
        s = _min_structure()
        dm = s.distance_matrix("min")
        # min(res1, res2) = |1-2| = 1.0
        assert dm.values[0, 1] == pytest.approx(1.0)
        assert dm.values[1, 0] == pytest.approx(1.0)

    def test_level_residue(self, mini_pdb):
        assert bv.read_pdb(mini_pdb).distance_matrix("min").level == "residue"


# ---------------------------------------------------------------------------
# Multi-model handling
# ---------------------------------------------------------------------------

class TestMultiModel:
    def test_raises_without_model_arg(self, multimodel_pdb):
        s = bv.read_pdb(multimodel_pdb)
        with pytest.raises(ValueError, match="model"):
            s.distance_matrix("ca")

    def test_succeeds_with_model_arg(self, multimodel_pdb):
        s = bv.read_pdb(multimodel_pdb)
        dm = s.distance_matrix("ca", model=1)
        assert dm.shape == (1, 1)   # 1 residue in each model of multimodel.pdb

    def test_preselect_model_also_works(self, multimodel_pdb):
        s = bv.read_pdb(multimodel_pdb).select(model=1)
        dm = s.distance_matrix("ca")
        assert dm.shape == (1, 1)


# ---------------------------------------------------------------------------
# Invalid mode
# ---------------------------------------------------------------------------

class TestInvalidMode:
    def test_bad_mode_raises(self, mini_pdb):
        s = bv.read_pdb(mini_pdb)
        with pytest.raises(ValueError, match="Unknown distance_matrix mode"):
            s.distance_matrix("xyz")


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

class TestExports:
    def test_distance_matrix_exported(self):
        assert hasattr(bv, "DistanceMatrix")

    def test_isinstance(self, mini_pdb):
        dm = bv.read_pdb(mini_pdb).distance_matrix()
        assert isinstance(dm, DistanceMatrix)
