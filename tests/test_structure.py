"""Tests for Structure construction, properties, and conversion."""
import numpy as np
import pandas as pd
import pytest

from bioviper2 import Structure


def _make_simple(n=4):
    """Build a minimal n-atom Structure in memory."""
    coords = np.arange(n * 3, dtype=float).reshape(n, 3)
    atoms = pd.DataFrame({
        "atom_name":  ["N", "CA", "C", "O"][:n],
        "res_name":   ["ALA"] * n,
        "chain_id":   ["A"] * n,
        "res_seq":    [1] * n,
    })
    return Structure(coords, atoms)


class TestConstruction:
    def test_basic_creation(self):
        s = _make_simple(4)
        assert s.n_atoms == 4
        assert s.shape == (4, 3)
        assert len(s) == 4

    def test_coords_dtype(self):
        coords = np.ones((3, 3), dtype=np.float32)
        atoms = pd.DataFrame({
            "atom_name": ["N", "CA", "C"],
            "res_name": ["ALA"] * 3,
            "chain_id": ["A"] * 3,
            "res_seq": [1] * 3,
        })
        s = Structure(coords, atoms)
        assert s.coords.dtype == np.float64

    def test_wrong_coords_shape(self):
        with pytest.raises(ValueError, match="coords must be shape"):
            Structure(np.ones((4, 2)), pd.DataFrame({
                "atom_name": ["N"], "res_name": ["ALA"],
                "chain_id": ["A"], "res_seq": [1],
            }))

    def test_mismatched_lengths(self):
        with pytest.raises(ValueError, match="row count"):
            Structure(np.ones((5, 3)), pd.DataFrame({
                "atom_name": ["N"], "res_name": ["ALA"],
                "chain_id": ["A"], "res_seq": [1],
            }))

    def test_missing_required_column(self):
        coords = np.ones((1, 3))
        atoms = pd.DataFrame({
            "atom_name": ["N"],
            "res_name": ["ALA"],
            "chain_id": ["A"],
            # res_seq missing
        })
        with pytest.raises(ValueError, match="missing required columns"):
            Structure(coords, atoms)

    def test_defaults_filled(self):
        s = _make_simple(2)
        # Optional columns should be present and have sensible defaults
        assert "b_factor" in s.atoms.columns
        assert "occupancy" in s.atoms.columns
        assert "model" in s.atoms.columns
        assert (s.atoms["occupancy"] == 1.0).all()
        assert (s.atoms["b_factor"] == 0.0).all()
        assert (s.atoms["model"] == 1).all()

    def test_index_is_range(self):
        s = _make_simple(3)
        assert list(s.atoms.index) == [0, 1, 2]

    def test_copy_independence(self):
        s = _make_simple(4)
        s2 = s.copy()
        s2.atoms.loc[0, "b_factor"] = 99.0
        s2._coords[0, 0] = -999.0
        # original unchanged
        assert s.atoms.loc[0, "b_factor"] == 0.0
        assert s.coords[0, 0] == 0.0


class TestProperties:
    def test_n_models_single(self):
        assert _make_simple().n_models == 1

    def test_models_array(self):
        np.testing.assert_array_equal(_make_simple().models, [1])

    def test_chains_order_preserved(self):
        coords = np.ones((6, 3))
        atoms = pd.DataFrame({
            "atom_name": ["CA"] * 6,
            "res_name": ["ALA"] * 6,
            "chain_id": ["B", "B", "A", "A", "C", "C"],
            "res_seq": [1, 2, 1, 2, 1, 2],
        })
        s = Structure(coords, atoms)
        assert list(s.chains) == ["B", "A", "C"]

    def test_n_chains(self):
        assert _make_simple().n_chains == 1

    def test_residues_dataframe(self):
        s = _make_simple(4)
        r = s.residues
        assert len(r) == 1  # all 4 atoms in the same residue
        assert r.iloc[0]["res_name"] == "ALA"
        assert r.iloc[0]["chain_id"] == "A"

    def test_n_residues(self):
        coords = np.ones((4, 3))
        atoms = pd.DataFrame({
            "atom_name": ["CA"] * 4,
            "res_name": ["ALA", "ALA", "GLY", "GLY"],
            "chain_id": ["A"] * 4,
            "res_seq": [1, 1, 2, 2],
        })
        s = Structure(coords, atoms)
        assert s.n_residues == 2

    def test_plddt_alias(self):
        s = _make_simple(2)
        pd.testing.assert_series_equal(s.plddt, s.atoms["b_factor"])

    def test_b_factors_alias(self):
        s = _make_simple(2)
        pd.testing.assert_series_equal(s.b_factors, s.atoms["b_factor"])

    def test_to_dataframe(self):
        s = _make_simple(4)
        df = s.to_dataframe()
        assert "x" in df.columns
        assert "y" in df.columns
        assert "z" in df.columns
        assert len(df) == 4
        np.testing.assert_allclose(df["x"].values, s.coords[:, 0])

    def test_repr_single_model(self):
        s = _make_simple(4)
        r = repr(s)
        assert "4 atoms" in r
        assert "1 chain" in r
        assert "1 model" in r

    def test_repr_multi_model(self):
        coords = np.ones((4, 3))
        atoms = pd.DataFrame({
            "atom_name": ["CA"] * 4,
            "res_name": ["ALA"] * 4,
            "chain_id": ["A"] * 4,
            "res_seq": [1] * 4,
            "model": [1, 1, 2, 2],
        })
        s = Structure(coords, atoms)
        r = repr(s)
        assert "2 models" in r

    def test_repr_many_chains(self):
        n = 14
        coords = np.ones((n, 3))
        atoms = pd.DataFrame({
            "atom_name": ["CA"] * n,
            "res_name": ["ALA"] * n,
            "chain_id": [chr(65 + i) for i in range(n)],  # A-N
            "res_seq": [1] * n,
        })
        s = Structure(coords, atoms)
        r = repr(s)
        assert "…+" in r  # truncated
