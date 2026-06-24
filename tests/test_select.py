"""Tests for Structure.select() and .iloc."""
import numpy as np
import pandas as pd
import pytest

from bioviper2 import Structure


def _two_chain():
    """2-chain structure with waters and a metal ion (19 atoms total)."""
    coords = np.arange(19 * 3, dtype=float).reshape(19, 3)
    atoms = pd.DataFrame({
        "atom_name": (
            ["N", "CA", "C", "O"] * 2   # chain A, res 1-2
            + ["N", "CA", "C", "O"] * 2   # chain B, res 1-2
            + ["O", "FE", "ZN"]            # waters/ligands in A
        ),
        "res_name": (
            ["ALA"] * 4 + ["GLY"] * 4
            + ["LEU"] * 4 + ["SER"] * 4
            + ["HOH", "HEM", "ZN"]
        ),
        "chain_id": (
            ["A"] * 8 + ["B"] * 8 + ["A"] * 3
        ),
        "res_seq": (
            [1, 1, 1, 1, 2, 2, 2, 2]   # chain A protein
            + [1, 1, 1, 1, 2, 2, 2, 2]  # chain B protein
            + [100, 200, 201]            # HETATM
        ),
        "element": (
            ["N", "C", "C", "O"] * 4
            + ["O", "FE", "ZN"]
        ),
        "hetero": [False] * 16 + [True] * 3,
        "record_name": ["ATOM"] * 16 + ["HETATM"] * 3,
        "model": [1] * 19,
    })
    return Structure(coords, atoms)


class TestSelectBasic:
    def test_chain_scalar(self):
        s = _two_chain()
        sa = s.select(chain="A")
        assert set(sa.atoms["chain_id"].unique()) == {"A"}
        assert sa.n_atoms == 11  # 8 protein + 3 hetero

    def test_chain_list(self):
        s = _two_chain()
        sab = s.select(chain=["A", "B"])
        assert sab.n_atoms == 19

    def test_resi_scalar(self):
        s = _two_chain()
        r1 = s.select(resi=1)
        # res_seq 1 exists in both chains A and B (8 atoms each, but A also has hetero at 100,200,201)
        # Only protein residue 1: 4 (A) + 4 (B) = 8
        assert (r1.atoms["res_seq"] == 1).all()

    def test_resi_list(self):
        s = _two_chain()
        r = s.select(resi=[1, 2])
        assert set(r.atoms["res_seq"].unique()) == {1, 2}

    def test_resi_range(self):
        s = _two_chain()
        r = s.select(resi=range(1, 3))
        assert set(r.atoms["res_seq"].unique()) == {1, 2}

    def test_atom_name(self):
        s = _two_chain()
        ca = s.select(atom="CA")
        assert (ca.atoms["atom_name"] == "CA").all()

    def test_atom_name_list(self):
        s = _two_chain()
        bb = s.select(atom=["N", "CA", "C", "O"], hetero=False)
        assert set(bb.atoms["atom_name"].unique()) <= {"N", "CA", "C", "O"}
        assert not bb.atoms["hetero"].any()

    def test_hetero_false(self):
        s = _two_chain()
        prot = s.select(hetero=False)
        assert not prot.atoms["hetero"].any()
        assert prot.n_atoms == 16

    def test_hetero_true(self):
        s = _two_chain()
        het = s.select(hetero=True)
        assert het.atoms["hetero"].all()
        assert het.n_atoms == 3

    def test_element(self):
        s = _two_chain()
        fe = s.select(element="FE")
        assert fe.n_atoms == 1
        assert fe.atoms.iloc[0]["element"] == "FE"

    def test_combined_and(self):
        s = _two_chain()
        ca_a = s.select(chain="A", atom="CA")
        assert (ca_a.atoms["chain_id"] == "A").all()
        assert (ca_a.atoms["atom_name"] == "CA").all()

    def test_model(self):
        coords = np.ones((4, 3))
        atoms = pd.DataFrame({
            "atom_name": ["CA"] * 4,
            "res_name": ["ALA"] * 4,
            "chain_id": ["A"] * 4,
            "res_seq": [1] * 4,
            "model": [1, 1, 2, 2],
        })
        s = Structure(coords, atoms)
        m1 = s.select(model=1)
        assert m1.n_atoms == 2
        assert (m1.atoms["model"] == 1).all()

    def test_empty_result(self):
        s = _two_chain()
        empty = s.select(chain="Z")
        assert empty.n_atoms == 0
        assert len(empty) == 0

    def test_fresh_rangeindex(self):
        s = _two_chain()
        sub = s.select(chain="B")
        assert list(sub.atoms.index) == list(range(sub.n_atoms))

    def test_coords_atoms_aligned(self):
        """After select, coords[i] and atoms.iloc[i] must correspond."""
        s = _two_chain()
        sub = s.select(chain="B")
        # The original coords for chain B atoms started at index 8
        orig_b_idx = s.atoms.index[s.atoms["chain_id"] == "B"].tolist()
        np.testing.assert_array_equal(sub.coords, s.coords[orig_b_idx])

    def test_res_name(self):
        s = _two_chain()
        hoh = s.select(res_name="HOH")
        assert hoh.n_atoms == 1
        assert hoh.atoms.iloc[0]["res_name"] == "HOH"


class TestIloc:
    def test_slice(self):
        s = _two_chain()
        sub = s.iloc[:4]
        assert sub.n_atoms == 4
        np.testing.assert_array_equal(sub.coords, s.coords[:4])

    def test_list(self):
        s = _two_chain()
        sub = s.iloc[[0, 5, 10]]
        assert sub.n_atoms == 3

    def test_fresh_index(self):
        s = _two_chain()
        sub = s.iloc[10:15]
        assert list(sub.atoms.index) == list(range(5))
