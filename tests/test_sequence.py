"""Tests for Structure.sequence() and Structure.sequence_residues()."""
import numpy as np
import pandas as pd
import pytest

import bioviper2 as bv
from bioviper2 import Structure


# ---------------------------------------------------------------------------
# Helpers — tiny in-memory structures
# ---------------------------------------------------------------------------

def _simple_structure():
    """ALA, GLY, SER — backbone only (no CB); each has a CA."""
    coords = np.array([
        [1., 0., 0.], [2., 0., 0.], [3., 0., 0.], [4., 0., 0.],   # ALA N/CA/C/O
        [5., 0., 0.], [6., 0., 0.], [7., 0., 0.], [8., 0., 0.],   # GLY
        [9., 0., 0.], [10.,0., 0.],[11.,0., 0.],[12.,0., 0.],      # SER
    ], dtype=float)
    atoms = pd.DataFrame({
        "atom_name": ["N","CA","C","O"] * 3,
        "res_name":  ["ALA"]*4 + ["GLY"]*4 + ["SER"]*4,
        "chain_id":  ["A"] * 12,
        "res_seq":   [1,1,1,1, 2,2,2,2, 3,3,3,3],
    })
    return Structure(coords, atoms)


def _missing_ca_structure():
    """ALA has CA, MET-like has no CA (simulates missing/partial residue)."""
    coords = np.array([
        [0., 0., 0.],  # ALA CA
        [1., 0., 0.],  # SER N  (no CA for this residue)
    ], dtype=float)
    atoms = pd.DataFrame({
        "atom_name": ["CA", "N"],
        "res_name":  ["ALA", "SER"],
        "chain_id":  ["A", "A"],
        "res_seq":   [1, 2],
    })
    return Structure(coords, atoms)


def _nonstandard_structure():
    """MSE (selenomethionine → M) and an unknown residue (XYZ → X)."""
    coords = np.array([
        [0., 0., 0.],
        [1., 0., 0.],
        [2., 0., 0.],
    ], dtype=float)
    atoms = pd.DataFrame({
        "atom_name": ["CA", "CA", "CA"],
        "res_name":  ["MSE", "ALA", "XYZ"],
        "chain_id":  ["A", "A", "A"],
        "res_seq":   [1, 2, 3],
    })
    return Structure(coords, atoms)


# ---------------------------------------------------------------------------
# sequence() — basic behaviour
# ---------------------------------------------------------------------------

class TestSequence:
    def test_all_residues(self):
        s = _simple_structure()
        assert s.sequence() == "AGS"

    def test_ca_only_same_when_all_have_ca(self):
        s = _simple_structure()
        assert s.sequence(ca_only=True) == "AGS"

    def test_ca_only_drops_residue_without_ca(self):
        s = _missing_ca_structure()
        # SER residue has no CA → excluded from ca_only
        assert s.sequence() == "AS"      # both residues present in .residues
        assert s.sequence(ca_only=True) == "A"   # only ALA has CA

    def test_ca_only_length_matches_distance_matrix(self):
        s = _missing_ca_structure()
        dm = s.distance_matrix("ca")
        assert len(s.sequence(ca_only=True)) == dm.shape[0]

    def test_nonstandard_mse_maps_to_M(self):
        s = _nonstandard_structure()
        seq = s.sequence()
        assert seq[0] == "M"

    def test_unknown_residue_maps_to_X(self):
        s = _nonstandard_structure()
        seq = s.sequence()
        assert seq[2] == "X"

    def test_mini_pdb_full(self, mini_pdb):
        # mini.pdb: chain A = ALA(1), GLY(2); chain B = LEU(1), SER(2)
        s = bv.read_pdb(mini_pdb)
        assert s.sequence() == "AGLS"

    def test_mini_pdb_chain_a(self, mini_pdb):
        s = bv.read_pdb(mini_pdb)
        assert s.select(chain="A").sequence() == "AG"

    def test_mini_pdb_chain_b(self, mini_pdb):
        s = bv.read_pdb(mini_pdb)
        assert s.select(chain="B").sequence() == "LS"

    def test_returns_str(self):
        s = _simple_structure()
        assert isinstance(s.sequence(), str)

    def test_empty_structure(self):
        """Empty structure (no atoms) → empty sequence."""
        coords = np.zeros((0, 3))
        atoms = pd.DataFrame(columns=["atom_name","res_name","chain_id","res_seq"])
        s = Structure(coords, atoms, validate=False)
        assert s.sequence() == ""
        assert s.sequence(ca_only=True) == ""


# ---------------------------------------------------------------------------
# sequence_residues() — the residue table with one_letter column
# ---------------------------------------------------------------------------

class TestSequenceResidues:
    def test_returns_dataframe(self):
        s = _simple_structure()
        sr = s.sequence_residues()
        assert isinstance(sr, pd.DataFrame)

    def test_has_one_letter_column(self):
        s = _simple_structure()
        sr = s.sequence_residues()
        assert "one_letter" in sr.columns

    def test_one_letter_matches_sequence(self):
        s = _simple_structure()
        sr = s.sequence_residues()
        assert "".join(sr["one_letter"]) == s.sequence()

    def test_ca_only_length_matches_distance_matrix(self):
        s = _missing_ca_structure()
        sr = s.sequence_residues(ca_only=True)
        dm = s.distance_matrix("ca")
        assert len(sr) == dm.shape[0]

    def test_has_residue_key_columns(self):
        s = _simple_structure()
        sr = s.sequence_residues()
        for col in ("chain_id", "res_seq", "icode", "res_name"):
            assert col in sr.columns

    def test_fresh_range_index(self):
        s = _simple_structure()
        sr = s.sequence_residues()
        assert list(sr.index) == list(range(len(sr)))

    def test_nonstandard_residue_in_one_letter(self):
        s = _nonstandard_structure()
        sr = s.sequence_residues()
        assert sr.iloc[0]["one_letter"] == "M"   # MSE
        assert sr.iloc[2]["one_letter"] == "X"   # XYZ → unknown


# ---------------------------------------------------------------------------
# Export checks
# ---------------------------------------------------------------------------

class TestExports:
    def test_aa3to1_dict_exported(self):
        from bioviper2.structure import _AA3TO1
        assert "ALA" in _AA3TO1
        assert _AA3TO1["ALA"] == "A"
        assert "MSE" in _AA3TO1

    def test_res_name_to_one_exported(self):
        from bioviper2.structure import _res_name_to_one
        assert _res_name_to_one("ALA") == "A"
        assert _res_name_to_one("xyz") == "X"   # case-insensitive
        assert _res_name_to_one("  ALA  ") == "A"  # strips whitespace
