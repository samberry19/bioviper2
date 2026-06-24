"""Tests for mmCIF reading, writing, and round-tripping."""
import numpy as np
import pytest
import textwrap
from pathlib import Path

import bioviper2 as bv
from bioviper2.io.mmcif import _tokenize_cif_line


# ---------------------------------------------------------------------------
# Tokenizer unit tests
# ---------------------------------------------------------------------------

class TestTokenizer:
    def test_bare_tokens(self):
        assert _tokenize_cif_line("ATOM 1 C CA") == ["ATOM", "1", "C", "CA"]

    def test_single_quoted(self):
        assert _tokenize_cif_line("'hello world' foo") == ["hello world", "foo"]

    def test_double_quoted(self):
        assert _tokenize_cif_line('"hello world" bar') == ["hello world", "bar"]

    def test_cif_null_dot(self):
        tokens = _tokenize_cif_line("ATOM . CA")
        assert tokens == ["ATOM", ".", "CA"]

    def test_cif_null_question(self):
        tokens = _tokenize_cif_line("? foo")
        assert tokens == ["?", "foo"]

    def test_empty_line(self):
        assert _tokenize_cif_line("") == []

    def test_comment_ignored(self):
        assert _tokenize_cif_line("# this is a comment") == []

    def test_mixed(self):
        tokens = _tokenize_cif_line("ATOM 1 C 'some name' . 3.14")
        assert tokens == ["ATOM", "1", "C", "some name", ".", "3.14"]


# ---------------------------------------------------------------------------
# Reading tests
# ---------------------------------------------------------------------------

class TestReadMMCIF:
    def test_basic_read(self, mini_cif):
        s = bv.read_mmcif(mini_cif)
        assert s.n_atoms == 16
        assert s.n_chains == 2
        assert set(s.chains) == {"A", "B"}
        assert s.n_residues == 4

    def test_coords_shape(self, mini_cif):
        s = bv.read_mmcif(mini_cif)
        assert s.coords.shape == (16, 3)

    def test_first_atom_coords(self, mini_cif):
        s = bv.read_mmcif(mini_cif)
        np.testing.assert_allclose(s.coords[0], [1.0, 2.0, 3.0])

    def test_b_factors(self, mini_cif):
        s = bv.read_mmcif(mini_cif)
        assert s.atoms.iloc[0]["b_factor"] == pytest.approx(10.0)

    def test_label_asym_id_preserved(self, mini_cif):
        s = bv.read_mmcif(mini_cif)
        # label_asym_id should be present (not all NA)
        assert "label_asym_id" in s.atoms.columns

    def test_label_seq_id_preserved(self, mini_cif):
        s = bv.read_mmcif(mini_cif)
        assert "label_seq_id" in s.atoms.columns

    def test_no_atom_site_raises(self, tmp_path):
        bad = tmp_path / "bad.cif"
        bad.write_text("data_test\n# no atom site\n")
        with pytest.raises(ValueError, match="_atom_site"):
            bv.read_mmcif(bad)

    def test_multiline_semicolon_raises(self, tmp_path):
        bad = tmp_path / "multiline.cif"
        bad.write_text(textwrap.dedent("""\
            data_test
            loop_
            _atom_site.group_PDB
            _atom_site.id
            _atom_site.type_symbol
            _atom_site.label_atom_id
            _atom_site.label_alt_id
            _atom_site.label_comp_id
            _atom_site.label_asym_id
            _atom_site.auth_asym_id
            _atom_site.label_entity_id
            _atom_site.label_seq_id
            _atom_site.auth_seq_id
            _atom_site.pdbx_PDB_ins_code
            _atom_site.Cartn_x
            _atom_site.Cartn_y
            _atom_site.Cartn_z
            _atom_site.occupancy
            _atom_site.B_iso_or_equiv
            _atom_site.pdbx_formal_charge
            _atom_site.pdbx_PDB_model_num
            ;
            multiline value
            ;
        """))
        with pytest.raises(ValueError, match="Multiline"):
            bv.read_mmcif(bad)

    def test_read_structure_dispatch(self, mini_cif):
        s = bv.read_structure(mini_cif)
        assert isinstance(s, bv.Structure)


# ---------------------------------------------------------------------------
# Cross-format equivalence
# ---------------------------------------------------------------------------

class TestCrossFormat:
    def test_same_n_atoms(self, mini_pdb, mini_cif):
        sp = bv.read_pdb(mini_pdb)
        sc = bv.read_mmcif(mini_cif)
        assert sp.n_atoms == sc.n_atoms

    def test_same_coords(self, mini_pdb, mini_cif):
        sp = bv.read_pdb(mini_pdb)
        sc = bv.read_mmcif(mini_cif)
        np.testing.assert_allclose(sp.coords, sc.coords, atol=1e-3)

    def test_same_atom_names(self, mini_pdb, mini_cif):
        sp = bv.read_pdb(mini_pdb)
        sc = bv.read_mmcif(mini_cif)
        assert list(sp.atoms["atom_name"]) == list(sc.atoms["atom_name"])

    def test_same_chain_ids(self, mini_pdb, mini_cif):
        sp = bv.read_pdb(mini_pdb)
        sc = bv.read_mmcif(mini_cif)
        assert list(sp.atoms["chain_id"]) == list(sc.atoms["chain_id"])

    def test_same_res_names(self, mini_pdb, mini_cif):
        sp = bv.read_pdb(mini_pdb)
        sc = bv.read_mmcif(mini_cif)
        assert list(sp.atoms["res_name"]) == list(sc.atoms["res_name"])

    def test_same_b_factors(self, mini_pdb, mini_cif):
        sp = bv.read_pdb(mini_pdb)
        sc = bv.read_mmcif(mini_cif)
        np.testing.assert_allclose(
            sp.atoms["b_factor"].values,
            sc.atoms["b_factor"].values,
            atol=1e-2,
        )


# ---------------------------------------------------------------------------
# Writing / round-trip tests
# ---------------------------------------------------------------------------

class TestWriteMMCIF:
    def test_round_trip_coords(self, mini_cif, tmp_path):
        s1 = bv.read_mmcif(mini_cif)
        out = tmp_path / "out.cif"
        bv.write_mmcif(s1, out)
        s2 = bv.read_mmcif(out)
        np.testing.assert_allclose(s1.coords, s2.coords, atol=1e-3)

    def test_round_trip_n_atoms(self, mini_cif, tmp_path):
        s1 = bv.read_mmcif(mini_cif)
        out = tmp_path / "out.cif"
        bv.write_mmcif(s1, out)
        s2 = bv.read_mmcif(out)
        assert s1.n_atoms == s2.n_atoms

    def test_round_trip_chain_ids(self, mini_cif, tmp_path):
        s1 = bv.read_mmcif(mini_cif)
        out = tmp_path / "out.cif"
        bv.write_mmcif(s1, out)
        s2 = bv.read_mmcif(out)
        assert list(s1.atoms["chain_id"]) == list(s2.atoms["chain_id"])

    def test_round_trip_b_factors(self, mini_cif, tmp_path):
        s1 = bv.read_mmcif(mini_cif)
        out = tmp_path / "out.cif"
        bv.write_mmcif(s1, out)
        s2 = bv.read_mmcif(out)
        np.testing.assert_allclose(
            s1.atoms["b_factor"].values,
            s2.atoms["b_factor"].values,
            atol=1e-2,
        )

    def test_write_structure_dispatch(self, mini_cif, tmp_path):
        s = bv.read_mmcif(mini_cif)
        out = tmp_path / "out.cif"
        bv.write_structure(s, out)
        s2 = bv.read_structure(out)
        assert s2.n_atoms == s.n_atoms
