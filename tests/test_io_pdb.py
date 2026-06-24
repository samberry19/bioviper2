"""Tests for PDB reading, writing, and round-tripping."""
import numpy as np
import pytest
import tempfile
from pathlib import Path

import bioviper2 as bv
from bioviper2.io.pdb import _guess_element, _format_atom_name


# ---------------------------------------------------------------------------
# _guess_element unit tests (the CA calcium vs alpha-carbon case)
# ---------------------------------------------------------------------------

class TestGuessElement:
    def test_alpha_carbon(self):
        # " CA " — standard ATOM, leading space → single letter C
        assert _guess_element(" CA ", False) == "C"

    def test_calcium_hetatm(self):
        # "CA  " — flush-left HETATM, known 2-letter element
        assert _guess_element("CA  ", True) == "CA"

    def test_iron(self):
        assert _guess_element("FE  ", True) == "FE"

    def test_zinc(self):
        assert _guess_element("ZN  ", True) == "ZN"

    def test_nitrogen(self):
        assert _guess_element(" N  ", False) == "N"

    def test_oxygen(self):
        assert _guess_element(" O  ", False) == "O"

    def test_sulphur(self):
        assert _guess_element(" SG ", False) == "S"

    def test_magnesium_hetatm(self):
        assert _guess_element(" MG ", True) == "MG"

    def test_hd_hydrogen(self):
        # "HD11" — leading H, single-letter element H
        assert _guess_element("HD11", False) == "H"


# ---------------------------------------------------------------------------
# _format_atom_name unit tests
# ---------------------------------------------------------------------------

class TestFormatAtomName:
    def test_ca_backbone(self):
        assert _format_atom_name("CA", "C") == " CA "

    def test_n_backbone(self):
        assert _format_atom_name("N", "N") == " N  "

    def test_o_backbone(self):
        assert _format_atom_name("O", "O") == " O  "

    def test_og1(self):
        assert _format_atom_name("OG1", "O") == " OG1"

    def test_4char_name(self):
        assert _format_atom_name("HD11", "H") == "HD11"

    def test_iron_hetatm(self):
        assert _format_atom_name("FE", "FE") == "FE  "

    def test_zinc_hetatm(self):
        assert _format_atom_name("ZN", "ZN") == "ZN  "

    def test_cb(self):
        assert _format_atom_name("CB", "C") == " CB "


# ---------------------------------------------------------------------------
# Reading tests
# ---------------------------------------------------------------------------

class TestReadPDB:
    def test_basic_read(self, mini_pdb):
        s = bv.read_pdb(mini_pdb)
        assert s.n_atoms == 16
        assert s.n_chains == 2
        assert set(s.chains) == {"A", "B"}
        assert s.n_residues == 4
        assert s.n_models == 1

    def test_coords_shape(self, mini_pdb):
        s = bv.read_pdb(mini_pdb)
        assert s.coords.shape == (16, 3)
        assert s.coords.dtype == np.float64

    def test_first_atom_coords(self, mini_pdb):
        s = bv.read_pdb(mini_pdb)
        np.testing.assert_allclose(s.coords[0], [1.0, 2.0, 3.0])

    def test_b_factors(self, mini_pdb):
        s = bv.read_pdb(mini_pdb)
        assert s.atoms.iloc[0]["b_factor"] == pytest.approx(10.0)
        assert s.atoms.iloc[4]["b_factor"] == pytest.approx(20.0)

    def test_chain_id(self, mini_pdb):
        s = bv.read_pdb(mini_pdb)
        assert s.atoms.iloc[0]["chain_id"] == "A"
        assert s.atoms.iloc[8]["chain_id"] == "B"

    def test_atom_serial(self, mini_pdb):
        s = bv.read_pdb(mini_pdb)
        assert int(s.atoms.iloc[0]["atom_serial"]) == 1
        assert int(s.atoms.iloc[8]["atom_serial"]) == 10

    def test_hetero_flag(self, hetatm_pdb):
        s = bv.read_pdb(hetatm_pdb)
        assert not s.atoms.iloc[0]["hetero"]  # protein N
        assert s.atoms.iloc[2]["hetero"]       # HOH

    def test_hetatm_element_fe(self, hetatm_pdb):
        s = bv.read_pdb(hetatm_pdb)
        fe = s.select(element="FE")
        assert fe.n_atoms == 1

    def test_hetatm_element_zn(self, hetatm_pdb):
        s = bv.read_pdb(hetatm_pdb)
        zn = s.select(element="ZN")
        assert zn.n_atoms == 1

    def test_multimodel(self, multimodel_pdb):
        s = bv.read_pdb(multimodel_pdb)
        assert s.n_models == 2
        assert s.n_atoms == 4
        assert set(s.atoms["model"].unique()) == {1, 2}

    def test_plddt_alias(self, af2_pdb):
        s = bv.read_pdb(af2_pdb)
        np.testing.assert_array_equal(s.plddt.values, s.atoms["b_factor"].values)
        assert s.atoms.iloc[0]["b_factor"] == pytest.approx(95.5)

    def test_icode(self, icode_pdb):
        s = bv.read_pdb(icode_pdb)
        assert s.n_residues == 4  # res 10, 10A, 10B, 11 are all distinct
        icodes = s.atoms["icode"].unique().tolist()
        assert "A" in icodes or "B" in icodes

    def test_empty_raises(self, tmp_path):
        empty = tmp_path / "empty.pdb"
        empty.write_text("REMARK empty\nEND\n")
        with pytest.raises(ValueError, match="No atom records"):
            bv.read_pdb(empty)

    def test_read_structure_dispatch(self, mini_pdb):
        s = bv.read_structure(mini_pdb)
        assert isinstance(s, bv.Structure)
        assert s.n_atoms == 16


# ---------------------------------------------------------------------------
# Writing / round-trip tests
# ---------------------------------------------------------------------------

class TestWritePDB:
    def test_round_trip_coords(self, mini_pdb, tmp_path):
        s1 = bv.read_pdb(mini_pdb)
        out = tmp_path / "out.pdb"
        bv.write_pdb(s1, out)
        s2 = bv.read_pdb(out)
        np.testing.assert_allclose(s1.coords, s2.coords, atol=1e-3)

    def test_round_trip_n_atoms(self, mini_pdb, tmp_path):
        s1 = bv.read_pdb(mini_pdb)
        out = tmp_path / "out.pdb"
        bv.write_pdb(s1, out)
        s2 = bv.read_pdb(out)
        assert s1.n_atoms == s2.n_atoms

    def test_round_trip_chain_ids(self, mini_pdb, tmp_path):
        s1 = bv.read_pdb(mini_pdb)
        out = tmp_path / "out.pdb"
        bv.write_pdb(s1, out)
        s2 = bv.read_pdb(out)
        assert set(s1.chains) == set(s2.chains)

    def test_round_trip_b_factors(self, mini_pdb, tmp_path):
        s1 = bv.read_pdb(mini_pdb)
        out = tmp_path / "out.pdb"
        bv.write_pdb(s1, out)
        s2 = bv.read_pdb(out)
        np.testing.assert_allclose(
            s1.atoms["b_factor"].values,
            s2.atoms["b_factor"].values,
            atol=1e-2,
        )

    def test_round_trip_atom_names(self, mini_pdb, tmp_path):
        s1 = bv.read_pdb(mini_pdb)
        out = tmp_path / "out.pdb"
        bv.write_pdb(s1, out)
        s2 = bv.read_pdb(out)
        assert list(s1.atoms["atom_name"]) == list(s2.atoms["atom_name"])

    def test_atom_name_justification_in_file(self, mini_pdb, tmp_path):
        """The raw written CA field must be ' CA ' (one leading space)."""
        s = bv.read_pdb(mini_pdb)
        out = tmp_path / "out.pdb"
        bv.write_pdb(s, out)
        with open(out) as fh:
            for line in fh:
                if line.startswith("ATOM") and line[12:16].strip() == "CA":
                    assert line[12:16] == " CA "
                    break

    def test_multimodel_roundtrip(self, multimodel_pdb, tmp_path):
        s1 = bv.read_pdb(multimodel_pdb)
        out = tmp_path / "out.pdb"
        bv.write_pdb(s1, out)
        s2 = bv.read_pdb(out)
        assert s1.n_models == s2.n_models
        assert s1.n_atoms == s2.n_atoms

    def test_multimodel_model_endmdl_present(self, multimodel_pdb, tmp_path):
        s = bv.read_pdb(multimodel_pdb)
        out = tmp_path / "out.pdb"
        bv.write_pdb(s, out)
        text = out.read_text()
        assert "MODEL" in text
        assert "ENDMDL" in text

    def test_write_structure_dispatch(self, mini_pdb, tmp_path):
        s = bv.read_pdb(mini_pdb)
        out = tmp_path / "out.pdb"
        bv.write_structure(s, out)
        s2 = bv.read_structure(out)
        assert s2.n_atoms == s.n_atoms
