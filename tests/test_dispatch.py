"""Tests for read_structure / write_structure extension dispatch."""
import pytest
import bioviper2 as bv
from bioviper2 import Structure


class TestReadStructureDispatch:
    def test_pdb_extension(self, mini_pdb):
        s = bv.read_structure(mini_pdb)
        assert isinstance(s, Structure)

    def test_cif_extension(self, mini_cif):
        s = bv.read_structure(mini_cif)
        assert isinstance(s, Structure)

    def test_unknown_extension_raises(self, tmp_path):
        bad = tmp_path / "foo.xyz"
        bad.write_text("")
        with pytest.raises(ValueError, match="Unrecognised extension"):
            bv.read_structure(bad)


class TestWriteStructureDispatch:
    def test_pdb_extension(self, mini_pdb, tmp_path):
        s = bv.read_structure(mini_pdb)
        out = tmp_path / "out.pdb"
        bv.write_structure(s, out)
        s2 = bv.read_structure(out)
        assert s2.n_atoms == s.n_atoms

    def test_cif_extension(self, mini_cif, tmp_path):
        s = bv.read_structure(mini_cif)
        out = tmp_path / "out.cif"
        bv.write_structure(s, out)
        s2 = bv.read_structure(out)
        assert s2.n_atoms == s.n_atoms

    def test_unknown_extension_raises(self, mini_pdb, tmp_path):
        s = bv.read_structure(mini_pdb)
        with pytest.raises(ValueError, match="Unrecognised extension"):
            bv.write_structure(s, tmp_path / "out.xyz")


class TestTopLevelExports:
    def test_structure_class_exported(self):
        assert hasattr(bv, "Structure")

    def test_read_structure_exported(self):
        assert hasattr(bv, "read_structure")

    def test_write_structure_exported(self):
        assert hasattr(bv, "write_structure")

    def test_read_pdb_exported(self):
        assert hasattr(bv, "read_pdb")

    def test_write_pdb_exported(self):
        assert hasattr(bv, "write_pdb")

    def test_read_mmcif_exported(self):
        assert hasattr(bv, "read_mmcif")

    def test_write_mmcif_exported(self):
        assert hasattr(bv, "write_mmcif")
