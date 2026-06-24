"""Shared fixtures for bioviper2 structure tests."""
import pytest
from pathlib import Path

DATA = Path(__file__).parent / "data"


@pytest.fixture
def mini_pdb():
    return DATA / "mini.pdb"


@pytest.fixture
def mini_cif():
    return DATA / "mini.cif"


@pytest.fixture
def multimodel_pdb():
    return DATA / "multimodel.pdb"


@pytest.fixture
def hetatm_pdb():
    return DATA / "hetatm.pdb"


@pytest.fixture
def af2_pdb():
    return DATA / "af2.pdb"


@pytest.fixture
def icode_pdb():
    return DATA / "icode.pdb"
