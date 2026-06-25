from pathlib import Path
from typing import Union

import pandas as pd

from ..msa import MSA
from ..structure import Structure
from .fasta import read_fasta, write_fasta, read_fasta_sequences, write_fasta_sequences
from .stockholm import read_stockholm, write_stockholm
from .clustal import read_clustal, write_clustal
from .a3m import read_a3m, write_a3m
from .pdb import read_pdb, write_pdb
from .mmcif import read_mmcif, write_mmcif

__all__ = [
    "read_fasta", "write_fasta",
    "read_fasta_sequences", "write_fasta_sequences",
    "read_stockholm", "write_stockholm",
    "read_clustal", "write_clustal",
    "read_a3m", "write_a3m",
    "read", "write",
    "read_sequences", "write_sequences",
    "read_pdb", "write_pdb",
    "read_mmcif", "write_mmcif",
    "read_structure", "write_structure",
]

# --- structure readers/writers (→ Structure) --------------------------------

_STRUCTURE_READERS = {
    ".pdb":   read_pdb,
    ".ent":   read_pdb,
    ".cif":   read_mmcif,
    ".mmcif": read_mmcif,
}

_STRUCTURE_WRITERS = {
    ".pdb":   write_pdb,
    ".ent":   write_pdb,
    ".cif":   write_mmcif,
    ".mmcif": write_mmcif,
}


def read_structure(filepath: Union[str, Path]) -> Structure:
    """Auto-detect format from extension and parse a structure file.

    Supported extensions: ``.pdb``, ``.ent`` (PDB format), ``.cif``,
    ``.mmcif`` (mmCIF format).
    """
    ext = Path(filepath).suffix.lower()
    reader = _STRUCTURE_READERS.get(ext)
    if reader is None:
        raise ValueError(
            f"Unrecognised extension '{ext}'. Supported: {sorted(_STRUCTURE_READERS)}"
        )
    return reader(filepath)


def write_structure(
    structure: Structure,
    filepath: Union[str, Path],
    **kwargs,
) -> None:
    """Auto-detect format from extension and write a structure file.

    Extra keyword arguments are forwarded to the format writer
    (e.g. ``renumber`` for PDB).
    """
    ext = Path(filepath).suffix.lower()
    writer = _STRUCTURE_WRITERS.get(ext)
    if writer is None:
        raise ValueError(
            f"Unrecognised extension '{ext}'. Supported: {sorted(_STRUCTURE_WRITERS)}"
        )
    writer(structure, filepath, **kwargs)


# --- alignment readers/writers (→ MSA) -------------------------------------

_READERS = {
    ".fa": read_fasta,
    ".fasta": read_fasta,
    ".fas": read_fasta,
    ".fna": read_fasta,
    ".faa": read_fasta,
    ".afa": read_fasta,
    ".sto": read_stockholm,
    ".stockholm": read_stockholm,
    ".stk": read_stockholm,
    ".aln": read_clustal,
    ".clw": read_clustal,
    ".clustal": read_clustal,
    ".a3m": read_a3m,
}

_WRITERS = {
    ".fa": write_fasta,
    ".fasta": write_fasta,
    ".fas": write_fasta,
    ".fna": write_fasta,
    ".faa": write_fasta,
    ".afa": write_fasta,
    ".sto": write_stockholm,
    ".stockholm": write_stockholm,
    ".stk": write_stockholm,
    ".aln": write_clustal,
    ".clw": write_clustal,
    ".clustal": write_clustal,
    ".a3m": write_a3m,
    ".csv": None,  # handled inline in write()
}

# --- unaligned sequence readers/writers (→ DataFrame) ----------------------

_SEQ_READERS = {
    ".fa": read_fasta_sequences,
    ".fasta": read_fasta_sequences,
    ".fas": read_fasta_sequences,
    ".fna": read_fasta_sequences,
    ".faa": read_fasta_sequences,
}

_SEQ_WRITERS = {
    ".fa": write_fasta_sequences,
    ".fasta": write_fasta_sequences,
    ".fas": write_fasta_sequences,
    ".fna": write_fasta_sequences,
    ".faa": write_fasta_sequences,
    ".csv": None,  # handled inline in write_sequences()
}


# --- dispatch functions -----------------------------------------------------

def read(filepath: Union[str, Path]) -> MSA:
    """Auto-detect format from extension and parse an alignment into an MSA."""
    ext = Path(filepath).suffix.lower()
    reader = _READERS.get(ext)
    if reader is None:
        raise ValueError(
            f"Unrecognised extension '{ext}'. Supported: {sorted(_READERS)}"
        )
    return reader(filepath)


def write(msa: MSA, filepath: Union[str, Path], **kwargs) -> None:
    """Auto-detect format from extension and write an MSA.

    Extra keyword arguments are forwarded to the format writer
    (e.g. line_width for FASTA/Stockholm/Clustal).
    CSV writes one row per sequence with all metadata columns included.
    """
    ext = Path(filepath).suffix.lower()

    if ext == ".csv":
        df = msa.to_sequences().rename("sequence").to_frame()
        if msa.metadata is not None:
            df = df.join(msa.metadata)
        df.to_csv(filepath, **kwargs)
        return

    writer = _WRITERS.get(ext)
    if writer is None:
        raise ValueError(
            f"Unrecognised extension '{ext}'. "
            f"Supported: {sorted(k for k in _WRITERS if k != '.csv')} + .csv"
        )
    writer(msa, filepath, **kwargs)


def read_sequences(filepath: Union[str, Path]) -> pd.DataFrame:
    """Auto-detect format from extension and read unaligned sequences into a DataFrame."""
    ext = Path(filepath).suffix.lower()
    reader = _SEQ_READERS.get(ext)
    if reader is None:
        raise ValueError(
            f"Unrecognised extension '{ext}'. Supported: {sorted(_SEQ_READERS)}"
        )
    return reader(filepath)


def write_sequences(
    df: pd.DataFrame,
    filepath: Union[str, Path],
    **kwargs,
) -> None:
    """Auto-detect format from extension and write a sequence DataFrame.

    Extra keyword arguments are forwarded to the format writer.
    CSV writes the full DataFrame using pandas to_csv.
    """
    ext = Path(filepath).suffix.lower()

    if ext == ".csv":
        df.to_csv(filepath, **kwargs)
        return

    writer = _SEQ_WRITERS.get(ext)
    if writer is None:
        raise ValueError(
            f"Unrecognised extension '{ext}'. "
            f"Supported: {sorted(k for k in _SEQ_WRITERS if k != '.csv')} + .csv"
        )
    writer(df, filepath, **kwargs)
