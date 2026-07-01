"""Substitution matrices and conversion utilities.

Provides BLOSUM62 and a simple nucleotide matrix; additional matrices can be
added to the ``_MATRICES`` registry.  The public entry point is
:func:`as_array`, which normalises any of the supported matrix formats into a
``(alphabet, S)`` pair suitable for vectorised computation.
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# BLOSUM62 (Henikoff & Henikoff 1992)
# Stored as a flat (a, b)->score dict; both orderings are present so lookups
# never need reversal.  This is also the format expected by align._score_matrix.
# ---------------------------------------------------------------------------

_BLOSUM62_ROWS = {
    "A": {"A": 4,"R":-1,"N":-2,"D":-2,"C": 0,"Q":-1,"E":-1,"G": 0,"H":-2,"I":-1,
          "L":-1,"K":-1,"M":-1,"F":-2,"P":-1,"S": 1,"T": 0,"W":-3,"Y":-2,"V": 0,
          "B":-2,"Z":-1,"X": 0,"*":-4},
    "R": {"A":-1,"R": 5,"N": 0,"D":-2,"C":-3,"Q": 1,"E": 0,"G":-2,"H": 0,"I":-3,
          "L":-2,"K": 2,"M":-1,"F":-3,"P":-2,"S":-1,"T":-1,"W":-3,"Y":-2,"V":-3,
          "B":-1,"Z": 0,"X":-1,"*":-4},
    "N": {"A":-2,"R": 0,"N": 6,"D": 1,"C":-3,"Q": 0,"E": 0,"G": 0,"H": 1,"I":-3,
          "L":-3,"K": 0,"M":-2,"F":-3,"P":-2,"S": 1,"T": 0,"W":-4,"Y":-2,"V":-3,
          "B": 3,"Z": 0,"X":-1,"*":-4},
    "D": {"A":-2,"R":-2,"N": 1,"D": 6,"C":-3,"Q": 0,"E": 2,"G":-1,"H":-1,"I":-3,
          "L":-4,"K":-1,"M":-3,"F":-3,"P":-1,"S": 0,"T":-1,"W":-4,"Y":-3,"V":-3,
          "B": 4,"Z": 1,"X":-1,"*":-4},
    "C": {"A": 0,"R":-3,"N":-3,"D":-3,"C": 9,"Q":-3,"E":-4,"G":-3,"H":-3,"I":-1,
          "L":-1,"K":-3,"M":-1,"F":-2,"P":-3,"S":-1,"T":-1,"W":-2,"Y":-2,"V":-1,
          "B":-3,"Z":-3,"X":-2,"*":-4},
    "Q": {"A":-1,"R": 1,"N": 0,"D": 0,"C":-3,"Q": 5,"E": 2,"G":-2,"H": 0,"I":-3,
          "L":-2,"K": 1,"M": 0,"F":-3,"P":-1,"S": 0,"T":-1,"W":-2,"Y":-1,"V":-2,
          "B": 0,"Z": 3,"X":-1,"*":-4},
    "E": {"A":-1,"R": 0,"N": 0,"D": 2,"C":-4,"Q": 2,"E": 5,"G":-2,"H": 0,"I":-3,
          "L":-3,"K": 1,"M":-2,"F":-3,"P":-1,"S": 0,"T":-1,"W":-3,"Y":-2,"V":-2,
          "B": 1,"Z": 4,"X":-1,"*":-4},
    "G": {"A": 0,"R":-2,"N": 0,"D":-1,"C":-3,"Q":-2,"E":-2,"G": 6,"H":-2,"I":-4,
          "L":-4,"K":-2,"M":-3,"F":-3,"P":-2,"S": 0,"T":-2,"W":-2,"Y":-3,"V":-3,
          "B":-1,"Z":-2,"X":-1,"*":-4},
    "H": {"A":-2,"R": 0,"N": 1,"D":-1,"C":-3,"Q": 0,"E": 0,"G":-2,"H": 8,"I":-3,
          "L":-3,"K":-1,"M":-2,"F":-1,"P":-2,"S":-1,"T":-2,"W":-2,"Y": 2,"V":-3,
          "B": 0,"Z": 0,"X":-1,"*":-4},
    "I": {"A":-1,"R":-3,"N":-3,"D":-3,"C":-1,"Q":-3,"E":-3,"G":-4,"H":-3,"I": 4,
          "L": 2,"K":-3,"M": 1,"F": 0,"P":-3,"S":-2,"T":-1,"W":-3,"Y":-1,"V": 3,
          "B":-3,"Z":-3,"X":-1,"*":-4},
    "L": {"A":-1,"R":-2,"N":-3,"D":-4,"C":-1,"Q":-2,"E":-3,"G":-4,"H":-3,"I": 2,
          "L": 4,"K":-2,"M": 2,"F": 0,"P":-3,"S":-2,"T":-1,"W":-2,"Y":-1,"V": 1,
          "B":-4,"Z":-3,"X":-1,"*":-4},
    "K": {"A":-1,"R": 2,"N": 0,"D":-1,"C":-3,"Q": 1,"E": 1,"G":-2,"H":-1,"I":-3,
          "L":-2,"K": 5,"M":-1,"F":-3,"P":-1,"S": 0,"T":-1,"W":-3,"Y":-2,"V":-2,
          "B": 0,"Z": 1,"X":-1,"*":-4},
    "M": {"A":-1,"R":-1,"N":-2,"D":-3,"C":-1,"Q": 0,"E":-2,"G":-3,"H":-2,"I": 1,
          "L": 2,"K":-1,"M": 5,"F": 0,"P":-2,"S":-1,"T":-1,"W":-1,"Y":-1,"V": 1,
          "B":-3,"Z":-1,"X":-1,"*":-4},
    "F": {"A":-2,"R":-3,"N":-3,"D":-3,"C":-2,"Q":-3,"E":-3,"G":-3,"H":-1,"I": 0,
          "L": 0,"K":-3,"M": 0,"F": 6,"P":-4,"S":-2,"T":-2,"W": 1,"Y": 3,"V":-1,
          "B":-3,"Z":-3,"X":-1,"*":-4},
    "P": {"A":-1,"R":-2,"N":-2,"D":-1,"C":-3,"Q":-1,"E":-1,"G":-2,"H":-2,"I":-3,
          "L":-3,"K":-1,"M":-2,"F":-4,"P": 7,"S":-1,"T":-1,"W":-4,"Y":-3,"V":-2,
          "B":-2,"Z":-1,"X":-2,"*":-4},
    "S": {"A": 1,"R":-1,"N": 1,"D": 0,"C":-1,"Q": 0,"E": 0,"G": 0,"H":-1,"I":-2,
          "L":-2,"K": 0,"M":-1,"F":-2,"P":-1,"S": 4,"T": 1,"W":-3,"Y":-2,"V":-2,
          "B": 0,"Z": 0,"X": 0,"*":-4},
    "T": {"A": 0,"R":-1,"N": 0,"D":-1,"C":-1,"Q":-1,"E":-1,"G":-2,"H":-2,"I":-1,
          "L":-1,"K":-1,"M":-1,"F":-2,"P":-1,"S": 1,"T": 5,"W":-2,"Y":-2,"V": 0,
          "B":-1,"Z":-1,"X": 0,"*":-4},
    "W": {"A":-3,"R":-3,"N":-4,"D":-4,"C":-2,"Q":-2,"E":-3,"G":-2,"H":-2,"I":-3,
          "L":-2,"K":-3,"M":-1,"F": 1,"P":-4,"S":-3,"T":-2,"W":11,"Y": 2,"V":-3,
          "B":-4,"Z":-3,"X":-2,"*":-4},
    "Y": {"A":-2,"R":-2,"N":-2,"D":-3,"C":-2,"Q":-1,"E":-2,"G":-3,"H": 2,"I":-1,
          "L":-1,"K":-2,"M":-1,"F": 3,"P":-3,"S":-2,"T":-2,"W": 2,"Y": 7,"V":-1,
          "B":-3,"Z":-2,"X":-1,"*":-4},
    "V": {"A": 0,"R":-3,"N":-3,"D":-3,"C":-1,"Q":-2,"E":-2,"G":-3,"H":-3,"I": 3,
          "L": 1,"K":-2,"M": 1,"F":-1,"P":-2,"S":-2,"T": 0,"W":-3,"Y":-1,"V": 4,
          "B":-3,"Z":-2,"X":-1,"*":-4},
    "B": {"A":-2,"R":-1,"N": 3,"D": 4,"C":-3,"Q": 0,"E": 1,"G":-1,"H": 0,"I":-3,
          "L":-4,"K": 0,"M":-3,"F":-3,"P":-2,"S": 0,"T":-1,"W":-4,"Y":-3,"V":-3,
          "B": 4,"Z": 1,"X":-1,"*":-4},
    "Z": {"A":-1,"R": 0,"N": 0,"D": 1,"C":-3,"Q": 3,"E": 4,"G":-2,"H": 0,"I":-3,
          "L":-3,"K": 1,"M":-1,"F":-3,"P":-1,"S": 0,"T":-1,"W":-3,"Y":-2,"V":-2,
          "B": 1,"Z": 4,"X":-1,"*":-4},
    "X": {"A": 0,"R":-1,"N":-1,"D":-1,"C":-2,"Q":-1,"E":-1,"G":-1,"H":-1,"I":-1,
          "L":-1,"K":-1,"M":-1,"F":-1,"P":-2,"S": 0,"T": 0,"W":-2,"Y":-1,"V":-1,
          "B":-1,"Z":-1,"X":-1,"*":-4},
    "*": {"A":-4,"R":-4,"N":-4,"D":-4,"C":-4,"Q":-4,"E":-4,"G":-4,"H":-4,"I":-4,
          "L":-4,"K":-4,"M":-4,"F":-4,"P":-4,"S":-4,"T":-4,"W":-4,"Y":-4,"V":-4,
          "B":-4,"Z":-4,"X":-4,"*": 1},
}

_BLOSUM62: dict[tuple[str, str], int] = {}
for _a, _row in _BLOSUM62_ROWS.items():
    for _b, _s in _row.items():
        _BLOSUM62[(_a, _b)] = _s
        _BLOSUM62[(_b, _a)] = _s

# ---------------------------------------------------------------------------
# Simple nucleotide match/mismatch matrix
# ---------------------------------------------------------------------------

_NUC_SIMPLE: dict[tuple[str, str], int] = {}
for _a in "ACGTUNacgtun":
    for _b in "ACGTUNacgtun":
        _a_u = _a.upper().replace("U", "T")
        _b_u = _b.upper().replace("U", "T")
        if _a_u == "N" or _b_u == "N":
            _NUC_SIMPLE[(_a, _b)] = 0
        else:
            _NUC_SIMPLE[(_a, _b)] = 2 if _a_u == _b_u else -3

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_MATRICES: dict[str, dict] = {
    "blosum62": _BLOSUM62,
    "nuc": _NUC_SIMPLE,
}

# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def _dict_to_array(
    d: dict,
    gap_chars: tuple[str, ...],
) -> tuple[list[str], np.ndarray]:
    """Convert a flat (a, b)->score dict to (alphabet, S array)."""
    chars = sorted({a for (a, _) in d if a not in gap_chars})
    A = len(chars)
    idx = {c: i for i, c in enumerate(chars)}
    S = np.zeros((A, A), dtype=np.float32)
    for (a, b), v in d.items():
        if a in idx and b in idx:
            S[idx[a], idx[b]] = float(v)
    return chars, S


def as_array(
    matrix,
    gap_chars: tuple[str, ...] = ("-", "."),
) -> tuple[list[str], np.ndarray]:
    """Convert any supported substitution-matrix format to (alphabet, S).

    Parameters
    ----------
    matrix : str | dict | tuple[list[str], np.ndarray]
        Substitution matrix in one of three forms:

        - ``str``  — name in the built-in registry, e.g. ``"blosum62"``.
        - ``dict`` — flat ``(char_a, char_b) -> score`` mapping.  Missing
          reverse pairs are filled from the forward pair; the resulting
          matrix must be symmetric.
        - ``tuple`` — ``(alphabet, S)`` where *alphabet* is an ordered list
          of single-character strings and *S* is a square numeric array of
          matching size.

    gap_chars : characters to exclude from the returned alphabet.

    Returns
    -------
    alphabet : list[str]
        Sorted list of non-gap symbols present in the matrix.
    S : np.ndarray, shape (A, A), float32
        Symmetric substitution score array indexed by *alphabet*.

    Raises
    ------
    ValueError
        If *matrix* is an unknown name, the shape is mismatched, or the
        resulting array is not symmetric.
    TypeError
        If *matrix* is not one of the expected types.
    """
    gap_set = set(gap_chars)

    if isinstance(matrix, str):
        name = matrix.lower()
        if name not in _MATRICES:
            raise ValueError(
                f"Unknown substitution matrix {matrix!r}. "
                f"Built-ins: {sorted(_MATRICES)}."
            )
        chars, S = _dict_to_array(_MATRICES[name], gap_chars)
        if not np.allclose(S, S.T):
            raise ValueError(
                f"Built-in matrix {matrix!r} is not symmetric — this is a bug."
            )
        return chars, S

    if isinstance(matrix, dict):
        # Fill missing reverse pairs from forward pair so callers can supply
        # either a half-matrix or a full one.
        filled: dict[tuple[str, str], float] = {}
        for (a, b), v in matrix.items():
            filled[(a, b)] = float(v)
            if (b, a) not in matrix:
                filled[(b, a)] = float(v)
        chars, S = _dict_to_array(filled, gap_chars)
        if not np.allclose(S, S.T):
            raise ValueError("Substitution matrix dict is not symmetric.")
        return chars, S

    if isinstance(matrix, tuple):
        alphabet, arr = matrix
        alphabet = [c for c in alphabet if c not in gap_set]
        S = np.asarray(arr, dtype=np.float32)
        if S.ndim != 2 or S.shape[0] != S.shape[1]:
            raise ValueError(
                f"Matrix array must be square; got shape {S.shape}."
            )
        if S.shape[0] != len(alphabet):
            raise ValueError(
                f"Matrix shape {S.shape} does not match alphabet length "
                f"{len(alphabet)}."
            )
        if not np.allclose(S, S.T):
            raise ValueError("Substitution matrix array is not symmetric.")
        return list(alphabet), S

    raise TypeError(
        f"matrix must be a str, dict, or (alphabet, array) tuple; "
        f"got {type(matrix).__name__}."
    )
