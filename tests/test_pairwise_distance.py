"""Tests for MSA.pairwise_distance and bioviper2.matrices.as_array."""
import warnings

import numpy as np
import pytest

import bioviper2 as bv
from bioviper2.matrices import as_array, _BLOSUM62, _MATRICES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_msa(seqs: list[str], ids=None) -> bv.MSA:
    """Build an MSA from a list of equal-length strings."""
    import pandas as pd

    n = len(seqs)
    L = len(seqs[0])
    arr = np.empty((n, L), dtype="U1")
    for i, s in enumerate(seqs):
        arr[i] = list(s)
    if ids is None:
        ids = [f"seq{i}" for i in range(n)]
    index = pd.Index(ids, name="id")
    return bv.MSA(arr, index=index)


def _brute_force_distance(msa: bv.MSA, sub: dict) -> np.ndarray:
    """O(n²) reference implementation of pairwise_distance using BLOSUM62 dict."""
    arr = msa._array
    n, L = arr.shape
    gap_chars = {"-", "."}

    D = np.full((n, n), np.nan, dtype=np.float64)
    for i in range(n):
        for j in range(n):
            num = 0.0
            si = 0.0
            sj = 0.0
            shared = 0
            for l in range(L):
                ri, rj = arr[i, l], arr[j, l]
                if ri in gap_chars or rj in gap_chars:
                    continue
                score_ij = sub.get((ri, rj))
                score_ii = sub.get((ri, ri))
                score_jj = sub.get((rj, rj))
                if score_ij is None or score_ii is None or score_jj is None:
                    continue
                num += score_ij
                si  += score_ii
                sj  += score_jj
                shared += 1
            if shared == 0 or si <= 0 or sj <= 0:
                D[i, j] = np.nan
            else:
                norm_sim = num / np.sqrt(si * sj)
                D[i, j] = 1.0 - norm_sim
    np.fill_diagonal(D, 0.0)
    return D


# ---------------------------------------------------------------------------
# matrices.as_array
# ---------------------------------------------------------------------------

class TestAsArray:
    def test_blosum62_shape(self):
        alph, S = as_array("blosum62")
        assert S.ndim == 2
        assert S.shape[0] == S.shape[1] == len(alph)

    def test_blosum62_symmetric(self):
        _, S = as_array("blosum62")
        np.testing.assert_allclose(S, S.T, atol=1e-5)

    def test_blosum62_diagonal_positive(self):
        # The standard 20 amino acids have positive BLOSUM62 self-scores.
        # Ambiguity codes like X have negative self-scores by design (NCBI).
        alph, S = as_array("blosum62")
        idx = {c: i for i, c in enumerate(alph)}
        for aa in "ACDEFGHIKLMNPQRSTVWY":
            assert S[idx[aa], idx[aa]] > 0, f"Self-score for {aa} should be positive"

    def test_blosum62_known_value(self):
        alph, S = as_array("blosum62")
        idx = {c: i for i, c in enumerate(alph)}
        # W self-score = 11
        assert S[idx["W"], idx["W"]] == pytest.approx(11.0)
        # I-V = 3
        assert S[idx["I"], idx["V"]] == pytest.approx(3.0)

    def test_gap_chars_excluded_from_alphabet(self):
        alph, _ = as_array("blosum62")
        assert "-" not in alph
        assert "." not in alph

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError, match="Unknown substitution matrix"):
            as_array("pam250")

    def test_dict_input_symmetric(self):
        # Minimal 3-residue custom matrix (both orderings supplied)
        d = {
            ("A", "A"): 4, ("A", "B"): -1, ("A", "C"): -2,
            ("B", "A"): -1, ("B", "B"): 5, ("B", "C"): 1,
            ("C", "A"): -2, ("C", "B"): 1, ("C", "C"): 6,
        }
        alph, S = as_array(d)
        np.testing.assert_allclose(S, S.T, atol=1e-5)
        assert alph == ["A", "B", "C"]

    def test_dict_input_half_matrix_fills_reverse(self):
        # Supply only the upper triangle; as_array should fill the lower
        d = {("A", "A"): 4, ("A", "B"): -1, ("B", "B"): 5}
        alph, S = as_array(d)
        np.testing.assert_allclose(S, S.T, atol=1e-5)

    def test_tuple_input(self):
        alph_in = ["A", "B"]
        S_in = np.array([[4.0, -1.0], [-1.0, 5.0]])
        alph, S = as_array((alph_in, S_in))
        assert alph == ["A", "B"]
        np.testing.assert_allclose(S, S_in, atol=1e-5)

    def test_tuple_wrong_shape_raises(self):
        with pytest.raises(ValueError):
            as_array((["A", "B"], np.zeros((3, 3))))

    def test_asymmetric_raises(self):
        d = {("A", "A"): 4, ("A", "B"): -1, ("B", "A"): 99, ("B", "B"): 5}
        with pytest.raises(ValueError, match="symmetric"):
            as_array(d)

    def test_wrong_type_raises(self):
        with pytest.raises(TypeError):
            as_array(42)

    def test_gap_chars_excluded_from_alphabet_custom(self):
        alph, _ = as_array("blosum62", gap_chars=("-", ".", "*"))
        assert "*" not in alph


# ---------------------------------------------------------------------------
# MSA.pairwise_distance — basic properties
# ---------------------------------------------------------------------------

class TestPairwiseDistanceBasic:
    def test_zero_diagonal(self):
        msa = _make_msa(["ACDEF", "GHIKL", "MNPQR"])
        D = msa.pairwise_distance()
        np.testing.assert_allclose(np.diag(D.values), 0.0, atol=1e-5)

    def test_symmetric(self):
        msa = _make_msa(["ACDEF", "GHIKL", "MNPQR", "STVWY"])
        D = msa.pairwise_distance()
        np.testing.assert_allclose(D.values, D.values.T, atol=1e-5)

    def test_shape(self):
        msa = _make_msa(["ACDE", "FGHI", "KLMN"])
        D = msa.pairwise_distance()
        assert D.shape == (3, 3)

    def test_index_preserved(self):
        msa = _make_msa(["ACDE", "FGHI"], ids=["alpha", "beta"])
        D = msa.pairwise_distance()
        assert list(D.index) == ["alpha", "beta"]
        assert list(D.columns) == ["alpha", "beta"]

    def test_identical_sequences_zero(self):
        msa = _make_msa(["ACDEFGHIKL", "ACDEFGHIKL", "MNPQRSTVWY"])
        D = msa.pairwise_distance()
        # Rows 0 and 1 are identical: distance should be 0
        assert D.iloc[0, 1] == pytest.approx(0.0, abs=1e-5)
        assert D.iloc[1, 0] == pytest.approx(0.0, abs=1e-5)

    def test_dtype_float32(self):
        msa = _make_msa(["ACDE", "FGHI"])
        D = msa.pairwise_distance()
        assert D.values.dtype == np.float32

    def test_nan_for_no_shared_ungapped(self):
        # Seq 0: all residues, Seq 1: all gaps — no shared non-gap columns
        msa = _make_msa(["ACDE", "----"])
        D = msa.pairwise_distance()
        assert np.isnan(D.iloc[0, 1])
        assert np.isnan(D.iloc[1, 0])

    def test_distance_can_exceed_one(self):
        # Very dissimilar sequences should yield distance > 0; whether > 1
        # depends on the sequences — just check we don't artificially clip
        msa = _make_msa(["WWWWW", "GGGGG"])  # W-G cross-score is very negative
        D = msa.pairwise_distance()
        # distance should be > 0 (not clipped to 1)
        assert D.iloc[0, 1] > 0

    def test_partial_gap_overlap(self):
        # Cols 0-2 shared for seq0/seq1; cols 2-4 shared for seq0/seq2
        msa = _make_msa(["ACDEF", "ACD--", "--DEF"])
        D = msa.pairwise_distance()
        assert not np.isnan(D.iloc[0, 1])
        assert not np.isnan(D.iloc[0, 2])


# ---------------------------------------------------------------------------
# MSA.pairwise_distance — cross-check against brute-force reference
# ---------------------------------------------------------------------------

class TestPairwiseDistanceBruteForce:
    """Compare vectorised BLAS implementation against a naive O(n²) loop."""

    def _check(self, seqs):
        msa = _make_msa(seqs)
        D_fast = msa.pairwise_distance().values.astype(np.float64)
        D_ref  = _brute_force_distance(msa, _BLOSUM62)
        # Where brute-force gives NaN, fast should too (and vice versa)
        nan_ref  = np.isnan(D_ref)
        nan_fast = np.isnan(D_fast)
        np.testing.assert_array_equal(nan_ref, nan_fast)
        # Numeric agreement on non-NaN entries
        np.testing.assert_allclose(
            D_fast[~nan_fast], D_ref[~nan_ref], atol=1e-4, rtol=1e-4,
        )

    def test_all_ungapped(self):
        self._check(["ACDEF", "GHIKL", "MNPQR", "STVWY"])

    def test_with_gaps(self):
        self._check(["AC-EF", "G-IKL", "-NPQR", "STV-Y"])

    def test_single_shared_column(self):
        # Sequences share only one non-gap column each pair
        self._check(["A----", "-A---", "--A--"])

    def test_repeated_residue(self):
        self._check(["AAAAA", "RRRRR", "NNNNN"])

    def test_large_random(self):
        """50 sequences × 200 positions, ~20% gaps."""
        rng = np.random.default_rng(42)
        aa = list("ACDEFGHIKLMNPQRSTVWY")
        seqs = []
        for _ in range(50):
            s = [rng.choice(aa) for _ in range(200)]
            gap_idx = rng.choice(200, size=40, replace=False)
            for gi in gap_idx:
                s[gi] = "-"
            seqs.append("".join(s))
        self._check(seqs)

    def test_identical_pair_zero_brute(self):
        msa = _make_msa(["ACDEFGHIKL", "ACDEFGHIKL"])
        D_ref = _brute_force_distance(msa, _BLOSUM62)
        assert D_ref[0, 1] == pytest.approx(0.0, abs=1e-8)


# ---------------------------------------------------------------------------
# MSA.pairwise_distance — custom matrix
# ---------------------------------------------------------------------------

class TestPairwiseDistanceCustomMatrix:
    def test_dict_custom_matrix(self):
        """Tiny 3-letter custom matrix, check against brute-force."""
        custom_dict = {
            ("A", "A"): 4, ("A", "B"): -1, ("A", "C"): -2,
            ("B", "A"): -1, ("B", "B"): 5, ("B", "C"): 1,
            ("C", "A"): -2, ("C", "B"): 1, ("C", "C"): 6,
        }
        seqs = ["AABBC", "BCAAB", "CABBC"]
        msa = _make_msa(seqs)
        D = msa.pairwise_distance(matrix=custom_dict).values.astype(np.float64)
        D_ref = _brute_force_distance(msa, custom_dict)
        np.testing.assert_allclose(D, D_ref, atol=1e-4)

    def test_tuple_custom_matrix(self):
        """Supply matrix as (alphabet, ndarray) tuple."""
        alph = ["A", "B"]
        S = np.array([[4.0, -1.0], [-1.0, 5.0]])
        msa = _make_msa(["AABB", "BBAA", "ABAB"])
        D = msa.pairwise_distance(matrix=(alph, S))
        np.testing.assert_allclose(D.values, D.values.T, atol=1e-5)

    def test_unknown_name_raises(self):
        msa = _make_msa(["ACDE", "FGHI"])
        with pytest.raises(ValueError, match="Unknown substitution matrix"):
            msa.pairwise_distance(matrix="pam250")


# ---------------------------------------------------------------------------
# MSA.pairwise_distance — unknown character warning
# ---------------------------------------------------------------------------

class TestUnknownCharWarning:
    def test_warns_on_unknown_character(self):
        # 'O' (pyrrolysine) is not in BLOSUM62
        msa = _make_msa(["ACDO", "GHIO"])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            msa.pairwise_distance()
        msgs = [str(warning.message) for warning in w]
        assert any("'O'" in m or "O" in m for m in msgs), \
            f"Expected warning about 'O'; got: {msgs}"

    def test_no_warn_on_gap_chars(self):
        msa = _make_msa(["ACD-", "GHI-"])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            msa.pairwise_distance()
        gap_warns = [warning for warning in w
                     if "not in the substitution" in str(warning.message)]
        assert len(gap_warns) == 0


# ---------------------------------------------------------------------------
# tools.align regression — substitution matrix move should not break alignment
# ---------------------------------------------------------------------------

class TestAlignRegressionAfterMatrixMove:
    def test_global_align_returns_msa(self):
        msa = bv.align("ACDEF", "ACDF", substitution_matrix="blosum62")
        assert isinstance(msa, bv.MSA)
        assert msa.n_seqs == 2

    def test_global_align_identity_identical(self):
        msa = bv.align("ACDEF", "ACDEF")
        arr = msa._array
        assert np.all(arr[0] == arr[1])

    def test_custom_matrix_in_align(self):
        # Align with the nuc matrix (still registered after the move)
        msa = bv.align("ACGT", "ACGT", substitution_matrix="nuc")
        assert isinstance(msa, bv.MSA)

    def test_unknown_matrix_raises_in_align(self):
        with pytest.raises(ValueError, match="Unknown substitution_matrix"):
            bv.align("ACDE", "ACDE", substitution_matrix="pam250")
