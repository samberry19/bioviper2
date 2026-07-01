"""Tests for bv.one_hot_encode_msa() and the STANDARD_AMINO_ACIDS constant."""
import numpy as np
import pandas as pd
import pytest

import bioviper2 as bv
from bioviper2 import one_hot_encode_msa, STANDARD_AMINO_ACIDS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_msa(seqs: list[str], ids=None) -> bv.MSA:
    n = len(seqs)
    L = len(seqs[0])
    arr = np.empty((n, L), dtype="U1")
    for i, s in enumerate(seqs):
        arr[i] = list(s)
    if ids is None:
        ids = [f"seq{i}" for i in range(n)]
    index = pd.Index(ids, name="id")
    return bv.MSA(arr, index=index)


# ---------------------------------------------------------------------------
# STANDARD_AMINO_ACIDS constant
# ---------------------------------------------------------------------------

class TestConstant:
    def test_length_20(self):
        assert len(STANDARD_AMINO_ACIDS) == 20

    def test_no_duplicates(self):
        assert len(set(STANDARD_AMINO_ACIDS)) == 20

    def test_no_gap_chars(self):
        assert "-" not in STANDARD_AMINO_ACIDS
        assert "." not in STANDARD_AMINO_ACIDS

    def test_exported_at_top_level(self):
        assert hasattr(bv, "STANDARD_AMINO_ACIDS")

    def test_one_hot_exported_at_top_level(self):
        assert hasattr(bv, "one_hot_encode_msa")


# ---------------------------------------------------------------------------
# Default behaviour: standard 20-AA alphabet, no gaps, DataFrame output
# ---------------------------------------------------------------------------

class TestStandardAlphabet:
    def test_df_shape(self):
        msa = _make_msa(["ACDE", "FGHI"])
        df = one_hot_encode_msa(msa)
        n, L = msa.shape
        assert df.shape == (n, L * 20)

    def test_df_index_is_sequence_ids(self):
        msa = _make_msa(["ACDE", "FGHI"], ids=["alpha", "beta"])
        df = one_hot_encode_msa(msa)
        assert list(df.index) == ["alpha", "beta"]

    def test_df_columns_multiindex(self):
        msa = _make_msa(["ACDE"])
        df = one_hot_encode_msa(msa)
        assert isinstance(df.columns, pd.MultiIndex)
        assert df.columns.names == ["position", "residue"]

    def test_df_column_residues_are_standard_aa(self):
        msa = _make_msa(["ACDE"])
        df = one_hot_encode_msa(msa)
        residues_in_cols = df.columns.get_level_values("residue").unique().tolist()
        assert sorted(residues_in_cols) == sorted(STANDARD_AMINO_ACIDS)

    def test_df_column_positions_from_column_index(self):
        msa = _make_msa(["ACDE"])
        df = one_hot_encode_msa(msa)
        positions = df.columns.get_level_values("position").unique().tolist()
        assert positions == list(range(4))

    def test_in_alphabet_residue_has_one_hot_vector(self):
        msa = _make_msa(["ACDE"])
        df = one_hot_encode_msa(msa)
        # Position 0 of seq0 is 'A' — its row should sum to 1 at (0, 'A')
        assert df.loc["seq0", (0, "A")] == pytest.approx(1.0)
        # All other residues at position 0 should be 0
        row0_pos0 = df.loc["seq0"].xs(0, level="position")
        assert row0_pos0.sum() == pytest.approx(1.0)
        assert row0_pos0["A"] == pytest.approx(1.0)

    def test_out_of_alphabet_residue_is_all_zero(self):
        # 'X' (ambiguity code) is not in STANDARD_AMINO_ACIDS
        msa = _make_msa(["XACD"])
        df = one_hot_encode_msa(msa)
        row0_pos0 = df.loc["seq0"].xs(0, level="position")
        assert row0_pos0.sum() == pytest.approx(0.0)

    def test_gap_is_all_zero_by_default(self):
        msa = _make_msa(["A-CD"])
        df = one_hot_encode_msa(msa)
        row0_pos1 = df.loc["seq0"].xs(1, level="position")
        assert row0_pos1.sum() == pytest.approx(0.0)

    def test_dtype_float32(self):
        msa = _make_msa(["ACDE"])
        df = one_hot_encode_msa(msa)
        assert df.values.dtype == np.float32

    def test_dtype_override(self):
        msa = _make_msa(["ACDE"])
        df = one_hot_encode_msa(msa, dtype=np.float64)
        assert df.values.dtype == np.float64


# ---------------------------------------------------------------------------
# The motivating use-case: cross-MSA comparability
# ---------------------------------------------------------------------------

class TestComparability:
    def test_same_columns_different_residues(self):
        """Two MSAs with no residues in common still produce identical columns."""
        msa1 = _make_msa(["AAAA"])
        msa2 = _make_msa(["WWWW"])
        df1 = one_hot_encode_msa(msa1, alphabet="standard")
        df2 = one_hot_encode_msa(msa2, alphabet="standard")
        # Column MultiIndex must be identical
        assert list(df1.columns) == list(df2.columns)

    def test_present_alphabet_differs_between_msas(self):
        msa1 = _make_msa(["AAAA"])
        msa2 = _make_msa(["WWWW"])
        df1 = one_hot_encode_msa(msa1, alphabet="present")
        df2 = one_hot_encode_msa(msa2, alphabet="present")
        # 'present' gives different columns — that's the problem being solved
        r1 = df1.columns.get_level_values("residue").unique().tolist()
        r2 = df2.columns.get_level_values("residue").unique().tolist()
        assert r1 != r2


# ---------------------------------------------------------------------------
# include_gap option
# ---------------------------------------------------------------------------

class TestIncludeGap:
    def test_alphabet_size_is_21(self):
        msa = _make_msa(["ACDE"])
        df = one_hot_encode_msa(msa, include_gap=True)
        assert df.shape == (1, 4 * 21)

    def test_gap_column_present_in_multiindex(self):
        msa = _make_msa(["ACDE"])
        df = one_hot_encode_msa(msa, include_gap=True)
        residues = df.columns.get_level_values("residue").unique().tolist()
        assert "-" in residues

    def test_gap_position_lights_up_gap_column(self):
        msa = _make_msa(["A-CD"])
        df = one_hot_encode_msa(msa, include_gap=True)
        # Position 1 is '-': only the gap column should be 1
        row0_pos1 = df.loc["seq0"].xs(1, level="position")
        assert row0_pos1["-"] == pytest.approx(1.0)
        assert row0_pos1.drop("-").sum() == pytest.approx(0.0)

    def test_dot_gap_also_lights_up_gap_column(self):
        msa = _make_msa(["A.CD"])
        df = one_hot_encode_msa(msa, include_gap=True)
        row0_pos1 = df.loc["seq0"].xs(1, level="position")
        assert row0_pos1["-"] == pytest.approx(1.0)

    def test_non_gap_position_does_not_light_gap_column(self):
        msa = _make_msa(["ACDE"])
        df = one_hot_encode_msa(msa, include_gap=True)
        # All positions are non-gap: gap column must be all zero
        gap_vals = df.xs("-", level="residue", axis=1)
        np.testing.assert_allclose(gap_vals.values, 0.0)

    def test_as_array_alphabet_ends_with_gap(self):
        msa = _make_msa(["ACDE"])
        arr, alph = one_hot_encode_msa(msa, include_gap=True, as_array=True)
        assert alph[-1] == "-"
        assert arr.shape == (1, 4, 21)


# ---------------------------------------------------------------------------
# alphabet="present"
# ---------------------------------------------------------------------------

class TestPresentAlphabet:
    def test_only_present_residues_in_columns(self):
        msa = _make_msa(["AACDE"])
        df = one_hot_encode_msa(msa, alphabet="present")
        residues = df.columns.get_level_values("residue").unique().tolist()
        assert sorted(residues) == ["A", "C", "D", "E"]

    def test_gaps_excluded_from_present(self):
        msa = _make_msa(["A-CDE"])
        df = one_hot_encode_msa(msa, alphabet="present")
        residues = df.columns.get_level_values("residue").unique().tolist()
        assert "-" not in residues

    def test_matches_get_dummies_column_set(self):
        msa = _make_msa(["ACDE", "ACDF"])
        df = one_hot_encode_msa(msa, alphabet="present")
        present_residues = sorted(df.columns.get_level_values("residue").unique())
        expected = sorted(set("ACDE") | set("ACDF"))
        assert present_residues == expected


# ---------------------------------------------------------------------------
# Custom alphabet
# ---------------------------------------------------------------------------

class TestCustomAlphabet:
    def test_nucleotide_alphabet(self):
        msa = _make_msa(["ACGT", "TTAC"])
        df = one_hot_encode_msa(msa, alphabet=["A", "C", "G", "T"])
        assert df.shape == (2, 4 * 4)
        residues = df.columns.get_level_values("residue").unique().tolist()
        assert sorted(residues) == ["A", "C", "G", "T"]

    def test_custom_residue_has_correct_encoding(self):
        msa = _make_msa(["ACGT"])
        df = one_hot_encode_msa(msa, alphabet=["A", "C", "G", "T"])
        # Position 2 is 'G'
        row0_pos2 = df.loc["seq0"].xs(2, level="position")
        assert row0_pos2["G"] == pytest.approx(1.0)
        assert row0_pos2.drop("G").sum() == pytest.approx(0.0)

    def test_duplicate_in_alphabet_raises(self):
        msa = _make_msa(["ACDE"])
        with pytest.raises(ValueError, match="duplicate"):
            one_hot_encode_msa(msa, alphabet=["A", "C", "A"])

    def test_multi_char_entry_raises(self):
        msa = _make_msa(["ACDE"])
        with pytest.raises(ValueError, match="single character"):
            one_hot_encode_msa(msa, alphabet=["A", "CD"])

    def test_invalid_alphabet_string_raises(self):
        msa = _make_msa(["ACDE"])
        with pytest.raises(ValueError, match="'standard' or 'present'"):
            one_hot_encode_msa(msa, alphabet="blosum62")


# ---------------------------------------------------------------------------
# as_array output
# ---------------------------------------------------------------------------

class TestAsArray:
    def test_returns_tuple(self):
        msa = _make_msa(["ACDE"])
        result = one_hot_encode_msa(msa, as_array=True)
        assert isinstance(result, tuple) and len(result) == 2

    def test_array_shape_standard(self):
        msa = _make_msa(["ACDE", "FGHI"])
        arr, alph = one_hot_encode_msa(msa, as_array=True)
        assert arr.shape == (2, 4, 20)
        assert len(alph) == 20

    def test_alphabet_list_matches_standard(self):
        msa = _make_msa(["ACDE"])
        _, alph = one_hot_encode_msa(msa, as_array=True)
        assert alph == list(STANDARD_AMINO_ACIDS)

    def test_reshape_equivalence_with_dataframe(self):
        """df.to_numpy().reshape(n, L, A) must equal the as_array output."""
        msa = _make_msa(["ACDE", "FGHI", "KLMN"])
        df = one_hot_encode_msa(msa)
        arr, _ = one_hot_encode_msa(msa, as_array=True)
        n, L = msa.shape
        A = 20
        np.testing.assert_array_equal(df.to_numpy().reshape(n, L, A), arr)

    def test_reshape_with_gap(self):
        msa = _make_msa(["A-CD"])
        df = one_hot_encode_msa(msa, include_gap=True)
        arr, alph = one_hot_encode_msa(msa, include_gap=True, as_array=True)
        A = 21
        np.testing.assert_array_equal(
            df.to_numpy().reshape(1, 4, A), arr
        )


# ---------------------------------------------------------------------------
# Argmax round-trip
# ---------------------------------------------------------------------------

class TestArgmaxRoundTrip:
    def test_argmax_recovers_residue(self):
        """np.argmax over the alphabet axis recovers the encoded residue."""
        seqs = ["ACDEF", "GHIKL"]
        msa = _make_msa(seqs)
        arr, alph = one_hot_encode_msa(msa, as_array=True)
        # arr shape: (2, 5, 20)
        for i, seq in enumerate(seqs):
            for l, res in enumerate(seq):
                if res in alph:
                    predicted = alph[np.argmax(arr[i, l])]
                    assert predicted == res, (
                        f"seq{i} pos{l}: expected {res!r}, got {predicted!r}"
                    )

    def test_gap_argmax_points_to_gap_column(self):
        msa = _make_msa(["A-CD"])
        arr, alph = one_hot_encode_msa(msa, include_gap=True, as_array=True)
        # Position 1 is '-'
        assert alph[np.argmax(arr[0, 1])] == "-"


# ---------------------------------------------------------------------------
# named column_index is propagated to positions
# ---------------------------------------------------------------------------

class TestNamedColumnIndex:
    def test_named_positions_appear_in_multiindex(self):
        msa = _make_msa(["ACDE"])
        msa.column_index = ["H1", "H2", "H3", "H4"]
        df = one_hot_encode_msa(msa)
        positions = df.columns.get_level_values("position").unique().tolist()
        assert positions == ["H1", "H2", "H3", "H4"]


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrors:
    def test_non_msa_raises_type_error(self):
        with pytest.raises(TypeError, match="MSA instance"):
            one_hot_encode_msa("ACDEFGHIKL")

    def test_dataframe_raises_type_error(self):
        with pytest.raises(TypeError, match="MSA instance"):
            one_hot_encode_msa(pd.DataFrame({"a": [1]}))
