# Test internal functions in indep module for missing code coverage

import pytest

from causaliq_data.indep import _statistic


class TestStatisticFunction:
    """Tests for the _statistic internal function."""

    def test_statistic_bad_arg_types_non_list_actuals(self):
        """Test _statistic with non-list actuals argument."""
        # Line 182: raise TypeError("_indep_tests bad arg types")
        with pytest.raises(TypeError, match="_indep_tests bad arg types"):
            _statistic("not_a_list", "x2")

    def test_statistic_bad_arg_types_non_string_type(self):
        """Test _statistic with non-string type argument."""
        # Line 182: raise TypeError("_indep_tests bad arg types")
        with pytest.raises(TypeError, match="_indep_tests bad arg types"):
            _statistic([[1, 2], [3, 4]], 123)  # type should be string

    def test_statistic_bad_arg_types_inner_list_not_list(self):
        """Test _statistic with inner elements that are not lists."""
        # Line 182: raise TypeError("_indep_tests bad arg types")
        with pytest.raises(TypeError, match="_indep_tests bad arg types"):
            _statistic([1, 2, 3], "x2")  # inner elements should be lists

    def test_statistic_misshapen_actuals_different_lengths(self):
        """Test _statistic with misshapen actuals (different row lengths)."""
        # Line 187: raise TypeError("_indep_tests misshapen actuals arg")
        with pytest.raises(
            TypeError, match="_indep_tests misshapen actuals arg"
        ):
            _statistic([[1, 2, 3], [4, 5]], "x2")  # different row lengths

    def test_statistic_empty_table_returns_zero(self):
        """Test _statistic with empty contingency table (all zeros)."""
        # Line 193: return (df, 0.0) when N == 0
        actuals = [[0, 0], [0, 0]]
        df, statistic = _statistic(actuals, "x2")

        # For a 2x2 table, df = (2-1) * (2-1) = 1
        assert df == 1
        # When all counts are zero, statistic should be 0.0
        assert statistic == 0.0

    def test_statistic_empty_table_returns_zero_mi(self):
        """Test _statistic with empty contingency table for MI test."""
        # Line 193: return (df, 0.0) when N == 0
        actuals = [[0, 0, 0], [0, 0, 0]]
        df, statistic = _statistic(actuals, "mi")

        # For a 2x3 table, df = (2-1) * (3-1) = 2
        assert df == 2
        # When all counts are zero, statistic should be 0.0
        assert statistic == 0.0

    def test_statistic_larger_empty_table(self):
        """Test _statistic with larger empty contingency table."""
        # Line 193: return (df, 0.0) when N == 0
        actuals = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        df, statistic = _statistic(actuals, "x2")

        # For a 3x3 table, df = (3-1) * (3-1) = 4
        assert df == 4
        # When all counts are zero, statistic should be 0.0
        assert statistic == 0.0

    def test_statistic_valid_input_sanity_check(self):
        """Sanity check that valid input doesn't raise exceptions."""
        # This should not raise any exceptions and should return valid results
        actuals = [[10, 5], [3, 8]]
        df, statistic = _statistic(actuals, "x2")

        assert isinstance(df, int)
        assert isinstance(statistic, float)
        assert df == 1  # (2-1) * (2-1) = 1
        assert statistic >= 0.0  # Chi-squared statistic should be non-negative
