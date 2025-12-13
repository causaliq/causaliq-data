"""Tests for uncovered lines in pandas.py module."""

import numpy as np
import pytest
from pandas import DataFrame

from causaliq_data.pandas import Pandas


class TestPandasUncoveredLines:
    """Test cases for currently uncovered lines in Pandas class."""

    def test_pandas_init_with_missing_values(self):
        """Test line 33: Pandas() initialization with missing values."""
        # Create DataFrame with NaN values
        df_with_nan = DataFrame({"A": [1, np.nan, 3], "B": [1, 2, 3]})

        with pytest.raises(
            ValueError, match="Pandas\\(\\) missing data unsupported"
        ):
            Pandas(df_with_nan)

    def test_pandas_init_with_small_dataframe(self):
        """Test line 32: Pandas() initialization with too few rows
        or columns."""
        # Test with only 1 row
        df_one_row = DataFrame({"A": [1], "B": [2]})
        with pytest.raises(
            ValueError, match="Pandas\\(\\) bad dataframe size"
        ):
            Pandas(df_one_row)

        # Test with only 1 column
        df_one_col = DataFrame({"A": [1, 2, 3]})
        with pytest.raises(
            ValueError, match="Pandas\\(\\) bad dataframe size"
        ):
            Pandas(df_one_col)

    def test_nodes_setter(self):
        """Test line 178: nodes setter property."""
        df = DataFrame({"A": ["0", "1"], "B": ["1", "0"]}, dtype="category")
        pandas_obj = Pandas(df)

        # Test setting nodes
        new_nodes = ("X", "Y")
        pandas_obj.nodes = new_nodes
        assert pandas_obj.nodes == new_nodes

    def test_node_types_setter(self):
        """Test line 218: node_types setter property."""
        df = DataFrame({"A": ["0", "1"], "B": ["1", "0"]}, dtype="category")
        pandas_obj = Pandas(df)

        # Test setting node types
        new_types = {"A": "int32", "B": "category"}
        pandas_obj.node_types = new_types
        assert pandas_obj.node_types == new_types

    def test_update_sample_with_name_mapping(self):
        """Test lines 316-325: _update_sample with name mapping changes."""
        df = DataFrame({"A": ["0", "1"], "B": ["1", "0"]}, dtype="category")
        pandas_obj = Pandas(df)

        # Set up initial state with different names
        old_ext_to_orig = {"X": "A", "Y": "B"}
        pandas_obj.ext_to_orig = old_ext_to_orig
        pandas_obj.orig_to_ext = {"A": "X", "B": "Y"}

        # Set up some node_values and node_types with old names
        pandas_obj._node_values = {
            "X": {"0": 1, "1": 1},
            "Y": {"0": 1, "1": 1},
        }
        pandas_obj._node_types = {"X": "category", "Y": "category"}

        # Now change the mapping
        pandas_obj.ext_to_orig = {"P": "A", "Q": "B"}
        pandas_obj.orig_to_ext = {"A": "P", "B": "Q"}

        # Call _update_sample with old mapping to trigger the renaming logic
        pandas_obj._update_sample(old_ext_to_orig=old_ext_to_orig)

        # Verify that node_values and node_types were updated with new names
        assert "P" in pandas_obj.node_values
        assert "Q" in pandas_obj.node_values
        assert "P" in pandas_obj.node_types
        assert "Q" in pandas_obj.node_types

    def test_randomise_names_invalid_seed_type(self):
        """Test line 387: randomise_names with invalid seed type."""
        df = DataFrame({"A": ["0", "1"], "B": ["1", "0"]}, dtype="category")
        pandas_obj = Pandas(df)

        with pytest.raises(
            TypeError, match="Data.randomise_names\\(\\) bad arg type"
        ):
            pandas_obj.randomise_names("invalid_seed")

    def test_randomise_names_with_seed(self):
        """Test lines 388-402: randomise_names functionality with seed."""
        df = DataFrame({"A": ["0", "1"], "B": ["1", "0"]}, dtype="category")
        pandas_obj = Pandas(df)

        # Store original state
        original_nodes = pandas_obj.nodes
        original_ext_to_orig = pandas_obj.ext_to_orig.copy()

        # Test with seed to generate random names
        pandas_obj.randomise_names(seed=42)

        # Check that external names changed but originals preserved
        assert pandas_obj.nodes == original_nodes  # Original nodes unchanged
        assert (
            pandas_obj.ext_to_orig != original_ext_to_orig
        )  # External names changed

        # Check that external names are randomized format
        for ext_name in pandas_obj.ext_to_orig.keys():
            assert ext_name.startswith("X")  # Should start with X
            assert len(ext_name) > 1  # Should be longer than just 'X'

    def test_randomise_names_revert_to_original(self):
        """Test lines 391-395: randomise_names reverting to original names."""
        df = DataFrame({"A": ["0", "1"], "B": ["1", "0"]}, dtype="category")
        pandas_obj = Pandas(df)

        # First randomize names
        pandas_obj.randomise_names(seed=42)
        assert pandas_obj.ext_to_orig != {n: n for n in pandas_obj.nodes}

        # Then revert by passing None
        pandas_obj.randomise_names(seed=None)

        # Check that names are reverted to original
        expected_mapping = {n: n for n in pandas_obj.nodes}
        assert pandas_obj.ext_to_orig == expected_mapping
        assert pandas_obj.orig_to_ext == expected_mapping

    def test_randomise_names_dataframe_column_renaming(self):
        """Test line 392: DataFrame column renaming in randomise_names."""
        df = DataFrame({"A": ["0", "1"], "B": ["1", "0"]}, dtype="category")
        pandas_obj = Pandas(df)

        # First set up a randomized state by calling randomise_names
        pandas_obj.randomise_names(seed=42)

        # Store the current randomized state
        pandas_obj.ext_to_orig.copy()

        # Call randomise_names again with a different seed
        pandas_obj.randomise_names(seed=123)

        # Verify the DataFrame columns match the original node names
        # (the method should revert to originals then re-randomize)
        assert set(pandas_obj.nodes).issubset(set(pandas_obj.df.columns))
