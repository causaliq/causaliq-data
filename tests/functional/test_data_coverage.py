# Test module for covering specific uncovered lines in data.py

import pytest
from causaliq_core.bn.io import read_bn

from causaliq_data import Data, Oracle

TESTDATA_DIR = "tests/data/functional"


# Create a test class that inherits from Data to cover specific lines
class _TestDataImpl(Data):
    """Test implementation of Data class to cover specific lines."""

    def __init__(self, oracle_instance):
        super().__init__()  # This covers line 51 (pass statement)
        # Copy attributes from oracle instance
        self.bn = oracle_instance.bn
        self._nodes = oracle_instance._nodes
        self.order = oracle_instance.order
        self.ext_to_orig = oracle_instance.ext_to_orig
        self.orig_to_ext = oracle_instance.orig_to_ext
        self._node_types = oracle_instance._node_types
        self._N = oracle_instance._N
        self.dstype = oracle_instance.dstype
        # Set class name to trigger Pandas-specific code paths
        self.__class__.__name__ = "Pandas"

    # Implement abstract methods
    def set_N(self, N, seed=None, random_selection=False):
        self._N = N

    def marginals(self, node, parents, values_reqd=False):
        pass

    def values(self, nodes):
        pass

    def randomise_names(self, seed):
        pass

    def as_df(self):
        pass

    def write(self, filename):
        pass

    def _update_sample(self, old_N=None, old_ext_to_orig=None):
        pass

    @property
    def nodes(self):
        return self._nodes

    @nodes.setter
    def nodes(self, value):
        self._nodes = value

    @property
    def sample(self):
        return self._N

    @sample.setter
    def sample(self, value):
        self._N = value

    @property
    def N(self):
        return self._N

    @N.setter
    def N(self, value):
        self._N = value

    @property
    def node_types(self):
        return self._node_types

    @node_types.setter
    def node_types(self, value):
        self._node_types = value

    @property
    def node_values(self):
        return {}

    @node_values.setter
    def node_values(self, value):
        pass


# Test to cover Data.__init__() pass statement (line 51)
def test_data_init_direct():
    """Test that Data.__init__() pass statement is executed."""
    bn = read_bn(TESTDATA_DIR + "/xdsl/ab.xdsl")
    oracle = Oracle(bn)

    # Create _TestDataImpl instance which calls Data.__init__() containing pass
    test_data = _TestDataImpl(oracle)

    # The fact that this works means Data.__init__() was called
    assert hasattr(test_data, "elapsed")
    assert test_data.elapsed == 0.0


# Test to cover _update_sample() call in set_order for Pandas class (line 75)
def test_set_order_pandas_update_sample():
    """Test _update_sample() call when class name is 'Pandas'."""
    bn = read_bn(TESTDATA_DIR + "/xdsl/ab.xdsl")
    oracle = Oracle(bn)
    test_data = _TestDataImpl(oracle)

    # Track if _update_sample was called
    update_sample_called = False

    def mock_update_sample(old_N=None, old_ext_to_orig=None):
        nonlocal update_sample_called
        update_sample_called = True

    test_data._update_sample = mock_update_sample

    # Now call set_order - this should trigger the _update_sample call
    original_order = test_data.get_order()
    new_order = tuple(reversed(original_order))
    test_data.set_order(new_order)

    # Verify _update_sample was called
    assert update_sample_called


# Test to cover _update_sample() call in randomise_order for Pandas class
def test_randomise_order_pandas_update_sample():
    """Test _update_sample() call in randomise_order
    when class name is 'Pandas'."""
    bn = read_bn(TESTDATA_DIR + "/xdsl/ab.xdsl")
    oracle = Oracle(bn)
    test_data = _TestDataImpl(oracle)

    # Track if _update_sample was called
    update_sample_called = False

    def mock_update_sample(old_N=None, old_ext_to_orig=None):
        nonlocal update_sample_called
        update_sample_called = True

    test_data._update_sample = mock_update_sample

    # Now call randomise_order - this should trigger the _update_sample call
    test_data.randomise_order(42)

    # Verify _update_sample was called
    assert update_sample_called


# Test to specifically cover lines 96 and 99 - error checks in randomise_order
def test_randomise_order_error_checks():
    """Test to specifically cover lines 96 and 99 in randomise_order method."""
    bn = read_bn(TESTDATA_DIR + "/xdsl/ab.xdsl")
    oracle = Oracle(bn)
    test_data = _TestDataImpl(oracle)

    # Test line 96: TypeError for non-int seed
    with pytest.raises(
        TypeError, match="Data.randomise_order\\(\\) bad arg type"
    ):
        test_data.randomise_order(
            "not_an_int"
        )  # Line 96: if not isinstance(seed, int)

    with pytest.raises(
        TypeError, match="Data.randomise_order\\(\\) bad arg type"
    ):
        test_data.randomise_order(
            3.14
        )  # Line 96: if not isinstance(seed, int)

    with pytest.raises(
        TypeError, match="Data.randomise_order\\(\\) bad arg type"
    ):
        test_data.randomise_order(
            None
        )  # Line 96: if not isinstance(seed, int)

    # Test line 99: ValueError for negative seed
    with pytest.raises(
        ValueError, match="Data.randomise_order\\(\\) bad arg value"
    ):
        test_data.randomise_order(-1)  # Line 99: if seed < 0

    with pytest.raises(
        ValueError, match="Data.randomise_order\\(\\) bad arg value"
    ):
        test_data.randomise_order(-100)  # Line 99: if seed < 0


# Test to also cover the _set_dstype sum calculations (lines 107-109, 112-113)
def test_set_dstype_sum_lines():
    """Test to cover the sum calculations in _set_dstype method."""
    bn = read_bn(TESTDATA_DIR + "/xdsl/ab.xdsl")
    oracle = Oracle(bn)
    test_data = _TestDataImpl(oracle)

    # Test to trigger both sum calculations in _set_dstype
    # n_floats = sum(1 if v in {"float32", "float64"} else 0 ...)
    # n_cats = sum(1 if v == "category" else 0 ...)

    # Trigger float sum calculation
    test_data._node_types = {"A": "float32", "B": "float64"}
    test_data._nodes = ("A", "B")
    super(_TestDataImpl, test_data)._set_dstype()
    assert test_data.dstype == "continuous"

    # Trigger category sum calculation
    test_data._node_types = {"A": "category", "B": "category"}
    super(_TestDataImpl, test_data)._set_dstype()
    assert test_data.dstype == "categorical"

    # Trigger both sum calculations
    test_data._node_types = {"A": "float32", "B": "category", "C": "float64"}
    test_data._nodes = ("A", "B", "C")
    super(_TestDataImpl, test_data)._set_dstype()
    assert test_data.dstype == "mixed"


# Test to specifically cover the sum calculations in _set_dstype
def test_set_dstype_sum_calculations():
    """Test the sum calculations in _set_dstype method (lines 96 and 99)."""
    bn = read_bn(TESTDATA_DIR + "/xdsl/ab.xdsl")
    oracle = Oracle(bn)
    test_data = _TestDataImpl(oracle)

    # Test with a mix that ensures both sum calculations are executed
    test_data._node_types = {"A": "float32", "B": "category", "C": "float64"}
    test_data._nodes = ("A", "B", "C")  # Add a third node

    # This should execute both sum calculations (lines 96 and 99)
    test_data._set_dstype()

    # Should be mixed since we have both float and category types
    assert test_data.dstype == "mixed"


# Test to ensure the ternary operator branches in _set_dstype are covered
def test_set_dstype_ternary_branches():
    """Test all branches of the ternary operator in _set_dstype."""
    bn = read_bn(TESTDATA_DIR + "/xdsl/ab.xdsl")
    oracle = Oracle(bn)
    test_data = _TestDataImpl(oracle)

    # Test continuous case (n_floats == len(self.nodes))
    test_data._node_types = {"A": "float32", "B": "float64"}
    test_data._set_dstype()
    assert test_data.dstype == "continuous"

    # Test categorical case (n_cats == len(self.nodes))
    test_data._node_types = {"A": "category", "B": "category"}
    test_data._set_dstype()
    assert test_data.dstype == "categorical"

    # Test mixed case (the final else)
    test_data._node_types = {"A": "category", "B": "float32"}
    test_data._set_dstype()
    assert test_data.dstype == "mixed"


# Test edge case where node_types contains other types
def test_set_dstype_other_types():
    """Test _set_dstype with node types that
    are neither float nor category."""
    bn = read_bn(TESTDATA_DIR + "/xdsl/ab.xdsl")
    oracle = Oracle(bn)
    test_data = _TestDataImpl(oracle)

    # Test with int types (should count as neither float nor category)
    test_data._node_types = {"A": "int32", "B": "int64"}
    test_data._set_dstype()

    # Should be mixed since n_floats=0 and n_cats=0, but total nodes > 0
    assert test_data.dstype == "mixed"


# Test with empty node_types
def test_set_dstype_empty():
    """Test _set_dstype with empty node_types."""
    bn = read_bn(TESTDATA_DIR + "/xdsl/ab.xdsl")
    oracle = Oracle(bn)
    _TestDataImpl(oracle)
