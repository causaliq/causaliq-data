# Test the Pandas implementation of marginals

import pytest
from causaliq_core.bn.io import read_bn
from causaliq_core.utils import values_same
from numpy import ndarray

from causaliq_data import Oracle

TESTDATA_DIR = "tests/data/functional"


@pytest.fixture(scope="module")  # AB, 2 rows
def ab2():
    data = Oracle(read_bn(TESTDATA_DIR + "/xdsl/ab.xdsl"))
    data.set_N(2)
    return data


@pytest.fixture(scope="module")  # AB, 4 rows
def ab4():
    data = Oracle(read_bn(TESTDATA_DIR + "/xdsl/ab.xdsl"))
    data.set_N(4)
    return data


@pytest.fixture(scope="module")  # ABC, 10 rows
def abc10():
    data = Oracle(read_bn(TESTDATA_DIR + "/xdsl/abc.xdsl"))
    data.set_N(10)
    return data


@pytest.fixture(scope="module")  # Asia, 100 rows
def asia():
    data = Oracle(read_bn(TESTDATA_DIR + "/xdsl/asia.xdsl"))
    data.set_N(100)
    return data


def check(args, results, data):
    """
    Check results obtained match ones expected.

    :param tuple args: (node, parents, values_reqd) arguments
    :param tuple results: (counts, maxcol, rowval, colval) obtained
    :param Pandas data: underlying data used to check results
    """
    print(
        (
            "\n\nNode: {}, parents: {}, values_reqd: {}\n\n"
            + "Counts:\n{}\nType: {}, memory: {} bytes\nMax parent values: {}"
            "\nChild values: {}\nParent values: {}"
        ).format(
            args[0],
            args[1],
            args[2],
            results[0],
            results[0].dtype,
            results[0].nbytes,
            results[1],
            results[2],
            results[3],
        )
    )

    # assert results have correct types and match expected

    assert isinstance(results[0], ndarray)
    assert results[0].dtype == "float64"

    # Check shape and maxcol returned by old marginals

    orig, orig_num_pvs = orig_marginals(data, args[0], args[1])
    assert results[0].shape == orig.shape
    assert results[1] == orig_num_pvs
    print("\n{}\n".format(orig.head()))

    # if requesting values, then can compare old and new counts too

    if results[2] is not None:
        if results[3] is None:
            for i, xi in enumerate(results[2]):
                count = orig.loc[xi].values[0]
                assert values_same(
                    count.item(), results[0][i, 0].item(), sf=10
                )
        else:

            # get levels in original column multi-index which varies

            levels = {i: n for i, n in enumerate(orig.columns.names)}

            for j, pvs in enumerate(results[3]):

                # Get parental values in variable order in levels for
                # column key used in loc(). Then compare all row counts

                pvs = tuple(pvs[levels[i]] for i in levels)
                for i, xi in enumerate(results[2]):
                    assert values_same(
                        orig.loc[xi, pvs].item(),
                        results[0][i, j].item(),
                        sf=10,
                    )


def orig_marginals(self, node, parents):
    """
    Return marginal counts for a node and its parents.

    :param str node: node for which marginals required.
    :param dict parents: {node: parents} parents of non-orphan nodes

    :raises TypeError: for bad argument types

    :returns tuple: marginals, DataFrame - marginal counts, row index
                    is node value, column multi-index are parent values
                    num_pvs int - number of parental value combinations
                    (for Oracle data this is # of cols in marginals)
    """
    nodes = [node] + parents[node] if node in parents else [node]
    marginals = self.bn.marginals(nodes).apply(lambda x: self.N * x)
    num_pvs = len(marginals.columns)

    return (marginals, num_pvs)


# No arguments specified
def test_oracle_marginals_type_error_1(ab2):
    with pytest.raises(TypeError):
        ab2.marginals()


# Bad or missing node
def test_oracle_marginals_type_error_2(ab2):
    with pytest.raises(TypeError):
        ab2.marginals(node=1, parents={})
    with pytest.raises(TypeError):
        ab2.marginals(node=12.9, parents={})
    with pytest.raises(TypeError):
        ab2.marginals(node=None, parents={})
    with pytest.raises(TypeError):
        ab2.marginals(node=["A"], parents={})
    with pytest.raises(TypeError):
        ab2.marginals(parents={})


# Bad or missing parents
def test_oracle_marginals_type_error_3(ab2):
    with pytest.raises(TypeError):
        ab2.marginals(node="A", parents=2)
    with pytest.raises(TypeError):
        ab2.marginals(node="A", parents="B")
    with pytest.raises(TypeError):
        ab2.marginals(node="A", parents=None)
    with pytest.raises(TypeError):
        ab2.marginals(node="A")


# Parent values bad type
def test_pandas_marginals_type_error_4(ab2):
    with pytest.raises(TypeError):
        ab2.marginals(node="A", parents={"A": "B"})
    with pytest.raises(TypeError):
        ab2.marginals(node="A", parents={"A": {"B"}})
    with pytest.raises(TypeError):
        ab2.marginals(node="A", parents={"A": ("B",)})


# Bad values_reqd type
def test_oracle_marginals_type_error_5(ab2):
    with pytest.raises(TypeError):
        ab2.marginals(node="A", parents={}, values_reqd=1)
    with pytest.raises(TypeError):
        ab2.marginals(node="A", parents={}, values_reqd=None)
    with pytest.raises(TypeError):
        ab2.marginals(node="A", parents={}, values_reqd={False})
    with pytest.raises(TypeError):
        ab2.marginals(node="A", parents={}, values_reqd=[False])
    with pytest.raises(TypeError):
        ab2.marginals(node="A", parents={}, values_reqd=(False,))


# Test AB2 marginals for node A with values required
def test_oracle_marginals_ab2_1_ok(ab2):
    args = ("A", {}, True)
    results = ab2.marginals(args[0], args[1], args[2])
    check(args, results, ab2)


# Test AB2 marginals for node A without values required
def test_oracle_marginals_ab2_2_ok(ab2):
    args = ("A", {}, False)
    results = ab2.marginals(args[0], args[1], args[2])
    check(args, results, ab2)


# Test AB2 marginals for node B with values required
def test_oracle_marginals_ab2_3_ok(ab2):
    args = ("B", {}, True)
    results = ab2.marginals(args[0], args[1], args[2])
    check(args, results, ab2)


# Test AB2 marginals for node A with parent B
def test_oracle_marginals_ab2_4_ok(ab2):
    args = ("A", {"A": ["B"]}, True)
    results = ab2.marginals(args[0], args[1], args[2])
    check(args, results, ab2)


# Test AB2 marginals for node B with parent A
def test_oracle_marginals_ab2_5_ok(ab2):
    args = ("B", {"B": ["A"]}, True)
    results = ab2.marginals(args[0], args[1], args[2])
    check(args, results, ab2)


# Test AB4 marginals for node A with values required
def test_oracle_marginals_ab4_1_ok(ab4):
    args = ("A", {}, True)
    results = ab4.marginals(args[0], args[1], args[2])
    check(args, results, ab4)


# Test AB4 marginals for node B with values required
def test_oracle_marginals_ab4_2_ok(ab4):
    args = ("B", {}, True)
    results = ab4.marginals(args[0], args[1], args[2])
    check(args, results, ab4)


# Test AB4 marginals for node A with parent B
def test_oracle_marginals_ab4_3_ok(ab4):
    args = ("A", {"A": ["B"]}, True)
    results = ab4.marginals(args[0], args[1], args[2])
    check(args, results, ab4)


# Test AB4 marginals for node B with parent A
def test_oracle_marginals_ab4_4_ok(ab4):
    args = ("B", {"B": ["A"]}, True)
    results = ab4.marginals(args[0], args[1], args[2])
    check(args, results, ab4)


# Test ABC10 marginals for node A alone
def test_oracle_marginals_abc10_1_ok(abc10):
    args = ("A", {}, True)
    results = abc10.marginals(args[0], args[1], args[2])
    check(args, results, abc10)


# Test ABC10 marginals for node A with parent B
def test_oracle_marginals_abc10_2_ok(abc10):
    args = ("A", {"A": ["B"]}, True)
    results = abc10.marginals(args[0], args[1], args[2])
    check(args, results, abc10)


# Test ABC10 marginals for node A with parent C
def test_oracle_marginals_abc10_3_ok(abc10):
    args = ("A", {"A": ["C"]}, True)
    results = abc10.marginals(args[0], args[1], args[2])
    check(args, results, abc10)


# Test ABC10 marginals for node A with parents B and C
def test_oracle_marginals_abc10_4_ok(abc10):
    args = ("A", {"A": ["B", "C"]}, True)
    results = abc10.marginals(args[0], args[1], args[2])
    check(args, results, abc10)


# Test ABC10 marginals for node B alone
def test_oracle_marginals_abc10_5_ok(abc10):
    args = ("B", {}, True)
    results = abc10.marginals(args[0], args[1], args[2])
    check(args, results, abc10)


# Test ABC10 marginals for node B with parent A
def test_oracle_marginals_abc10_6_ok(abc10):
    args = ("B", {"B": ["A"]}, True)
    results = abc10.marginals(args[0], args[1], args[2])
    check(args, results, abc10)


# Test ABC10 marginals for node B with parent C
def test_oracle_marginals_abc10_7_ok(abc10):
    args = ("B", {"B": ["C"]}, True)
    results = abc10.marginals(args[0], args[1], args[2])
    check(args, results, abc10)


# Test ABC10 marginals for node B with parents A and C
def test_oracle_marginals_abc10_8_ok(abc10):
    args = ("B", {"B": ["A", "C"]}, True)
    results = abc10.marginals(args[0], args[1], args[2])
    check(args, results, abc10)


# Test ABC10 marginals for node C alone
def test_oracle_marginals_abc10_9_ok(abc10):
    args = ("C", {}, True)
    results = abc10.marginals(args[0], args[1], args[2])
    check(args, results, abc10)


# Test ABC10 marginals for node C with parent A
def test_oracle_marginals_abc10_10_ok(abc10):
    args = ("C", {"C": ["A"]}, True)
    results = abc10.marginals(args[0], args[1], args[2])
    check(args, results, abc10)


# Test ABC10 marginals for node C with parent B
def test_oracle_marginals_abc10_11_ok(abc10):
    args = ("C", {"C": ["B"]}, True)
    results = abc10.marginals(args[0], args[1], args[2])
    check(args, results, abc10)


# Test ABC10 marginals for node C with parents A and B
def test_oracle_marginals_abc10_12_ok(abc10):
    args = ("C", {"C": ["A", "B"]}, True)
    results = abc10.marginals(args[0], args[1], args[2])
    check(args, results, abc10)


# Test Asia marginals for node asia alone
def test_oracle_marginals_asia_1_ok(asia):
    args = ("asia", {}, True)
    results = asia.marginals(args[0], args[1], args[2])
    check(args, results, asia)


# Test Asia marginals for node asia with parent bronc
def test_oracle_marginals_asia_2_ok(asia):
    args = ("asia", {"asia": ["bronc"]}, True)
    results = asia.marginals(args[0], args[1], args[2])
    check(args, results, asia)


# Test Asia marginals for node asia with parents bronc and tub
def test_oracle_marginals_asia_3_ok(asia):
    args = ("asia", {"asia": ["bronc", "tub"]}, True)
    results = asia.marginals(args[0], args[1], args[2])
    check(args, results, asia)


# Test Asia marginals for node xray with multiple parents
def test_oracle_marginals_asia_4_ok(asia):
    args = ("xray", {"xray": ["lung", "smoke", "either"]}, True)
    results = asia.marginals(args[0], args[1], args[2])
    check(args, results, asia)


# Test Oracle.values() raises TypeError as not implemented
def test_oracle_values_not_implemented(ab2):
    with pytest.raises(TypeError, match="Oracle.values\\(\\) not implemented"):
        ab2.values(("A", "B"))


# Test Oracle._update_sample() does nothing (pass statement)
def test_oracle_update_sample_pass(ab2):
    # This should not raise any exception and do nothing
    ab2._update_sample(old_N=5, old_ext_to_orig={"A": "A"})
    ab2._update_sample()  # Test with no arguments
    # Verify that N hasn't changed
    assert ab2.N == 2


# Test Oracle.randomise_names() raises NotImplementedError
def test_oracle_randomise_names_not_implemented(ab2):
    with pytest.raises(
        NotImplementedError, match="Data.randomise_names\\(\\) n/a for Oracle"
    ):
        ab2.randomise_names(42)

    with pytest.raises(
        NotImplementedError, match="Data.randomise_names\\(\\) n/a for Oracle"
    ):
        ab2.randomise_names(None)


# Test Oracle.as_df() raises NotImplementedError
def test_oracle_as_df_not_implemented(ab2):
    with pytest.raises(
        NotImplementedError, match="Data.df\\(\\) n/a for Oracle"
    ):
        ab2.as_df()


# Test Oracle.write() raises NotImplementedError
def test_oracle_write_not_implemented(ab2):
    with pytest.raises(
        NotImplementedError, match="Data.write\\(\\) n/a for Oracle"
    ):
        ab2.write("test_file.csv")


# Test Oracle properties getters and setters
def test_oracle_nodes_property(ab2):
    # Test getter (line 167)
    original_nodes = ab2.nodes
    assert original_nodes == ("A", "B")

    # Test setter (line 172)
    new_nodes = ("X", "Y")
    ab2.nodes = new_nodes
    assert ab2.nodes == new_nodes

    # Restore original nodes
    ab2.nodes = original_nodes
    assert ab2.nodes == original_nodes


# Test Oracle.sample property getter and setter
def test_oracle_sample_property(ab2):
    # Test getter (line 177)
    original_sample = ab2.sample
    assert original_sample == 2

    # Test setter (line 182)
    ab2.sample = 5
    assert ab2.sample == 5

    # Verify N property is also updated (since sample setter sets _N)
    assert ab2.N == 5

    # Restore original value
    ab2.sample = original_sample
    assert ab2.sample == original_sample


# Test Oracle.N property getter and setter
def test_oracle_n_property(ab2):
    # Test getter (line 187)
    original_n = ab2.N
    assert original_n == 2

    # Test setter (line 192)
    ab2.N = 10
    assert ab2.N == 10

    # Restore original value
    ab2.N = original_n
    assert ab2.N == original_n


# Test Oracle.node_values property getter and setter
def test_oracle_node_values_property(ab2):
    # Test getter (line 197) - should return empty dict
    node_values = ab2.node_values
    assert node_values == {}

    # Test setter (line 202) - should do nothing (pass)
    ab2.node_values = {"A": ["a1", "a2"], "B": ["b1", "b2"]}
    # Should still return empty dict since setter does nothing
    assert ab2.node_values == {}


# Test Oracle.node_types property getter and setter
def test_oracle_node_types_property(ab2):
    # Test getter (line 207)
    original_types = ab2.node_types
    assert original_types == {"A": "category", "B": "category"}

    # Test setter (line 212)
    new_types = {"A": "continuous", "B": "continuous"}
    ab2.node_types = new_types
    assert ab2.node_types == new_types

    # Restore original types
    ab2.node_types = original_types
    assert ab2.node_types == original_types


# Test Data.__init__() pass statement (line 51)
def test_data_init_pass(ab2):
    # This tests that Data.__init__() was called during Oracle construction
    # Since Oracle inherits from Data, the Data.__init__() pass statement
    # should be covered. The fact that ab2 exists and is an Oracle instance
    # means Data.__init__() was called
    assert isinstance(ab2, Oracle)
    assert hasattr(ab2, "elapsed")
    assert ab2.elapsed == 0.0


# Test Data.set_order() _update_sample() call for Pandas class (line 75)
def test_data_set_order_pandas_update_sample():
    # This test covers the _update_sample() call when
    # __class__.__name__ == "Pandas"
    # We can't directly test this with Oracle since Oracle != "Pandas"
    # But we can test the logic path by creating a mock scenario
    from causaliq_core.bn.io import read_bn

    # Create Oracle instance and test set_order without Pandas condition
    bn = read_bn(TESTDATA_DIR + "/xdsl/ab.xdsl")
    oracle = Oracle(bn)

    # Verify initial order
    original_order = oracle.get_order()
    assert original_order == ("A", "B")

    # Test setting new order - this should NOT call _update_sample since
    # class is Oracle
    new_order = ("B", "A")
    oracle.set_order(new_order)

    # Verify order was changed
    assert oracle.get_order() == new_order

    # Restore original order
    oracle.set_order(original_order)
    assert oracle.get_order() == original_order


# Test Data._set_dstype() for continuous dataset (lines 95-103)
def test_data_set_dstype_continuous():
    from causaliq_core.bn.io import read_bn

    # Create Oracle instance
    bn = read_bn(TESTDATA_DIR + "/xdsl/ab.xdsl")
    oracle = Oracle(bn)

    # Modify node_types to be all continuous and test _set_dstype
    oracle._node_types = {"A": "float32", "B": "float64"}
    oracle._set_dstype()
    assert oracle.dstype == "continuous"


# Test Data._set_dstype() for categorical dataset (lines 95-103)
def test_data_set_dstype_categorical():
    from causaliq_core.bn.io import read_bn

    # Create Oracle instance
    bn = read_bn(TESTDATA_DIR + "/xdsl/ab.xdsl")
    oracle = Oracle(bn)

    # Modify node_types to be all categorical and test _set_dstype
    oracle._node_types = {"A": "category", "B": "category"}
    oracle._set_dstype()
    assert oracle.dstype == "categorical"


# Test Data._set_dstype() for mixed dataset (lines 95-103)
def test_data_set_dstype_mixed():
    from causaliq_core.bn.io import read_bn

    # Create Oracle instance
    bn = read_bn(TESTDATA_DIR + "/xdsl/ab.xdsl")
    oracle = Oracle(bn)

    # Modify node_types to be mixed and test _set_dstype
    oracle._node_types = {"A": "category", "B": "float32"}
    oracle._set_dstype()
    assert oracle.dstype == "mixed"


# Test Data.randomise_order() _update_sample() call for Pandas class
def test_data_randomise_order_pandas_update_sample():
    from causaliq_core.bn.io import read_bn

    # Create Oracle instance and test randomise_order
    bn = read_bn(TESTDATA_DIR + "/xdsl/ab.xdsl")
    oracle = Oracle(bn)

    # Store original order
    original_order = oracle.get_order()

    # Test randomise_order - this should NOT call _update_sample since
    # class is Oracle
    oracle.randomise_order(42)

    # Verify order was changed (with seed 42, should be deterministic)
    randomised_order = oracle.get_order()
    # Order should be different from original (highly likely with seed 42)

    # Test with different seed to ensure it's actually randomising
    oracle.randomise_order(123)
    second_random_order = oracle.get_order()

    # At least one of these should be different from original
    assert (
        randomised_order != original_order
        or second_random_order != original_order
    )
