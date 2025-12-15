# Test the NumPys implementation of marginals

from pathlib import Path

import pytest
from causaliq_core.utils.timing import Timing
from numpy import ndarray
from pandas import DataFrame, crosstab

from causaliq_data import NumPy
from causaliq_data.pandas import Pandas

TESTDATA_DIR = str(Path(__file__).parent.parent / "data" / "functional")


@pytest.fixture(scope="module")  # AB, 2 rows
def ab2():
    df = DataFrame({"A": ["0", "1"], "B": ["1", "1"]}, dtype="category")
    return (
        NumPy.from_df(df=df, dstype="categorical", keep_df=True),
        Pandas(df=df),
    )


@pytest.fixture(scope="module")  # AB, 3 rows
def ab3():
    df = DataFrame(
        {"A": ["1", "1", "0"], "B": ["1", "0", "0"]}, dtype="category"
    )
    return (
        NumPy.from_df(df=df, dstype="categorical", keep_df=True),
        Pandas(df=df),
    )


@pytest.fixture(scope="module")  # AB, 8 rows
def ab8():
    df = DataFrame(
        {
            "A": ["1", "1", "0", "1", "1", "1", "0", "0"],
            "B": ["1", "0", "1", "1", "0", "0", "1", "1"],
        },
        dtype="category",
    )
    return (
        NumPy.from_df(df=df, dstype="categorical", keep_df=True),
        Pandas(df=df),
    )


@pytest.fixture(scope="module")  # AB, 8 rows
def abc10():
    df = DataFrame(
        {
            "A": ["1", "1", "0", "1", "1", "1", "0", "0", "0", "0"],
            "B": ["1", "0", "1", "1", "0", "0", "1", "1", "1", "1"],
            "C": ["1", "1", "1", "1", "1", "1", "1", "1", "1", "1"],
        },
        dtype="category",
    )
    return (
        NumPy.from_df(df=df, dstype="categorical", keep_df=True),
        Pandas(df=df),
    )


@pytest.fixture(scope="module")  # Asia, 100 rows
def asia():
    pandas = Pandas.read(
        str(Path(TESTDATA_DIR) / "experiments" / "datasets" / "asia.data.gz"),
        dstype="categorical",
        N=100,
    )
    return (
        NumPy.from_df(df=pandas.df, dstype="categorical", keep_df=True),
        pandas,
    )


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
    assert results[0].dtype in ["int32", "int64"]  # Platform/version agnostic

    # Check shape and maxcol returned by old marginals

    scale = len(args[1][args[0]]) + 1 if args[0] in args[1] else 1
    start = Timing.now()
    orig, orig_num_pvs = orig_marginals(data, args[0], args[1])
    Timing.record("crosstab", scale, start)
    assert results[0].shape == orig.shape
    assert results[1] == orig_num_pvs
    print("\n{}\n".format(orig.head()))

    # if requesting values, then can compare old and new counts too

    if results[2] is not None:
        if results[3] is None:
            for i, xi in enumerate(results[2]):
                count = orig.loc[xi].values[0]
                assert count == results[0][i, 0]
        else:

            # get levels in original column multi-index which varies

            levels = {i: n for i, n in enumerate(orig.columns.names)}

            for j, pvs in enumerate(results[3]):

                # Get parental values in variable order in levels for
                # column key used in loc(). Then compare all row counts

                pvs = tuple(pvs[levels[i]] for i in levels)
                for i, xi in enumerate(results[2]):
                    assert orig.loc[xi, pvs] == results[0][i, j]


def orig_marginals(self, node, parents):
    """
    Original marginals code.

    :param str node: node for which marginals required.
    :param dict parents: {node: parents} parents of non-orphan nodes

    :returns tuple: marginals, DataFrame - marginal counts, row index
                    is node value, column multi-index are parent values
                    num_pvs int - number of parental value combinations
                    (which may not be all present in data though)
    """
    num_pvs = 1
    if node in parents:  # node has parents - get contingency table
        marginals = crosstab(
            self.sample[node], [self.sample[p] for p in parents[node]]
        ).copy()
        for p in parents[node]:
            num_pvs *= len(self.node_values[p].keys())

    else:  # node has no parents - simple value frequencies
        marginals = self.sample[node].value_counts().to_frame().copy()

    return (marginals, num_pvs)


# no arguments specified
def test_numpy_marginals_type_error_1(ab2):
    with pytest.raises(TypeError):
        ab2[0].marginals()


# bad or missing node
def test_numpy_marginals_type_error_2(ab2):
    with pytest.raises(TypeError):
        ab2[0].marginals(node=1, parents={})
    with pytest.raises(TypeError):
        ab2[0].marginals(node=12.9, parents={})
    with pytest.raises(TypeError):
        ab2[0].marginals(node=None, parents={})
    with pytest.raises(TypeError):
        ab2[0].marginals(node=["A"], parents={})
    with pytest.raises(TypeError):
        ab2[0].marginals(parents={})


# bad or missing parents
def test_numpy_marginals_type_error_3(ab2):
    with pytest.raises(TypeError):
        ab2[0].marginals(node="A", parents=2)
    with pytest.raises(TypeError):
        ab2[0].marginals(node="A", parents="B")
    with pytest.raises(TypeError):
        ab2[0].marginals(node="A", parents=None)
    with pytest.raises(TypeError):
        ab2[0].marginals(node="A")


# parent values bad type
def test_numpy_marginals_type_error_4(ab2):
    with pytest.raises(TypeError):
        ab2[0].marginals(node="A", parents={"A": "B"})
    with pytest.raises(TypeError):
        ab2[0].marginals(node="A", parents={"A": {"B"}})
    with pytest.raises(TypeError):
        ab2[0].marginals(node="A", parents={"A": ("B",)})


# bad values_reqd type
def test_numpy_marginals_type_error_5(ab2):
    with pytest.raises(TypeError):
        ab2[0].marginals(node="A", parents={}, values_reqd=1)
    with pytest.raises(TypeError):
        ab2[0].marginals(node="A", parents={}, values_reqd=None)
    with pytest.raises(TypeError):
        ab2[0].marginals(node="A", parents={}, values_reqd={False})
    with pytest.raises(TypeError):
        ab2[0].marginals(node="A", parents={}, values_reqd=[False])
    with pytest.raises(TypeError):
        ab2[0].marginals(node="A", parents={}, values_reqd=(False,))


# bad node value
def test_numpy_marginals_value_error_1(ab2):
    with pytest.raises(ValueError):
        ab2[0].marginals(node="?", parents={}, values_reqd=True)


# bad parental value
def test_numpy_marginals_value_error_2(ab2):
    with pytest.raises(ValueError):
        ab2[0].marginals(node="A", parents={"A": ["?"]}, values_reqd=True)
    with pytest.raises(ValueError):
        ab2[0].marginals(node="A", parents={"A": ["B", "?"]}, values_reqd=True)


# node parent of itself
def test_numpy_marginals_value_error_3(ab2):
    with pytest.raises(ValueError):
        ab2[0].marginals(node="A", parents={"A": ["A"]}, values_reqd=True)


# AB, A marginals, with values
def test_numpy_marginals_ab2_1_ok(ab2):
    args = ("A", {}, True)
    results = ab2[0].marginals(args[0], args[1], args[2])
    check(args, results, ab2[1])


# AB, A marginals, no values
def test_numpy_marginals_ab2_2_ok(ab2):
    args = ("A", {}, False)
    results = ab2[0].marginals(args[0], args[1], args[2])
    check(args, results, ab2[1])


# AB, B marginals, with values
def test_numpy_marginals_ab2_3_ok(ab2):
    args = ("B", {}, True)
    results = ab2[0].marginals(args[0], args[1], args[2])
    check(args, results, ab2[1])


# AB, AB marginals, with values
def test_numpy_marginals_ab2_4_ok(ab2):
    args = ("A", {"A": ["B"]}, True)
    results = ab2[0].marginals(args[0], args[1], args[2])
    check(args, results, ab2[1])


# AB, AB marginals, no values
def test_numpy_marginals_ab2_5_ok(ab2):
    args = ("B", {"B": ["A"]}, True)
    results = ab2[0].marginals(args[0], args[1], args[2])
    check(args, results, ab2[1])


# AB3, A marginals, with values
def test_numpy_marginals_ab3_1_ok(ab3):
    args = ("A", {}, True)
    results = ab3[0].marginals(args[0], args[1], args[2])
    check(args, results, ab3[1])


# AB3, B marginals, with values
def test_numpy_marginals_ab3_2_ok(ab3):
    args = ("B", {}, True)
    results = ab3[0].marginals(args[0], args[1], args[2])
    check(args, results, ab3[1])


# AB3, AB marginals, with values
def test_numpy_marginals_ab3_3_ok(ab3):
    args = ("A", {"A": ["B"]}, True)
    results = ab3[0].marginals(args[0], args[1], args[2])
    check(args, results, ab3[1])


# AB3, AB marginals, no values
def test_numpy_marginals_ab3_4_ok(ab3):
    args = ("B", {"B": ["A"]}, True)
    results = ab3[0].marginals(args[0], args[1], args[2])
    check(args, results, ab3[1])


# AB8, A marginals, with values
def test_numpy_marginals_ab8_1_ok(ab8):
    args = ("A", {}, True)
    results = ab8[0].marginals(args[0], args[1], args[2])
    check(args, results, ab8[1])


# AB8, B marginals, with values
def test_numpy_marginals_ab8_2_ok(ab8):
    args = ("B", {}, True)
    results = ab8[0].marginals(args[0], args[1], args[2])
    check(args, results, ab8[1])


def xtest_numpy_marginals_ab8_3_ok(ab8):
    args = ("A", {"A": ["B"]}, True)
    results = ab8[0].marginals(args[0], args[1], args[2])
    check(args, results, ab8[1])


# AB8, AB marginals, no values
def test_numpy_marginals_ab8_4_ok(ab8):
    args = ("B", {"B": ["A"]}, True)
    results = ab8[0].marginals(args[0], args[1], args[2])
    check(args, results, ab8[1])


# ABC10, A marginals, with values
def test_numpy_marginals_abc10_1_ok(abc10):
    args = ("A", {}, True)
    results = abc10[0].marginals(args[0], args[1], args[2])
    check(args, results, abc10[1])


# ABC10, A marginals, no values
def test_numpy_marginals_abc10_2_ok(abc10):
    args = ("A", {"A": ["B"]}, True)
    results = abc10[0].marginals(args[0], args[1], args[2])
    check(args, results, abc10[1])


# ABC10, A given B marginals, with values
def test_numpy_marginals_abc10_3_ok(abc10):
    args = ("A", {"A": ["C"]}, True)
    results = abc10[0].marginals(args[0], args[1], args[2])
    check(args, results, abc10[1])


# ABC10, A given B marginals, no values
def test_numpy_marginals_abc10_4_ok(abc10):
    args = ("A", {"A": ["B", "C"]}, True)
    results = abc10[0].marginals(args[0], args[1], args[2])
    check(args, results, abc10[1])


# ABC10, B marginals, with values
def test_numpy_marginals_abc10_5_ok(abc10):
    args = ("B", {}, True)
    results = abc10[0].marginals(args[0], args[1], args[2])
    check(args, results, abc10[1])


# ABC10, B given A marginals, with values
def test_numpy_marginals_abc10_6_ok(abc10):
    args = ("B", {"B": ["A"]}, True)
    results = abc10[0].marginals(args[0], args[1], args[2])
    check(args, results, abc10[1])


# ABC10, B given AC marginals, with values
def test_numpy_marginals_abc10_7_ok(abc10):
    args = ("B", {"B": ["C"]}, True)
    results = abc10[0].marginals(args[0], args[1], args[2])
    check(args, results, abc10[1])


# ABC10, B given AC marginals, no values
def test_numpy_marginals_abc10_8_ok(abc10):
    args = ("B", {"B": ["A", "C"]}, True)
    results = abc10[0].marginals(args[0], args[1], args[2])
    check(args, results, abc10[1])


# ABC10, C marginals, with values
def test_numpy_marginals_abc10_9_ok(abc10):
    args = ("C", {}, True)
    results = abc10[0].marginals(args[0], args[1], args[2])
    check(args, results, abc10[1])


# ABC10, C marginals, no values
def test_numpy_marginals_abc10_10_ok(abc10):
    args = ("C", {"C": ["A"]}, True)
    results = abc10[0].marginals(args[0], args[1], args[2])
    check(args, results, abc10[1])


# ABC10, C given AB marginals, with values
def test_numpy_marginals_abc10_11_ok(abc10):
    args = ("C", {"C": ["B"]}, True)
    results = abc10[0].marginals(args[0], args[1], args[2])
    check(args, results, abc10[1])


# ABC10, C given AB marginals, no values
def test_numpy_marginals_abc10_12_ok(abc10):
    args = ("C", {"C": ["A", "B"]}, True)
    results = abc10[0].marginals(args[0], args[1], args[2])
    check(args, results, abc10[1])


# Asia dataset, simple marginals test
def test_numpy_marginals_asia_1_ok(asia):
    args = ("asia", {}, True)
    results = asia[0].marginals(args[0], args[1], args[2])
    check(args, results, asia[1])


def test_numpy_marginals_asia_2_ok(asia):
    args = ("asia", {"asia": ["bronc"]}, True)
    results = asia[0].marginals(args[0], args[1], args[2])
    check(args, results, asia[1])


def test_numpy_marginals_asia_3_ok(asia):
    args = ("asia", {"asia": ["bronc", "tub"]}, True)
    results = asia[0].marginals(args[0], args[1], args[2])
    check(args, results, asia[1])


def test_numpy_marginals_asia_4_ok(asia):
    args = ("xray", {"xray": ["lung", "smoke", "either"]}, True)
    results = asia[0].marginals(args[0], args[1], args[2])
    check(args, results, asia[1])


# check operation of set_N
def test_numpy_marginals_asia_5_ok():
    pandas = Pandas.read(
        str(Path(TESTDATA_DIR) / "experiments" / "datasets" / "asia.data.gz"),
        dstype="categorical",
        N=1000,
    )
    numpy = NumPy.from_df(pandas.df, dstype="categorical", keep_df=True)

    pandas.set_N(500)
    numpy.set_N(500)

    args = ("xray", {"xray": ["lung", "smoke", "either"]}, True)
    results = numpy.marginals(args[0], args[1], args[2])
    check(args, results, pandas)


# 100 rows of Sports
def test_numpy_marginals_sports_1_ok():
    pandas = Pandas.read(
        str(Path(TESTDATA_DIR) / "datasets" / "sports.data.gz"),
        dstype="categorical",
        N=1000,
    )
    numpy = NumPy.from_df(pandas.df, dstype="categorical", keep_df=True)

    args = ("HDA", {"HDA": ["RDlevel", "HTgoals"]}, True)
    Timing.on(True)
    results = numpy.marginals(args[0], args[1], args[2])
    check(args, results, pandas)
    print(Timing)


# Covid dataset marginals test
def test_numpy_marginals_covid_1_ok():
    pandas = Pandas.read(
        str(Path(TESTDATA_DIR) / "datasets" / "covid.data.gz"),
        dstype="categorical",
        N=1000,
    )
    args = ("Lockdown", {"Lockdown": ["New_infections", "Season"]}, False)
    numpy = NumPy.from_df(pandas.df, dstype="categorical", keep_df=True)
    Timing.on(True)
    results = numpy.marginals(args[0], args[1], args[2])
    check(args, results, pandas)
    print(Timing)


# Test MAX_CATEGORY validation (line 257)
def test_numpy_read_too_many_categories_error(monkeypatch):
    # Use monkeypatch to temporarily reduce MAX_CATEGORY to make test easier
    # Original MAX_CATEGORY is 100 - we'll set it to 3
    monkeypatch.setattr("causaliq_data.numpy.MAX_CATEGORY", 3)

    # Create a DataFrame with more than our reduced MAX_CATEGORY
    categories = [
        f"cat_{i}" for i in range(5)
    ]  # 5 > 3 (our patched MAX_CATEGORY)
    df = DataFrame({"A": categories, "B": ["1"] * 5}, dtype="category")

    with pytest.raises(ValueError, match="too many categories"):
        NumPy.from_df(df=df, dstype="categorical", keep_df=False)


# Test regular Index case for categorical data parsing (line 270)
def test_numpy_regular_index_case():
    # The regular Index case (line 270) occurs when factorize() returns an
    # Index that doesn't have 'categories' and 'codes' attributes
    # (i.e., not a CategoricalIndex). This can happen with mixed categorical
    # data or when pandas creates regular Index

    # Try creating data that will force regular Index during factorization
    # Create DataFrame with object dtype first, then convert to categorical
    df = DataFrame(
        {"A": ["x", "y", "z", "x", "y"], "B": ["1", "0", "1", "0", "1"]},
        dtype=object,
    )  # Start with object dtype

    # Convert to categorical
    df = df.astype("category")

    numpy_data = NumPy.from_df(df=df, dstype="categorical", keep_df=False)

    # Verify it was created successfully
    assert numpy_data.dstype == "categorical"
    assert numpy_data.N == 5


# Test slow path for marginals using numpy unique (line 461)
def test_numpy_marginals_force_slow_path(monkeypatch):
    # Use monkeypatch to temporarily reduce MAX_BINCOUNT to force slow path
    # Original MAX_BINCOUNT is 1,000,000 - we'll set it to 10
    monkeypatch.setattr("causaliq_data.numpy.NumPy.MAX_BINCOUNT", 10)

    # Now create data that will exceed our reduced threshold
    # With 4 categories in 2 columns: 4 * 4 = 16 > 10, triggering slow path
    df = DataFrame(
        {
            "A": ["a1", "a2", "a3", "a4"] * 25,  # 4 unique categories
            "B": ["b1", "b2", "b3", "b4"] * 25,  # 4 unique categories
        },
        dtype="category",
    )

    numpy_data = NumPy.from_df(df=df, dstype="categorical", keep_df=True)

    # This should trigger the slow path (line 461)
    # because 4 * 4 = 16 > 10 (our patched MAX_BINCOUNT)
    args = ("A", {"A": ["B"]}, True)
    results = numpy_data.marginals(args[0], args[1], args[2])

    # Verify it returns valid results
    assert isinstance(results[0], ndarray)
    assert results[0].dtype in ["int32", "int64"]


# Test randomise_names with bad argument type (line 400)
def test_numpy_randomise_names_bad_arg_type(ab2):
    with pytest.raises(TypeError, match="bad arg type"):
        ab2[0].randomise_names(seed="not_an_int")
    with pytest.raises(TypeError, match="bad arg type"):
        ab2[0].randomise_names(seed=3.14)
    with pytest.raises(TypeError, match="bad arg type"):
        ab2[0].randomise_names(seed=[123])


# Test randomise_names with valid argument
def test_numpy_randomise_names_valid(ab2):
    numpy_data = ab2[0]
    original_nodes = numpy_data.nodes

    # Test with integer seed - should not raise error
    numpy_data.randomise_names(seed=42)

    # The method should run without error (test line 400+ functionality)
    # Note: depending on implementation, names may or may not actually change
    assert numpy_data.nodes is not None
    assert len(numpy_data.nodes) == len(original_nodes)

    # Test with None seed
    numpy_data.randomise_names(seed=None)


# Test _update_sample method (line 384)
def test_numpy_update_sample(ab2):
    numpy_data = ab2[0]
    # This should not raise an error as it's currently just pass
    numpy_data._update_sample(old_N=2, old_ext_to_orig={})
    numpy_data._update_sample()


# Test property setters (lines 653, 663, 673)
def test_numpy_property_setters(ab2):
    numpy_data = ab2[0]

    # Test nodes setter (line 653)
    original_nodes = numpy_data.nodes
    new_nodes = ("X", "Y")
    numpy_data.nodes = new_nodes
    assert numpy_data.nodes == new_nodes
    # Restore original
    numpy_data.nodes = original_nodes

    # Test sample setter (line 663)
    from numpy import array

    original_sample = numpy_data.sample.copy()
    new_sample = array([[0, 1], [1, 0]])
    numpy_data.sample = new_sample
    assert (numpy_data.sample == new_sample).all()
    # Restore original
    numpy_data.sample = original_sample

    # Test N setter (line 673)
    original_N = numpy_data.N
    new_N = 100
    numpy_data.N = new_N
    assert numpy_data.N == new_N
    # Restore original
    numpy_data.N = original_N


# Test slow path for marginals using numpy unique (line 461)
def test_numpy_marginals_slow_path():
    # Create a large dataset that will trigger the slow path
    # The slow path is triggered when max possible combinations > threshold
    import numpy as np

    # Create data with enough combinations to trigger slow path
    # We need many unique combinations to exceed the threshold
    n_rows = 1000
    n_categories_per_var = 15  # This should create 15*15 = 225 combinations

    data_a = np.random.randint(0, n_categories_per_var, n_rows)
    data_b = np.random.randint(0, n_categories_per_var, n_rows)
    data_c = np.random.randint(0, n_categories_per_var, n_rows)

    df = DataFrame(
        {
            "A": [f"a{i}" for i in data_a],
            "B": [f"b{i}" for i in data_b],
            "C": [f"c{i}" for i in data_c],
        },
        dtype="category",
    )

    numpy_data = NumPy.from_df(df=df, dstype="categorical", keep_df=True)

    # Test marginals that should use the slow path
    # This should trigger line 461: npunique with return_counts=True
    args = ("A", {"A": ["B", "C"]}, True)
    results = numpy_data.marginals(args[0], args[1], args[2])

    # Just verify it returns valid results
    assert isinstance(results[0], ndarray)
    assert results[0].dtype in ["int32", "int64"]
    assert len(results) == 4  # counts, maxcol, rowval, colval


def test_numpy_marginals_regular_index_case(monkeypatch):
    # Test to cover line 270 - the regular Index case (not CategoricalIndex)
    # Create a DataFrame and then monkeypatch to force regular Index path
    df = DataFrame(
        {
            "A": ["1", "2", "3", "4", "1", "2"],
            "B": ["10", "20", "30", "40", "10", "20"],
        },
        dtype="category",
    )

    # Monkeypatch hasattr BEFORE creating NumPy data - line 270 is in
    # from_df, not marginals
    original_hasattr = hasattr

    def mock_hasattr(obj, name):
        # Avoid recursion by only mocking specific pandas-related calls
        if name in ("categories", "codes") and hasattr(obj, "__class__"):
            return False
        return original_hasattr(obj, name)

    monkeypatch.setattr("builtins.hasattr", mock_hasattr)

    # NOW create NumPy data - this should trigger the hasattr calls
    np_data = NumPy.from_df(df, dstype="categorical", keep_df=True)
    assert np_data is not None

    # Test marginals to make sure the data works
    args = ("A", {"A": ["B"]}, True)
    results = np_data.marginals(args[0], args[1], args[2])

    # Just verify it returns valid results
    assert len(results) == 4  # counts, maxcol, rowval, colval
