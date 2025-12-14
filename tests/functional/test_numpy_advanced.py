"""
Tests for NumPy data adapter advanced functionality with file-based data.

These tests validate complex operations like set_order and randomise_names
that require external test data files.
"""

import os

from causaliq_core.bn.io import read_bn

from causaliq_data import NumPy, Pandas

# Path to test data relative to this file
TESTDATA_DIR = os.path.join(
    os.path.dirname(__file__), "..", "data", "functional"
)


def test_set_order_asia_1_ok():  # Asia, N=100 - optimal/worst/original order
    bn = read_bn(TESTDATA_DIR + "/discrete/small/asia.dsc")
    pandas = Pandas.read(
        TESTDATA_DIR + "/experiments/datasets/asia.data.gz",
        dstype="categorical",
        N=100,
    )
    data = NumPy.from_df(df=pandas.as_df(), dstype="categorical", keep_df=True)
    orig_data = data.data.copy()

    std_order = data.nodes

    # switch to optimal order

    order = tuple(bn.dag.ordered_nodes())
    assert order == (
        "asia",
        "smoke",
        "bronc",
        "lung",
        "tub",
        "either",
        "dysp",
        "xray",
    )
    data.set_order(order)

    assert isinstance(data, NumPy)
    assert data.order == (0, 5, 1, 4, 6, 3, 2, 7)
    assert data.nodes == std_order
    assert data.N == 100
    assert data.ext_to_orig == {n: n for n in std_order}
    assert data.orig_to_ext == {n: n for n in std_order}

    # Note get_order() and as_df() DO reflect new order, but data unchanged

    assert data.get_order() == order
    assert tuple(data.as_df().columns) == order
    assert (data.data == orig_data).all().all()

    # switch to worst order

    order = order[::-1]
    assert order == (
        "xray",
        "dysp",
        "either",
        "tub",
        "lung",
        "bronc",
        "smoke",
        "asia",
    )
    data.set_order(order)

    assert isinstance(data, NumPy)
    assert data.order == (7, 2, 3, 6, 4, 1, 5, 0)
    assert data.nodes == std_order
    assert data.N == 100
    assert data.ext_to_orig == {n: n for n in std_order}
    assert data.orig_to_ext == {n: n for n in std_order}

    # Note get_order() and as_df() DO reflect new order, but data unchanged

    assert data.get_order() == order
    assert tuple(data.as_df().columns) == order
    assert (data.data == orig_data).all().all()

    # revert to standard order

    data.set_order(std_order)

    assert isinstance(data, NumPy)
    assert data.order == (0, 1, 2, 3, 4, 5, 6, 7)
    assert data.nodes == std_order
    assert data.N == 100
    assert data.ext_to_orig == {n: n for n in std_order}
    assert data.orig_to_ext == {n: n for n in std_order}

    # Note get_order() and as_df() DO reflect new order, but data unchanged

    assert data.get_order() == std_order
    assert tuple(data.as_df().columns) == std_order
    assert (data.data == orig_data).all().all()


def test_rand_name_xyz10_1_ok():  # XYZ, N10 - randomise names
    pandas = Pandas.read(
        TESTDATA_DIR + "/simple/xyz_10.csv", dstype="continuous"
    )
    data = NumPy.from_df(df=pandas.as_df(), dstype="continuous", keep_df=True)
    orig_data = data.data.copy()
    assert tuple(data.as_df().columns) == ("X", "Y", "Z")
    assert data.get_order() == ("X", "Y", "Z")

    print("\n\nOriginal XYZ names:\n{}".format(data.as_df().head()))
    assert data.get_order() == ("X", "Y", "Z")
    assert data.node_values == {}
    assert data.node_types == {"X": "float32", "Y": "float32", "Z": "float32"}

    data.randomise_names(seed=0)
    print("\n\nRandom names, seed 0:\n{}".format(data.as_df().head()))
    assert tuple(data.as_df().columns) == ("X001X", "X002Y", "X000Z")
    assert data.get_order() == ("X001X", "X002Y", "X000Z")
    assert data.node_values == {}
    assert data.node_types == {
        "X001X": "float32",
        "X002Y": "float32",
        "X000Z": "float32",
    }
    assert (data.data == orig_data).all().all()

    data.randomise_names(seed=1)
    print("\n\nRandom names, seed 1:\n{}".format(data.as_df().head()))
    assert tuple(data.as_df().columns) == ("X002X", "X000Y", "X001Z")
    assert data.get_order() == ("X002X", "X000Y", "X001Z")
    assert data.node_values == {}
    assert data.node_types == {
        "X002X": "float32",
        "X000Y": "float32",
        "X001Z": "float32",
    }
    assert (data.data == orig_data).all().all()

    data.randomise_names(seed=0)
    print("\n\nRandom names, seed 0:\n{}".format(data.as_df().head()))
    assert tuple(data.as_df().columns) == ("X001X", "X002Y", "X000Z")
    assert data.get_order() == ("X001X", "X002Y", "X000Z")
    assert data.node_values == {}
    assert data.node_types == {
        "X001X": "float32",
        "X002Y": "float32",
        "X000Z": "float32",
    }
    assert (data.data == orig_data).all().all()

    data.randomise_names()
    print("\n\nRandom names, seed None:\n{}".format(data.as_df().head()))
    assert tuple(data.as_df().columns) == ("X", "Y", "Z")
    assert data.get_order() == ("X", "Y", "Z")
    assert data.node_values == {}
    assert data.node_types == {"X": "float32", "Y": "float32", "Z": "float32"}
    assert (data.data == orig_data).all().all()


def test_rand_name_asia_1_ok():  # Asia, N=100 - randomise names
    bn = read_bn(TESTDATA_DIR + "/discrete/small/asia.dsc")
    pandas = Pandas.read(
        TESTDATA_DIR + "/experiments/datasets/asia.data.gz",
        dstype="categorical",
        N=100,
    )
    data = NumPy.from_df(df=pandas.as_df(), dstype="categorical", keep_df=True)
    std_order = tuple(bn.dag.nodes)
    orig_data = data.data.copy()

    data.randomise_names(seed=0)

    assert isinstance(data, NumPy)
    assert (data.data == orig_data).all().all()  # NB data.df unchanged
    assert data.order == (0, 1, 2, 3, 4, 5, 6, 7)  # order unchanged
    assert data.nodes == std_order  # always original order
    assert data.N == 100  # size unchanged
    assert data.ext_to_orig == {
        "X004asia": "asia",
        "X007bronc": "bronc",
        "X000dysp": "dysp",
        "X006either": "either",
        "X002lung": "lung",
        "X003smoke": "smoke",
        "X005tub": "tub",
        "X001xray": "xray",
    }
    assert data.orig_to_ext == {
        "asia": "X004asia",
        "bronc": "X007bronc",
        "dysp": "X000dysp",
        "either": "X006either",
        "lung": "X002lung",
        "smoke": "X003smoke",
        "tub": "X005tub",
        "xray": "X001xray",
    }
    assert data.node_values == {
        "X004asia": {"no": 97, "yes": 3},
        "X007bronc": {"no": 56, "yes": 44},
        "X000dysp": {"no": 56, "yes": 44},
        "X006either": {"no": 92, "yes": 8},
        "X002lung": {"no": 93, "yes": 7},
        "X003smoke": {"no": 57, "yes": 43},
        "X005tub": {"no": 99, "yes": 1},
        "X001xray": {"no": 90, "yes": 10},
    }
    assert data.node_types == {
        "X004asia": "category",
        "X007bronc": "category",
        "X000dysp": "category",
        "X006either": "category",
        "X002lung": "category",
        "X003smoke": "category",
        "X005tub": "category",
        "X001xray": "category",
    }
    assert data.dstype == "categorical"
    assert data.get_order() == tuple(data.ext_to_orig)
    assert tuple(data.as_df().columns) == tuple(data.ext_to_orig)
    assert (data.data == orig_data).all().all()
