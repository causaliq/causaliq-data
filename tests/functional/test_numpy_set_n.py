"""
Tests for NumPy data adapter set_N method with file-based data.

These tests validate set_N functionality that
requires external test data files.
"""

import os

from causaliq_data import NumPy, Pandas

# Path to test data relative to this file
TESTDATA_DIR = os.path.join(
    os.path.dirname(__file__), "..", "data", "functional"
)


def test_set_N_abc5_1_ok():  # ABC, 5 discrete rows, randomising order
    pandas = Pandas.read(
        TESTDATA_DIR + "/simple/abc_5.csv", dstype="categorical"
    )
    data = NumPy.from_df(df=pandas.as_df(), dstype="categorical", keep_df=True)

    print("\n\nOriginal Dataset:\n{}\n".format(data.as_df()))
    assert data.node_values == {
        "A": {"1": 3, "0": 2},
        "B": {"1": 3, "0": 2},
        "C": {"0": 4, "1": 1},
    }
    assert data.as_df().to_dict(orient="list") == {
        "A": ["0", "0", "1", "1", "1"],
        "B": ["0", "1", "0", "1", "1"],
        "C": ["0", "0", "0", "1", "0"],
    }

    data.set_N(3)
    print("\n\nSetting N=3:\n{}\n".format(data.as_df()))
    assert data.node_values == {
        "A": {"0": 2, "1": 1},
        "B": {"0": 2, "1": 1},
        "C": {"0": 3},
    }
    assert data.as_df().to_dict(orient="list") == {
        "A": ["0", "0", "1"],
        "B": ["0", "1", "0"],
        "C": ["0", "0", "0"],
    }

    data.set_N(4, seed=1)
    print("\n\nSetting N=4, seed=1:\n{}\n".format(data.as_df()))
    assert data.node_values == {
        "A": {"0": 2, "1": 2},
        "B": {"0": 2, "1": 2},
        "C": {"0": 3, "1": 1},
    }
    assert data.as_df().to_dict(orient="list") == {
        "A": ["0", "0", "1", "1"],
        "B": ["0", "1", "0", "1"],
        "C": ["0", "0", "0", "1"],
    }

    data.set_N(4, seed=2)
    print("\n\nSetting N=4, seed=2:\n{}\n".format(data.as_df()))
    assert data.node_values == {
        "A": {"0": 2, "1": 2},
        "B": {"0": 2, "1": 2},
        "C": {"0": 3, "1": 1},
    }
    assert data.as_df().to_dict(orient="list") == {
        "A": ["1", "1", "0", "0"],
        "B": ["1", "0", "0", "1"],
        "C": ["1", "0", "0", "0"],
    }

    data.set_N(5)
    print("\n\nSetting N=5, no seed:\n{}\n".format(data.as_df()))
    assert data.node_values == {
        "A": {"1": 3, "0": 2},
        "B": {"1": 3, "0": 2},
        "C": {"0": 4, "1": 1},
    }
    assert data.as_df().to_dict(orient="list") == {
        "A": ["0", "0", "1", "1", "1"],
        "B": ["0", "1", "0", "1", "1"],
        "C": ["0", "0", "0", "1", "0"],
    }


def test_set_N_abc5_2_ok():  # ABC, 5 discrete rows, randomising selection
    pandas = Pandas.read(
        TESTDATA_DIR + "/simple/abc_5.csv", dstype="categorical"
    )
    data = NumPy.from_df(df=pandas.as_df(), dstype="categorical", keep_df=True)

    print("\n\nOriginal Dataset:\n{}\n".format(data.as_df()))
    assert data.node_values == {
        "A": {"1": 3, "0": 2},
        "B": {"1": 3, "0": 2},
        "C": {"0": 4, "1": 1},
    }
    assert data.as_df().to_dict(orient="list") == {
        "A": ["0", "0", "1", "1", "1"],
        "B": ["0", "1", "0", "1", "1"],
        "C": ["0", "0", "0", "1", "0"],
    }

    data.set_N(N=3, random_selection=True)
    print("\n\nN=3, seed=None, random selection:\n{}\n".format(data.as_df()))
    assert data.as_df().to_dict(orient="list") == {
        "A": ["1", "1", "1"],
        "B": ["0", "1", "1"],
        "C": ["0", "1", "0"],
    }

    data.set_N(N=3, seed=1, random_selection=True)
    print("\n\nN=3, seed=1, random selection:\n{}\n".format(data.as_df()))
    assert data.as_df().to_dict(orient="list") == {
        "A": ["1", "0", "1"],
        "B": ["0", "1", "1"],
        "C": ["0", "0", "1"],
    }

    data.set_N(N=2, seed=None, random_selection=True)
    print("\n\nN=2, seed=None, random selection:\n{}\n".format(data.as_df()))
    assert data.as_df().to_dict(orient="list") == {
        "A": ["1", "1"],
        "B": ["1", "1"],
        "C": ["1", "0"],
    }

    data.set_N(N=5)
    print("\n\nN=5, seed=None:\n{}\n".format(data.as_df()))
    assert data.node_values == {
        "A": {"1": 3, "0": 2},
        "B": {"1": 3, "0": 2},
        "C": {"0": 4, "1": 1},
    }
    assert data.as_df().to_dict(orient="list") == {
        "A": ["0", "0", "1", "1", "1"],
        "B": ["0", "1", "0", "1", "1"],
        "C": ["0", "0", "0", "1", "0"],
    }


def test_set_N_xyz10_1_ok():  # XYZ, 10 continuous rows, randmising order
    pandas = Pandas.read(
        TESTDATA_DIR + "/simple/xyz_10.csv", dstype="continuous"
    )
    data = NumPy.from_df(df=pandas.as_df(), dstype="continuous", keep_df=True)

    print("\n\nOriginal Dataset:\n{}\n".format(data.as_df()))
    rdf = data.as_df().map(lambda x: round(x, 1))
    assert rdf.to_dict(orient="list") == {
        "X": [0.0, 0.1, 0.2, 0.6, 1.1, 2.2, 4.0, 4.4, 6.0, 7.1],
        "Y": [3.1, 0.0, 5.4, 2.8, 0.3, 3.1, 6.0, 6.6, 0.2, 3.9],
        "Z": [4.0, 2.2, 1.7, 9.9, 0.3, 0.8, 9.0, 1.9, 0.5, 1.4],
    }

    data.set_N(6)
    print("\n\nSetting N=6, no seed:\n{}\n".format(data.as_df()))
    rdf = data.as_df().map(lambda x: round(x, 1))
    assert rdf.to_dict(orient="list") == {
        "X": [0.0, 0.2, 0.6, 1.1, 4.0, 4.4],
        "Y": [3.1, 5.4, 2.8, 0.3, 6.0, 6.6],
        "Z": [4.0, 1.7, 9.9, 0.3, 9.0, 1.9],
    }

    data.set_N(4, seed=3)
    print("\n\nSetting N=4, seed=3:\n{}\n".format(data.as_df()))
    rdf = data.as_df().map(lambda x: round(x, 1))
    assert rdf.to_dict(orient="list") == {
        "X": [0.0, 0.2, 1.1, 4.4],
        "Y": [3.1, 5.4, 0.3, 6.6],
        "Z": [4.0, 1.7, 0.3, 1.9],
    }

    data.set_N(10)
    print("\n\nSetting N=10, no seed:\n{}\n".format(data.as_df()))
    rdf = data.as_df().map(lambda x: round(x, 1))
    assert rdf.to_dict(orient="list") == {
        "X": [0.0, 0.1, 0.2, 0.6, 1.1, 2.2, 4.0, 4.4, 6.0, 7.1],
        "Y": [3.1, 0.0, 5.4, 2.8, 0.3, 3.1, 6.0, 6.6, 0.2, 3.9],
        "Z": [4.0, 2.2, 1.7, 9.9, 0.3, 0.8, 9.0, 1.9, 0.5, 1.4],
    }

    data.set_N(4, seed=3)
    print("\n\nSetting N=4, seed=3:\n{}\n".format(data.as_df()))
    rdf = data.as_df().map(lambda x: round(x, 1))
    assert rdf.to_dict(orient="list") == {
        "X": [0.0, 0.2, 1.1, 4.4],
        "Y": [3.1, 5.4, 0.3, 6.6],
        "Z": [4.0, 1.7, 0.3, 1.9],
    }


def test_set_N_asia_1_ok():  # Asia, N=100 - set N to 50
    pandas = Pandas.read(
        TESTDATA_DIR + "/experiments/datasets/asia.data.gz",
        dstype="categorical",
        N=100,
    )
    data = NumPy.from_df(df=pandas.as_df(), dstype="categorical", keep_df=True)

    data.set_N(50)

    assert isinstance(data, NumPy)
    assert data.order == (0, 1, 2, 3, 4, 5, 6, 7)
    assert data.nodes == (
        "asia",
        "bronc",
        "dysp",
        "either",
        "lung",
        "smoke",
        "tub",
        "xray",
    )
    assert data.N == 50
    assert data.ext_to_orig == {
        "asia": "asia",
        "bronc": "bronc",
        "dysp": "dysp",
        "either": "either",
        "lung": "lung",
        "smoke": "smoke",
        "tub": "tub",
        "xray": "xray",
    }
    assert data.orig_to_ext == {
        "asia": "asia",
        "bronc": "bronc",
        "dysp": "dysp",
        "either": "either",
        "lung": "lung",
        "smoke": "smoke",
        "tub": "tub",
        "xray": "xray",
    }
    assert data.node_values == {
        "asia": {"no": 47, "yes": 3},
        "bronc": {"no": 29, "yes": 21},
        "dysp": {"no": 28, "yes": 22},
        "either": {"no": 45, "yes": 5},
        "lung": {"no": 46, "yes": 4},
        "smoke": {"yes": 26, "no": 24},
        "tub": {"no": 49, "yes": 1},
        "xray": {"no": 44, "yes": 6},
    }
    assert data.node_types == {
        "asia": "category",
        "bronc": "category",
        "dysp": "category",
        "either": "category",
        "lung": "category",
        "smoke": "category",
        "tub": "category",
        "xray": "category",
    }
    assert data.dstype == "categorical"

    assert data.get_order() == (
        "asia",
        "bronc",
        "dysp",
        "either",
        "lung",
        "smoke",
        "tub",
        "xray",
    )

    assert (data.as_df() == pandas.as_df()[:50]).all().all()
