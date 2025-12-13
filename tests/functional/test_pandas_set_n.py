# Test Pandas set_N() function

import pytest
from pandas import read_csv

from causaliq_data import Pandas
from tests.conftest import TESTDATA_DIR

# Type errors


# set_N requires N argument
def test_set_N_type_error_1():
    """Asia, N=100 - no args."""
    df = read_csv(
        TESTDATA_DIR + "/experiments/datasets/asia.data.gz",
        dtype="category",
        nrows=100,
    )

    data = Pandas(df=df)

    with pytest.raises(TypeError):
        data.set_N()


# set_N requires integer N argument
def test_set_N_type_error_2():
    """Asia, N=100 - non-integer arg."""
    df = read_csv(
        TESTDATA_DIR + "/experiments/datasets/asia.data.gz",
        dtype="category",
        nrows=100,
    )

    data = Pandas(df=df)

    with pytest.raises(TypeError):
        data.set_N(2.1)
    with pytest.raises(TypeError):
        data.set_N(True)
    with pytest.raises(TypeError):
        data.set_N([2])


# set_N seed parameter must be integer if provided
def test_set_N_type_error_3():
    """Asia, N=100 - invalid seed type."""
    df = read_csv(
        TESTDATA_DIR + "/experiments/datasets/asia.data.gz",
        dtype="category",
        nrows=100,
    )

    data = Pandas(df=df)

    with pytest.raises(TypeError):
        data.set_N(N=10, seed=True)
    with pytest.raises(TypeError):
        data.set_N(N=10, seed=[1])
    with pytest.raises(TypeError):
        data.set_N(N=10, seed=2.1)


# set_N random_selection parameter must be boolean
def test_set_N_type_error_4():
    """Asia, N=100 - invalid random_selection type."""
    df = read_csv(
        TESTDATA_DIR + "/experiments/datasets/asia.data.gz",
        dtype="category",
        nrows=100,
    )

    data = Pandas(df=df)

    with pytest.raises(TypeError):
        data.set_N(N=10, random_selection=1)
    with pytest.raises(TypeError):
        data.set_N(N=10, random_selection=1)
    with pytest.raises(TypeError):
        data.set_N(N=10, random_selection={True})


# Value errors


# set_N requires positive N value
def test_set_N_value_error_1():
    """Asia, N=100 - set non-positive N."""
    df = read_csv(
        TESTDATA_DIR + "/experiments/datasets/asia.data.gz",
        dtype="category",
        nrows=100,
    )

    data = Pandas(df=df)

    with pytest.raises(ValueError):
        data.set_N(0)
    with pytest.raises(ValueError):
        data.set_N(-3)


# set_N cannot exceed available data rows
def test_set_N_value_error_2():
    """Asia, N=100 - larger than amount of data."""
    df = read_csv(
        TESTDATA_DIR + "/experiments/datasets/asia.data.gz",
        dtype="category",
        nrows=100,
    )

    data = Pandas(df=df)

    with pytest.raises(ValueError):
        data.set_N(101)


# set_N seed must be in valid range (0-100)
def test_set_N_value_error_3():
    """Asia, N=100 - invalid seed values."""
    df = read_csv(
        TESTDATA_DIR + "/experiments/datasets/asia.data.gz",
        dtype="category",
        nrows=100,
    )

    data = Pandas(df=df)

    with pytest.raises(ValueError):
        data.set_N(80, seed=-1)
    with pytest.raises(ValueError):
        data.set_N(80, seed=101)


# set_N random_selection is not supported (reserved for future)
def test_set_N_value_error_4():
    """Asia, N=100 - random_selection unsupported."""
    df = read_csv(
        TESTDATA_DIR + "/experiments/datasets/asia.data.gz",
        dtype="category",
        nrows=100,
    )

    data = Pandas(df=df)

    with pytest.raises(ValueError):
        data.set_N(80, random_selection=True)


# Successful operations


# Verify set_N adjusts sample size and recounts node values correctly
def test_set_N_abc5_1_ok():
    """ABC, 5 discrete rows, randomising order."""
    df = read_csv(TESTDATA_DIR + "/simple/abc_5.csv", dtype="category")
    data = Pandas(df=df)
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
        "A": ["1", "1", "0", "0"],
        "B": ["1", "0", "0", "1"],
        "C": ["1", "0", "0", "0"],
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
        "B": ["0", "1", "1", "0"],
        "C": ["0", "1", "0", "0"],
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


# Verify set_N works with continuous data and row randomization
def test_set_N_xyz10_1_ok():
    """XYZ, 10 continuous rows, randomising order."""
    df = read_csv(TESTDATA_DIR + "/simple/xyz_10.csv", dtype="float32")
    data = Pandas(df=df)

    print("\n\nOriginal Dataset:\n{}\n".format(data.as_df()))
    rdf = data.as_df().map(lambda x: round(x, 1))
    assert rdf.to_dict(orient="list") == {
        "X": [1.1, 0.0, 0.2, 4.4, 0.6, 4.0, 2.2, 0.1, 7.1, 6.0],
        "Y": [0.3, 3.1, 5.4, 6.6, 2.8, 6.0, 3.1, 0.0, 3.9, 0.2],
        "Z": [0.3, 4.0, 1.7, 1.9, 9.9, 9.0, 0.8, 2.2, 1.4, 0.5],
    }

    data.set_N(6)
    print("\n\nSetting N=6, no seed:\n{}\n".format(data.as_df()))
    rdf = data.as_df().map(lambda x: round(x, 1))
    assert rdf.to_dict(orient="list") == {
        "X": [1.1, 0.0, 0.2, 4.4, 0.6, 4.0],
        "Y": [0.3, 3.1, 5.4, 6.6, 2.8, 6.0],
        "Z": [0.3, 4.0, 1.7, 1.9, 9.9, 9.0],
    }

    data.set_N(4, seed=3)
    print("\n\nSetting N=4, seed=3:\n{}\n".format(data.as_df()))
    rdf = data.as_df().map(lambda x: round(x, 1))
    assert rdf.to_dict(orient="list") == {
        "X": [4.4, 0.0, 1.1, 0.2],
        "Y": [6.6, 3.1, 0.3, 5.4],
        "Z": [1.9, 4.0, 0.3, 1.7],
    }

    data.set_N(10)
    print("\n\nSetting N=10, no seed:\n{}\n".format(data.as_df()))
    rdf = data.as_df().map(lambda x: round(x, 1))
    assert rdf.to_dict(orient="list") == {
        "X": [1.1, 0.0, 0.2, 4.4, 0.6, 4.0, 2.2, 0.1, 7.1, 6.0],
        "Y": [0.3, 3.1, 5.4, 6.6, 2.8, 6.0, 3.1, 0.0, 3.9, 0.2],
        "Z": [0.3, 4.0, 1.7, 1.9, 9.9, 9.0, 0.8, 2.2, 1.4, 0.5],
    }

    data.set_N(4, seed=3)
    print("\n\nSetting N=4, seed=3:\n{}\n".format(data.as_df()))
    rdf = data.as_df().map(lambda x: round(x, 1))
    assert rdf.to_dict(orient="list") == {
        "X": [4.4, 0.0, 1.1, 0.2],
        "Y": [6.6, 3.1, 0.3, 5.4],
        "Z": [1.9, 4.0, 0.3, 1.7],
    }


# Verify set_N can decrease and increase sample size on Asia dataset
def test_set_N_asia_1_ok():
    """Asia, N=100 - set N to 50."""
    df = read_csv(
        TESTDATA_DIR + "/experiments/datasets/asia.data.gz",
        dtype="category",
        nrows=100,
    )

    data = Pandas(df=df)
    data.set_N(50)

    assert isinstance(data, Pandas)
    assert (data.df == df).all().all()
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
    assert len(data.sample) == 50
    assert (data.sample == df[:50]).all().all()

    # Note can increase sample size too

    data.set_N(80)

    assert isinstance(data, Pandas)
    assert (data.df == df).all().all()
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
    assert data.N == 80
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
    assert len(data.sample) == 80
    assert (data.sample == df[:80]).all().all()

    # Can increase size back up to original size

    data.set_N(100)

    assert isinstance(data, Pandas)
    assert (data.df == df).all().all()
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
    assert data.N == 100
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
        "asia": {"no": 97, "yes": 3},
        "bronc": {"no": 56, "yes": 44},
        "dysp": {"no": 56, "yes": 44},
        "either": {"no": 92, "yes": 8},
        "lung": {"no": 93, "yes": 7},
        "smoke": {"no": 57, "yes": 43},
        "tub": {"no": 99, "yes": 1},
        "xray": {"no": 90, "yes": 10},
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
    assert len(data.sample) == 100
    assert (data.sample == df).all().all()

    # Check first five rows

    print(data.sample.head())
    assert data.sample[:5].to_dict() == {
        "asia": {0: "no", 1: "no", 2: "no", 3: "no", 4: "no"},
        "bronc": {0: "no", 1: "yes", 2: "yes", 3: "no", 4: "yes"},
        "dysp": {0: "no", 1: "yes", 2: "yes", 3: "no", 4: "yes"},
        "either": {0: "no", 1: "no", 2: "no", 3: "yes", 4: "no"},
        "lung": {0: "no", 1: "no", 2: "no", 3: "no", 4: "no"},
        "smoke": {0: "yes", 1: "yes", 2: "yes", 3: "no", 4: "no"},
        "tub": {0: "no", 1: "no", 2: "no", 3: "yes", 4: "no"},
        "xray": {0: "no", 1: "no", 2: "no", 3: "yes", 4: "no"},
    }


# Verify set_N with seed randomizes row order and reverts without seed
def test_set_N_asia_2_ok():
    """Asia, N=100 - set N to 50, randomise rows."""
    df = read_csv(
        TESTDATA_DIR + "/experiments/datasets/asia.data.gz",
        dtype="category",
        nrows=100,
    )

    data = Pandas(df=df)
    print("\nOriginal 5/100 randomised rows:\n{}".format(data.as_df().head()))
    assert data.as_df()[:5].to_dict(orient="list") == {
        "asia": ["no", "no", "no", "no", "no"],
        "bronc": ["no", "yes", "yes", "no", "yes"],
        "dysp": ["no", "yes", "yes", "no", "yes"],
        "either": ["no", "no", "no", "yes", "no"],
        "lung": ["no", "no", "no", "no", "no"],
        "smoke": ["yes", "yes", "yes", "no", "no"],
        "tub": ["no", "no", "no", "yes", "no"],
        "xray": ["no", "no", "no", "yes", "no"],
    }

    data.set_N(50, seed=1)

    assert isinstance(data, Pandas)
    assert (data.df == df).all().all()
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

    print("\n1st 5/50 randomised rows:\n{}".format(data.as_df().head()))

    assert data.sample[:5].to_dict(orient="list") == {
        "asia": ["no", "no", "yes", "no", "no"],
        "bronc": ["no", "no", "no", "no", "yes"],
        "dysp": ["no", "no", "no", "no", "yes"],
        "either": ["no", "no", "no", "no", "no"],
        "lung": ["no", "no", "no", "no", "no"],
        "smoke": ["yes", "no", "no", "no", "yes"],
        "tub": ["no", "no", "no", "no", "no"],
        "xray": ["no", "no", "no", "no", "no"],
    }

    # Note can increase sample size too

    data.set_N(80, seed=2)

    assert isinstance(data, Pandas)
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
    assert data.N == 80
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
        "asia": {"no": 77, "yes": 3},
        "bronc": {"no": 45, "yes": 35},
        "dysp": {"no": 43, "yes": 37},
        "either": {"no": 74, "yes": 6},
        "lung": {"no": 75, "yes": 5},
        "smoke": {"no": 43, "yes": 37},
        "tub": {"no": 79, "yes": 1},
        "xray": {"no": 72, "yes": 8},
    }

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

    print("\n1st 5/80 randomised rows:\n{}".format(data.sample.head()))
    assert data.as_df()[:5].to_dict(orient="list") == {
        "asia": ["no", "no", "no", "no", "no"],
        "bronc": ["no", "yes", "no", "yes", "yes"],
        "dysp": ["no", "yes", "no", "no", "yes"],
        "either": ["no", "no", "no", "no", "no"],
        "lung": ["no", "no", "no", "no", "no"],
        "smoke": ["yes", "yes", "no", "yes", "yes"],
        "tub": ["no", "no", "no", "no", "no"],
        "xray": ["no", "no", "no", "yes", "no"],
    }

    # Can increase size back up to original size, and without seed reverts
    # to original order.

    data.set_N(100)

    assert isinstance(data, Pandas)
    assert (data.df == df).all().all()
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
    assert data.N == 100
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
        "asia": {"no": 97, "yes": 3},
        "bronc": {"no": 56, "yes": 44},
        "dysp": {"no": 56, "yes": 44},
        "either": {"no": 92, "yes": 8},
        "lung": {"no": 93, "yes": 7},
        "smoke": {"no": 57, "yes": 43},
        "tub": {"no": 99, "yes": 1},
        "xray": {"no": 90, "yes": 10},
    }

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
    assert len(data.sample) == 100

    # Check first five rows

    print("\n1st 5/100 randomised rows:\n{}".format(data.sample.head()))
    assert data.as_df()[:5].to_dict(orient="list") == {
        "asia": ["no", "no", "no", "no", "no"],
        "bronc": ["no", "yes", "yes", "no", "yes"],
        "dysp": ["no", "yes", "yes", "no", "yes"],
        "either": ["no", "no", "no", "yes", "no"],
        "lung": ["no", "no", "no", "no", "no"],
        "smoke": ["yes", "yes", "yes", "no", "no"],
        "tub": ["no", "no", "no", "yes", "no"],
        "xray": ["no", "no", "no", "yes", "no"],
    }
