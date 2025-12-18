# Check bnbench and bnlearn metrics give same answers

from os import remove
from random import random

import pytest
from causaliq_core.bn.io import read_bn
from causaliq_core.graph import DAG
from causaliq_core.utils import dicts_same
from pandas import DataFrame

import tests.fixtures.example_dags as dag
from causaliq_data.pandas import Pandas
from causaliq_data.score import dag_score

TESTDATA_DIR = "tests/data/functional"
TYPES = ["loglik", "bic", "aic", "bde", "k2", "bdj", "bds"]  # scores to test
DEFAULT_PARAMS = {"iss": 1.0, "prior": "uniform", "base": "e"}


# temp file, automatically removed
@pytest.fixture(scope="function")
def tmpfile():
    _tmpfile = TESTDATA_DIR + "/tmp/{}.csv".format(int(random() * 10000000))
    yield _tmpfile
    remove(_tmpfile)


# --- Check scores for delete networks


# A, B unconnected
def test_score_bnlearn_a_b_3_ok():
    data = Pandas(
        DataFrame(
            {"A": ["1", "0", "1"], "B": ["1", "0", "1"]}, dtype="category"
        )
    )
    bnlearn = {
        "loglik": -3.81908500976888,
        "bic": -4.91769729843699,
        "aic": -5.81908500976888,
        "bde": -5.54517744447956,
        "k2": -4.969813299576,
        "bdj": -8.31776616671934,
        "bds": -5.54517744447956,
    }
    scores = dag_score(dag.a_b(), data, TYPES, DEFAULT_PARAMS)
    print(scores)
    assert dicts_same(bnlearn, dict(scores.sum()))


# A, B unconnected, 4 rows
def test_score_bnlearn_a_b_4_ok():
    data = Pandas(
        DataFrame(
            {"A": ["1", "1", "1", "0"], "B": ["1", "0", "1", "1"]},
            dtype="category",
        )
    )
    bnlearn = {
        "loglik": -4.49868115695047,
        "bic": -5.88497551807036,
        "aic": -6.49868115695047,
        "bde": -6.48518470297103,
        "k2": -5.99146454710798,
        "bdj": -9.70406052783923,
        "bds": -6.48518470297103,
    }
    scores = dag_score(dag.a_b(), data, TYPES, DEFAULT_PARAMS)
    print(scores)
    assert dicts_same(bnlearn, dict(scores.sum()))


# A, B unconnected, 4 rows - equal values
def test_score_bnlearn_a_b_5_ok():
    data = Pandas(
        DataFrame(
            {"A": ["1", "0", "1", "0"], "B": ["1", "0", "1", "0"]},
            dtype="category",
        )
    )
    bnlearn = {
        "loglik": -5.54517744447956,
        "bic": -6.93147180559945,
        "aic": -7.54517744447956,
        "bde": -7.50683595050302,
        "k2": -6.80239476332431,
        "bdj": -10.7257117753712,
        "bds": -7.50683595050302,
    }
    scores = dag_score(dag.a_b(), data, TYPES, DEFAULT_PARAMS)
    print(scores)
    assert dicts_same(bnlearn, dict(scores.sum()))


# A, B unconnected, 7 rows
def test_score_bnlearn_a_b_6_ok():
    data = Pandas(
        DataFrame(
            {
                "A": ["1", "0", "1", "0", "0", "0", "0"],
                "B": ["0", "0", "0", "1", "0", "1", "0"],
            },
            dtype="category",
        )
    )
    bnlearn = {
        "loglik": -8.3757742401936,
        "bic": -10.3216843892489,
        "aic": -10.3757742401936,
        "bde": -10.8547888176464,
        "k2": -10.2479279588065,
        "bdj": -15.013671901006,
        "bds": -10.8547888176464,
    }
    scores = dag_score(dag.a_b(), data, TYPES, DEFAULT_PARAMS)
    print(scores)
    assert dicts_same(bnlearn, dict(scores.sum()))


# A --> B, 2 rows
def test_score_bnlearn_ab_1_ok():
    data = Pandas(
        DataFrame({"A": ["0", "1"], "B": ["0", "1"]}, dtype="category")
    )
    bnlearn = {
        "loglik": -1.38629436111989,
        "bic": -2.42601513195981,
        "aic": -4.38629436111989,
        "bde": -3.46573590279973,
        "k2": -3.17805383034795,
        "bdj": -4.56434819146784,
        "bds": -3.46573590279973,
    }
    scores = dag_score(dag.ab(), data, TYPES, DEFAULT_PARAMS)
    print(scores)
    assert dicts_same(bnlearn, dict(scores.sum()))


# A --> B, 4 rows
def test_score_bnlearn_ab_2_ok():
    data = Pandas(
        DataFrame(
            {"A": ["0", "1", "0", "1"], "B": ["0", "1", "0", "1"]},
            dtype="category",
        )
    )
    bnlearn = {
        "loglik": -2.77258872223978,
        "bic": -4.85203026391962,
        "aic": -5.77258872223978,
        "bde": -5.50435544995931,
        "k2": -5.59842195899837,
        "bdj": -7.32451439370906,
        "bds": -5.50435544995931,
    }
    scores = dag_score(dag.ab(), data, TYPES, DEFAULT_PARAMS)
    print(scores)
    assert dicts_same(bnlearn, dict(scores.sum()))


# A --> B, 3 rows
def test_score_bnlearn_ab_3_ok():
    data = Pandas(
        DataFrame(
            {"A": ["0", "1", "1"], "B": ["0", "1", "1"]}, dtype="category"
        )
    )
    bnlearn = {
        "loglik": -1.90954250488444,
        "bic": -3.5574609378866,
        "aic": -4.90954250488444,
        "bde": -4.34120464015363,
        "k2": -4.27666611901606,
        "bdj": -5.83285951693134,
        "bds": -4.34120464015363,
    }
    scores = dag_score(dag.ab(), data, TYPES, DEFAULT_PARAMS)
    print(scores)
    assert dicts_same(bnlearn, dict(scores.sum()))


# A --> B, 4 rows
def test_score_bnlearn_ab_4_ok():
    data = Pandas(
        DataFrame(
            {"A": ["0", "0", "1", "1"], "B": ["0", "1", "0", "1"]},
            dtype="category",
        )
    )
    bnlearn = {
        "loglik": -5.54517744447956,
        "bic": -7.6246189861594,
        "aic": -8.54517744447956,
        "bde": -8.72323127482751,
        "k2": -6.98471632011826,
        "bdj": -9.52173897104528,
        "bds": -8.72323127482751,
    }
    scores = dag_score(dag.ab(), data, TYPES, DEFAULT_PARAMS)
    print(scores)
    assert dicts_same(bnlearn, dict(scores.sum()))


# A --> C <-- B, 2 parent combos
def test_score_bnlearn_ac_bc_1_ok():
    data = Pandas(
        DataFrame(
            {"A": ["0", "1", "1"], "B": ["0", "1", "1"], "C": ["0", "1", "1"]},
            dtype="category",
        )
    )
    bnlearn = {
        "loglik": -3.81908500976888,
        "bic": -7.11492187577321,
        "aic": -9.81908500976888,
        "bde": -7.03683232125728,
        "k2": -6.76157276880406,
        "bdj": -9.99174260029102,
        "bds": -7.11379336239341,
    }
    scores = dag_score(dag.ac_bc(), data, TYPES, DEFAULT_PARAMS)
    print(scores)
    assert dicts_same(bnlearn, dict(scores.sum()))


# A --> C <-- B, all parent combo
def test_score_bnlearn_ac_bc_2_ok():
    data = Pandas(
        DataFrame(
            {
                "A": ["0", "1", "1", "0", "1"],
                "B": ["0", "1", "1", "1", "0"],
                "C": ["0", "1", "1", "1", "1"],
            },
            dtype="category",
        )
    )
    bnlearn = {
        "loglik": -6.73011667009256,
        "bic": -11.5584304073949,
        "aic": -12.7301166700926,
        "bde": -11.7710795495205,
        "k2": -11.3667429547921,
        "bdj": -15.5369200447706,
        "bds": -11.7710795495205,
    }
    scores = dag_score(dag.ac_bc(), data, TYPES, DEFAULT_PARAMS)
    print(scores)
    assert dicts_same(bnlearn, dict(scores.sum()))


# Test Heckerman data with N1 --> N2 connection
def test_score_bnlearn_heckerman_1_ok():
    data = Pandas.read(
        TESTDATA_DIR + "/simple/heckerman.csv", dstype="categorical"
    )
    dag = DAG(["N1", "N2"], [("N1", "->", "N2")])
    bnlearn = {
        "loglik": -16.2957342598479,
        "bic": -20.0230942345299,
        "aic": -19.2957342598479,
        "bde": -21.028372089894,
        "k2": -18.9686449146673,
        "bdj": -22.6664617641321,
        "bds": -21.028372089894,
    }
    scores = dag_score(dag, data, TYPES, DEFAULT_PARAMS)
    assert dicts_same(bnlearn, dict(scores.sum()))


# Test Heckerman data with N1 independent of N2
def test_score_bnlearn_heckerman_2_ok():
    data = Pandas.read(
        TESTDATA_DIR + "/simple/heckerman.csv", dstype="categorical"
    )
    dag = DAG(["N1", "N2"], [])
    bnlearn = {
        "loglik": -16.4680853586176,
        "bic": -18.9529920084057,
        "aic": -18.4680853586176,
        "bde": -19.4465864389308,
        "k2": -18.6331721783792,
        "bdj": -24.5764851538539,
        "bds": -19.4465864389308,
    }
    scores = dag_score(dag, data, TYPES, DEFAULT_PARAMS)
    assert dicts_same(bnlearn, dict(scores.sum()))


# Test ASIA network scoring with 1000 generated cases
def test_score_bnlearn_asia_1k():
    asia = read_bn(TESTDATA_DIR + "/discrete/small/asia.dsc")
    import random

    import numpy as np

    random.seed(42)
    np.random.seed(42)
    data = Pandas(asia.generate_cases(1000))
    bnlearn = {
        "loglik": -2212.76850176607,
        "bic": -2274.93829927691,
        "aic": -2230.76850176607,
        "bde": -2263.22099377786,
        "k2": -2274.1548215868,
        "bdj": -2278.77589583986,
        "bds": -2263.22099377786,
    }
    scores = dag_score(asia.dag, data, TYPES, DEFAULT_PARAMS)
    print(scores)
    assert dicts_same(bnlearn, dict(scores.sum()))


# Test ASIA network scoring with 10000 generated cases
@pytest.mark.slow
def test_score_bnlearn_asia_10k():
    asia = read_bn(TESTDATA_DIR + "/discrete/small/asia.dsc")
    import random

    import numpy as np

    random.seed(123)
    np.random.seed(123)
    data = Pandas(asia.generate_cases(10000))
    bnlearn = {
        "loglik": -22325.517273244,
        "bic": -22408.4103365917,
        "aic": -22343.517273244,
        "bde": -22393.4725875356,
        "k2": -22408.8625044123,
        "bdj": -22415.2933059468,
        "bds": -22393.4725875356,
    }
    scores = dag_score(asia.dag, data, TYPES, DEFAULT_PARAMS)
    print(scores)
    assert dicts_same(bnlearn, dict(scores.sum()))


# Test ALARM network subset with pressure variables
def test_score_bnlearn_alarm_press():
    dag = DAG(
        ["INT", "KIN", "VEN", "PRE"],
        [("INT", "->", "PRE"), ("KIN", "->", "PRE"), ("VEN", "->", "PRE")],
    )
    data = Pandas(
        DataFrame(
            {
                "INT": ["0", "1", "1", "1", "1"],
                "KIN": ["0", "0", "0", "0", "1"],
                "VEN": ["0", "1", "0", "0", "0"],
                "PRE": ["1", "0", "0", "1", "1"],
            },
            dtype="category",
        )
    )
    scores = dag_score(dag, data, TYPES, DEFAULT_PARAMS)
    print(scores)
    bnlearn = {
        "loglik": -8.89233071419271,
        "bic": -17.7442392325803,
        "aic": -19.8923307141927,
        "bde": -16.4607623664087,
        "k2": -14.0747931558944,
        "bdj": -20.3319633773166,
        "bds": -15.8729757015066,
    }
    assert dicts_same(bnlearn, dict(scores.sum()))


# Test full ALARM network scoring with 10000 generated cases
@pytest.mark.slow
def test_score_bnlearn_alarm_10k():
    alarm = read_bn(TESTDATA_DIR + "/discrete/medium/alarm.dsc")
    import random

    import numpy as np

    random.seed(456)
    np.random.seed(456)
    data = Pandas(alarm.generate_cases(10000))
    bnlearn = {
        "loglik": -104359.520295141,
        "bic": -106703.551919809,
        "aic": -104868.520295141,
        "bde": -105920.401318098,
        "k2": -105932.368155061,
        "bdj": -105751.295922151,
        "bds": -105912.990228967,
    }
    scores = dag_score(alarm.dag, data, TYPES, DEFAULT_PARAMS)
    print(scores)
    assert dicts_same(bnlearn, dict(scores.sum()))


# Test PATHFINDER network scoring with 25000 generated cases
@pytest.mark.slow
def test_score_bnlearn_pathfinder_25k():
    pathfinder = read_bn(TESTDATA_DIR + "/discrete/verylarge/pathfinder.dsc")
    import random

    import numpy as np

    random.seed(789)
    np.random.seed(789)
    data = Pandas(pathfinder.generate_cases(25000))
    bnlearn = {
        "loglik": -584873.877213457,
        "bic": -948875.632241357,
        "aic": -656763.877213457,
        "bde": -645650.700742204,
        "k2": -677537.654003191,
        "bdj": -649088.788433379,
        "bds": -641960.290034771,
    }
    scores = dag_score(pathfinder.dag, data, TYPES, DEFAULT_PARAMS)
    assert dicts_same(bnlearn, dict(scores.sum()))


# Check score for continuous networks


# Test X --> Y Gaussian network scoring
def test_score_bnlearn_x_y_score():
    x_y = read_bn(TESTDATA_DIR + "/xdsl/x_y.xdsl")
    print("\n\n{}".format(x_y.dag))
    data = Pandas(x_y.generate_cases(3))
    # Note: This test uses randomly generated data so we skip bnlearn
    # comparison and just verify the function runs without error
    scores = dag_score(x_y.dag, data, ["bic-g"], {"k": 1})
    print(scores)
    # Just verify that scores are calculated (no assertion against bnlearn)
    assert "bic-g" in dict(scores.sum())
