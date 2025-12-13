# Test Pandas read() method - successful operations

from causaliq_core.utils import dicts_same

from causaliq_data import Pandas
from tests.conftest import TESTDATA_DIR


# read() correctly loads simple categorical dataset
def test_pandas_read_ab_1_ok():
    """Reads AB csv file."""
    data = Pandas.read(TESTDATA_DIR + "/simple/ab_3.csv")
    assert isinstance(data, Pandas)
    print("\nPandas is:\n{}\n".format(data.sample))
    assert data.N == 3
    assert data.nodes == ("A", "B")
    assert data.dstype == "categorical"
    assert data.node_types == {"A": "category", "B": "category"}
    assert data.node_values == {
        "A": {"1": 2, "0": 1},
        "B": {"0": 2, "1": 1},
    }


# read() respects explicit categorical dstype
def test_pandas_read_ab_2_ok():
    """Reads AB csv file, force categorical."""
    data = Pandas.read(TESTDATA_DIR + "/simple/ab_3.csv", dstype="categorical")
    assert isinstance(data, Pandas)
    print("\nPandas is:\n{}\n".format(data.sample))
    assert data.N == 3
    assert data.nodes == ("A", "B")
    assert data.dstype == "categorical"
    assert data.node_types == {"A": "category", "B": "category"}
    assert data.node_values == {
        "A": {"1": 2, "0": 1},
        "B": {"0": 2, "1": 1},
    }


# read() can force categorical data to continuous
def test_pandas_read_ab_3_ok():
    """Reads AB csv file, force to floats."""
    data = Pandas.read(TESTDATA_DIR + "/simple/ab_3.csv", dstype="continuous")
    assert isinstance(data, Pandas)
    print("\nPandas is:\n{}\n".format(data.sample))
    assert data.N == 3
    assert data.nodes == ("A", "B")
    assert data.dstype == "continuous"
    assert data.node_types == {"A": "float32", "B": "float32"}
    assert data.node_values == {}
    assert dicts_same(
        data.sample["A"].to_dict(), {0: 1.0, 1: 1.0, 2: 0.0}, sf=3
    )
    assert dicts_same(
        data.sample["B"].to_dict(), {0: 1.0, 1: 0.0, 2: 0.0}, sf=3
    )


# read() auto-detects continuous variables
def test_pandas_read_xy_1_ok():
    """Reads a XY csv file - detect var types."""
    data = Pandas.read(TESTDATA_DIR + "/simple/xy_3.csv")
    assert isinstance(data, Pandas)
    print("\nPandas is:\n{}\n".format(data.sample))
    assert data.N == 3
    assert data.nodes == ("F1", "F2")
    assert data.node_values == {}
    assert data.dstype == "continuous"
    assert data.node_types == {"F1": "float32", "F2": "float32"}
    assert dicts_same(
        data.sample["F1"].to_dict(), {0: 1.01, 1: -0.45, 2: 1.22}, sf=3
    )
    assert dicts_same(
        data.sample["F2"].to_dict(), {0: 1.21, 1: 0.67, 2: -1.41}, sf=3
    )


# read() respects explicit continuous dstype
def test_pandas_read_xy_2_ok():
    """Reads a XY csv file, specify as continuous."""
    data = Pandas.read((TESTDATA_DIR + "/simple/xy_3.csv"), "continuous")
    assert isinstance(data, Pandas)
    print("\nPandas is:\n{}\n".format(data.sample))
    assert data.N == 3
    assert data.nodes == ("F1", "F2")
    assert data.dstype == "continuous"
    assert data.node_values == {}
    assert data.node_types == {"F1": "float32", "F2": "float32"}
    assert dicts_same(
        data.sample["F1"].to_dict(), {0: 1.01, 1: -0.45, 2: 1.22}, sf=3
    )
    assert dicts_same(
        data.sample["F2"].to_dict(), {0: 1.21, 1: 0.67, 2: -1.41}, sf=3
    )


# read() can force continuous data to categorical
def test_pandas_read_xy_3_ok():
    """Reads a XY csv file, force to categorical."""
    data = Pandas.read((TESTDATA_DIR + "/simple/xy_3.csv"), "categorical")
    assert isinstance(data, Pandas)
    print("\nPandas is:\n{}\n".format(data.sample))
    assert data.N == 3
    assert data.nodes == ("F1", "F2")
    assert data.node_values == {
        "F1": {"-0.45": 1, "1.01": 1, "1.22": 1},
        "F2": {"-1.41": 1, "0.67": 1, "1.21": 1},
    }
    assert data.dstype == "categorical"
    assert data.node_types == {"F1": "category", "F2": "category"}
    assert data.sample["F1"].to_dict() == {0: "1.01", 1: "-0.45", 2: "1.22"}
    assert data.sample["F2"].to_dict() == {0: "1.21", 1: "0.67", 2: "-1.41"}


# read() handles gzipped categorical data files
def test_pandas_read_cancer_1_ok():
    """Reads a gzipped Cancer file OK."""
    data = Pandas.read(TESTDATA_DIR + "/experiments/datasets/cancer.data.gz")
    assert isinstance(data, Pandas)
    assert data.N == 1000
    assert data.nodes == ("Cancer", "Dyspnoea", "Pollution", "Smoker", "Xray")
    assert data.node_values == {
        "Cancer": {"False": 986, "True": 14},
        "Dyspnoea": {"False": 700, "True": 300},
        "Pollution": {"low": 897, "high": 103},
        "Smoker": {"False": 693, "True": 307},
        "Xray": {"negative": 805, "positive": 195},
    }
    assert all([t == "category" for t in data.node_types.values()])
    assert data.dstype == "categorical"

    print("\nMemory usage is:")
    for n in data.nodes:
        print(
            "  {} needs {} bytes".format(
                n, data.sample[n].memory_usage(index=False, deep=True)
            )
        )


# read() respects N parameter to limit rows read
def test_pandas_read_cancer_2_ok():
    """Reads a gzipped Cancer file, N=10."""
    data = Pandas.read(
        TESTDATA_DIR + "/experiments/datasets/cancer.data.gz", N=10
    )
    assert isinstance(data, Pandas)
    assert data.N == 10
    assert data.nodes == ("Cancer", "Dyspnoea", "Pollution", "Smoker", "Xray")
    assert data.node_values == {
        "Cancer": {"False": 10},
        "Dyspnoea": {"False": 5, "True": 5},
        "Pollution": {"low": 7, "high": 3},
        "Smoker": {"False": 7, "True": 3},
        "Xray": {"negative": 8, "positive": 2},
    }
    assert all([t == "category" for t in data.node_types.values()])
    assert data.dstype == "categorical"


# read() auto-detects mixed datasets
def test_pandas_read_mix_1_ok():
    """Reads a mixed file - detect var types."""
    data = Pandas.read(TESTDATA_DIR + "/simple/mix_2.csv")
    assert isinstance(data, Pandas)
    print("\nPandas is:\n{}\n".format(data.sample))
    assert data.N == 2
    assert data.nodes == ("C", "I", "F")
    assert data.node_values == {
        "C": {"A": 1, "B": 1},
        "I": {"-1": 1, "3": 1},
    }
    assert data.dstype == "mixed"
    assert data.node_types == {
        "C": "category",
        "I": "category",
        "F": "float32",
    }


# read() can force mixed data to categorical
def test_pandas_read_mix_2_ok():
    """Reads a mixed file - force to category."""
    data = Pandas.read(
        TESTDATA_DIR + "/simple/mix_2.csv", dstype="categorical"
    )
    assert isinstance(data, Pandas)
    print("\nPandas is:\n{}\n".format(data.sample))
    assert data.N == 2
    assert data.nodes == ("C", "I", "F")
    assert data.node_values == {
        "C": {"A": 1, "B": 1},
        "I": {"-1": 1, "3": 1},
        "F": {"2.7": 1, "-0.3": 1},
    }
    assert data.dstype == "categorical"
    assert data.node_types == {
        "C": "category",
        "I": "category",
        "F": "category",
    }


# read() handles large gzipped continuous datasets
def test_pandas_read_gauss_1_ok():
    """Reads gzipped Gaussian test data file."""
    data = Pandas.read(TESTDATA_DIR + "/simple/gauss.data.gz")
    assert isinstance(data, Pandas)
    assert data.N == 5000
    assert data.nodes == ("A", "B", "C", "D", "E", "F", "G")
    assert data.node_values == {}
    assert all([t == "float32" for t in data.node_types.values()])
    assert data.dstype == "continuous"

    print("\nMemory usage is:")
    for n in data.nodes:
        print(
            "  {} needs {} bytes".format(
                n, data.sample[n].memory_usage(index=False, deep=True)
            )
        )


# read() handles Asia benchmark dataset
def test_pandas_read_asia_1_ok():
    """Reads a gzipped Asia data file."""
    data = Pandas.read(TESTDATA_DIR + "/experiments/datasets/asia.data.gz")
    assert isinstance(data, Pandas)
    assert data.N == 1000
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
    assert data.node_values == {
        "asia": {"no": 990, "yes": 10},
        "bronc": {"no": 576, "yes": 424},
        "dysp": {"no": 586, "yes": 414},
        "either": {"no": 930, "yes": 70},
        "lung": {"no": 943, "yes": 57},
        "smoke": {"no": 503, "yes": 497},
        "tub": {"no": 986, "yes": 14},
        "xray": {"no": 889, "yes": 111},
    }
    assert all([t == "category" for t in data.node_types.values()])
    assert data.dstype == "categorical"


# read() handles Alarm benchmark dataset
def test_pandas_read_alarm_1_ok():
    """Reads a gzipped Alarm data file."""
    data = Pandas.read(TESTDATA_DIR + "/experiments/datasets/alarm.data.gz")
    assert isinstance(data, Pandas)
    assert data.N == 1000
    assert data.nodes == (
        "ANAPHYLAXIS",
        "ARTCO2",
        "BP",
        "CATECHOL",
        "CO",
        "CVP",
        "DISCONNECT",
        "ERRCAUTER",
        "ERRLOWOUTPUT",
        "EXPCO2",
        "FIO2",
        "HISTORY",
        "HR",
        "HRBP",
        "HREKG",
        "HRSAT",
        "HYPOVOLEMIA",
        "INSUFFANESTH",
        "INTUBATION",
        "KINKEDTUBE",
        "LVEDVOLUME",
        "LVFAILURE",
        "MINVOL",
        "MINVOLSET",
        "PAP",
        "PCWP",
        "PRESS",
        "PULMEMBOLUS",
        "PVSAT",
        "SAO2",
        "SHUNT",
        "STROKEVOLUME",
        "TPR",
        "VENTALV",
        "VENTLUNG",
        "VENTMACH",
        "VENTTUBE",
    )
    assert data.node_values == {
        "ANAPHYLAXIS": {"FALSE": 990, "TRUE": 10},
        "ARTCO2": {"HIGH": 746, "LOW": 195, "NORMAL": 59},
        "BP": {"HIGH": 411, "LOW": 382, "NORMAL": 207},
        "CATECHOL": {"HIGH": 903, "NORMAL": 97},
        "CO": {"HIGH": 637, "NORMAL": 186, "LOW": 177},
        "CVP": {"NORMAL": 731, "HIGH": 160, "LOW": 109},
        "DISCONNECT": {"FALSE": 898, "TRUE": 102},
        "ERRCAUTER": {"FALSE": 903, "TRUE": 97},
        "ERRLOWOUTPUT": {"FALSE": 948, "TRUE": 52},
        "EXPCO2": {"LOW": 863, "NORMAL": 55, "HIGH": 42, "ZERO": 40},
        "FIO2": {"NORMAL": 954, "LOW": 46},
        "HISTORY": {"FALSE": 943, "TRUE": 57},
        "HR": {"HIGH": 808, "NORMAL": 173, "LOW": 19},
        "HRBP": {"HIGH": 753, "LOW": 180, "NORMAL": 67},
        "HREKG": {"HIGH": 729, "LOW": 174, "NORMAL": 97},
        "HRSAT": {"HIGH": 732, "LOW": 171, "NORMAL": 97},
        "HYPOVOLEMIA": {"FALSE": 796, "TRUE": 204},
        "INSUFFANESTH": {"FALSE": 927, "TRUE": 73},
        "INTUBATION": {"NORMAL": 919, "ONESIDED": 47, "ESOPHAGEAL": 34},
        "KINKEDTUBE": {"FALSE": 963, "TRUE": 37},
        "LVEDVOLUME": {"NORMAL": 707, "HIGH": 212, "LOW": 81},
        "LVFAILURE": {"FALSE": 952, "TRUE": 48},
        "MINVOL": {"ZERO": 700, "HIGH": 205, "LOW": 66, "NORMAL": 29},
        "MINVOLSET": {"NORMAL": 891, "HIGH": 55, "LOW": 54},
        "PAP": {"NORMAL": 884, "HIGH": 63, "LOW": 53},
        "PCWP": {"NORMAL": 677, "HIGH": 217, "LOW": 106},
        "PRESS": {"HIGH": 519, "LOW": 256, "NORMAL": 196, "ZERO": 29},
        "PULMEMBOLUS": {"FALSE": 988, "TRUE": 12},
        "PVSAT": {"LOW": 786, "HIGH": 184, "NORMAL": 30},
        "SAO2": {"LOW": 783, "HIGH": 182, "NORMAL": 35},
        "SHUNT": {"NORMAL": 897, "HIGH": 103},
        "STROKEVOLUME": {"NORMAL": 771, "LOW": 188, "HIGH": 41},
        "TPR": {"NORMAL": 412, "LOW": 301, "HIGH": 287},
        "VENTALV": {"ZERO": 681, "HIGH": 201, "LOW": 84, "NORMAL": 34},
        "VENTLUNG": {"ZERO": 725, "LOW": 235, "HIGH": 29, "NORMAL": 11},
        "VENTMACH": {"NORMAL": 819, "LOW": 62, "ZERO": 61, "HIGH": 58},
        "VENTTUBE": {"LOW": 723, "ZERO": 205, "HIGH": 58, "NORMAL": 14},
    }
    assert all([t == "category" for t in data.node_types.values()])
    assert data.dstype == "categorical"
