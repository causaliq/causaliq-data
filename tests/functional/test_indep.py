# Test probability independence testing code

import pytest
from causaliq_core.bn import BN
from causaliq_core.bn.io import read_bn
from causaliq_core.graph.io.bayesys import read as read_dag
from causaliq_core.utils import FileFormatError, dicts_same, values_same
from pandas import DataFrame

from causaliq_data.indep import check_test_params, indep
from causaliq_data.pandas import Pandas

TESTDATA_DIR = "tests/data/functional"
TYPES = ["x2", "mi"]


# --- Failure cases


# Test bad primary argument types
def test_indep_type_error_1():
    bn_cancer = read_bn(TESTDATA_DIR + "/discrete/small/cancer.dsc")
    with pytest.raises(TypeError):
        indep()
    with pytest.raises(TypeError):
        indep(6, "a")
    with pytest.raises(TypeError):
        indep(
            "A",
            "B",
            6,
            DataFrame({"A": ["1", "0"], "B": ["1", "0"]}),
            types="mi",
        )
    with pytest.raises(TypeError):
        indep("A", "B", None, {"A": ["1", "0"], "B": ["1", "0"]}, types="mi")
    with pytest.raises(TypeError):
        indep(
            "A",
            "B",
            None,
            DataFrame({"A": ["1", "0"], "B": ["1", "0"]}),
            bn=False,
            types="mi",
        )
    with pytest.raises(TypeError):
        indep("A", "B", None, None, bn=False, types="mi")
    with pytest.raises(TypeError):
        indep("A", "B", None, None, bn=bn_cancer, N="badtype", types="mi")
    with pytest.raises(TypeError):
        indep("A", "B", None, None, bn=bn_cancer, N=True, types="mi")
    with pytest.raises(TypeError):
        indep(
            "A",
            "B",
            None,
            DataFrame({"A": ["1", "0"], "B": ["1", "0"]}),
            bn=bn_cancer,
            types="mi",
        )


# Test bad types in z list
def test_indep_type_error_2():
    lizards_data = TESTDATA_DIR + "/simple/lizards.csv"
    with pytest.raises(TypeError):
        indep(
            "A",
            "B",
            ["C", True],
            DataFrame({"A": ["1", "0"], "B": ["1", "0"], "C": ["2", "3"]}),
            types="mi",
        )
    with pytest.raises(TypeError):
        indep(
            "Diameter", "Height", [10, "Species"], lizards_data, types=["mi"]
        )


# Test bad types in types list
def test_indep_type_error_3():
    lizards_data = TESTDATA_DIR + "/simple/lizards.csv"
    with pytest.raises(TypeError):
        indep(
            "Diameter", "Height", ["Species"], lizards_data, types=["mi", 3.5]
        )
    with pytest.raises(TypeError):
        indep(
            "Diameter",
            "Height",
            ["Species"],
            lizards_data,
            types=["x2", ["mi"]],
        )


# Test non-existent file for data
def test_indep_file_error_1():
    with pytest.raises(FileNotFoundError):
        indep("Diameter", "Height", ["Species"], "nonexistent.txt")


# Test binary file for data
def test_indep_file_error_2():
    with pytest.raises(FileFormatError):
        indep(
            "Diameter", "Height", ["Species"], TESTDATA_DIR + "/misc/null.sys"
        )


# Test variable name duplicated
def test_indep_value_error_1():
    with pytest.raises(ValueError):
        indep(
            "Diameter",
            "Height",
            ["Diameter"],
            TESTDATA_DIR + "/simple/lizards.csv",
        )
    with pytest.raises(ValueError):
        indep(
            "Height",
            "Height",
            ["Diameter"],
            TESTDATA_DIR + "/simple/lizards.csv",
        )
    with pytest.raises(ValueError):
        indep(
            "Diameter",
            "Height",
            ["Species", "Species"],
            TESTDATA_DIR + "/simple/lizards.csv",
        )


# Test variable names not in data
def test_indep_value_error_2():
    with pytest.raises(ValueError):
        indep(
            "Diameter",
            "Height",
            ["Unknown"],
            TESTDATA_DIR + "/simple/lizards.csv",
        )
    with pytest.raises(ValueError):
        indep(
            "Diameter",
            "Height",
            ["Species", "Unknown"],
            TESTDATA_DIR + "/simple/lizards.csv",
        )
    with pytest.raises(ValueError):
        indep(
            "Unknown",
            "Height",
            ["Species"],
            TESTDATA_DIR + "/simple/lizards.csv",
        )
    with pytest.raises(ValueError):
        indep(
            "Diameter",
            "Unknown",
            ["Species"],
            TESTDATA_DIR + "/simple/lizards.csv",
        )


# Test variable names not in BN
def test_indep_value_error_3():
    bn_cancer = read_bn(TESTDATA_DIR + "/discrete/small/cancer.dsc")
    with pytest.raises(ValueError):
        indep("Smoker", "Pollution", ["Unknown"], data=None, bn=bn_cancer)
    with pytest.raises(ValueError):
        indep("Xray", "Cancer", ["Smoker", "Unknown"], data=None, bn=bn_cancer)
    with pytest.raises(ValueError):
        indep("Unknown", "Smoker", ["Pollution"], data=None, bn=bn_cancer)
    with pytest.raises(ValueError):
        indep("Cancer", "Unknown", ["Pollution"], data=None, bn=bn_cancer)


# Test sample size is negative
def test_indep_value_error_4():
    bn_cancer = read_bn(TESTDATA_DIR + "/discrete/small/cancer.dsc")
    with pytest.raises(ValueError):
        indep("Smoker", "Pollution", ["Cancer"], data=None, N=-1, bn=bn_cancer)
    with pytest.raises(ValueError):
        indep("Smoker", "Pollution", ["Cancer"], data=None, N=-3, bn=bn_cancer)


# Test duplicate tests specified
def test_indep_value_error_5():
    with pytest.raises(ValueError):
        indep(
            "Diameter",
            "Height",
            None,
            TESTDATA_DIR + "/simple/lizards.csv",
            types=["mi", "mi"],
        )
    with pytest.raises(ValueError):
        indep(
            "Diameter",
            "Height",
            None,
            TESTDATA_DIR + "/simple/lizards.csv",
            types=["mi", "x2", "mi"],
        )


# Test empty list of tests specified
def test_indep_value_error_6():
    with pytest.raises(ValueError):
        indep(
            "Diameter",
            "Height",
            None,
            TESTDATA_DIR + "/simple/lizards.csv",
            types=[],
        )


# Test unsupported test specified
def test_indep_value_error_7():
    with pytest.raises(ValueError):
        indep(
            "Diameter",
            "Height",
            None,
            TESTDATA_DIR + "/simple/lizards.csv",
            types=["mi", "unsupported"],
        )
    with pytest.raises(ValueError):
        indep(
            "Diameter",
            "Height",
            None,
            TESTDATA_DIR + "/simple/lizards.csv",
            types="unsupported",
        )


# --- Successful independence tests


# Test A, B deterministic, 2 cases
def test_indep_a_b_ok1():
    data = DataFrame({"A": ["1", "0"], "B": ["1", "0"]})
    test = indep("A", "B", None, data, types=TYPES)
    print("\nIndependence tests for 2 deterministic cases:\n{}".format(test))
    # Golden reference values (previously from bnlearn_indep)
    expected = {
        "x2": {"df": 1.0, "statistic": 2.0, "p_value": 0.15729920705028488},
        "mi": {
            "df": 1.0,
            "statistic": 2.772588722239781,
            "p_value": 0.09589096714246559,
        },
    }
    for type in TYPES:
        assert dicts_same(expected[type], test[type].to_dict())


# Test A, B deterministic, 10 cases
def test_indep_a_b_ok2():
    data = DataFrame(
        {
            "A": ["1", "1", "1", "1", "1", "0", "0", "0", "0", "0"],
            "B": ["0", "0", "0", "0", "0", "1", "1", "1", "1", "1"],
        }
    )
    test = indep("A", "B", None, data, types=TYPES)
    print("\nIndependence tests for 10 deterministic cases:\n{}".format(test))
    # Golden reference values (previously from bnlearn_indep)
    expected = {
        "x2": {"df": 1.0, "statistic": 10.0, "p_value": 0.0015654022580025018},
        "mi": {
            "df": 1.0,
            "statistic": 13.862943611198906,
            "p_value": 0.00019663765560207214,
        },
    }
    for type in TYPES:
        assert dicts_same(expected[type], test[type].to_dict())


# Test A->B check data, cpt give same p-value
def test_indep_ab_ok1():

    ab = read_bn(TESTDATA_DIR + "/discrete/tiny/ab.dsc")  # A-->B BN
    N = 1000
    data = Pandas(df=ab.generate_cases(N))

    ab = BN.fit(ab.dag, data)  # re-fit BN so CPTs match data

    # Check CI test results from data and CPT parameters match

    dep_data = indep("A", "B", None, data.sample, types=TYPES)
    dep_cpt = indep("A", "B", None, None, ab, N, types=TYPES)
    for type in TYPES:
        assert dicts_same(dep_data[type].to_dict(), dep_cpt[type].to_dict())


# Test A->B->C - check A, B dependencies match
def test_indep_abc_1_ok():

    abc = read_bn(TESTDATA_DIR + "/discrete/tiny/abc.dsc")  # A-->B BN
    N = 1000
    data = Pandas(df=abc.generate_cases(N))
    print(abc.global_distribution())

    abc = BN.fit(abc.dag, data)  # re-fit BN so CPTs match data

    # Check CI test results from data and CPT parameters match

    dep_data = indep("A", "B", None, data.sample, types=TYPES)
    dep_cpt = indep("A", "B", None, None, abc, N, types=TYPES)
    for type in TYPES:
        assert dicts_same(dep_data[type].to_dict(), dep_cpt[type].to_dict())


# Test A->B->C - check B, C dependencies match
def test_indep_abc_2_ok():

    abc = read_bn(TESTDATA_DIR + "/discrete/tiny/abc.dsc")  # get A-->B-->C BN
    N = 1000
    data = Pandas(df=abc.generate_cases(N))
    print(abc.global_distribution())

    abc = BN.fit(abc.dag, data)  # re-fit BN so CPTs match data

    # Check CI test results from data and CPT parameters match

    dep_data = indep("B", "C", None, data.sample, types=TYPES)
    dep_cpt = indep("B", "C", None, None, abc, N, types=TYPES)
    for type in TYPES:
        assert dicts_same(dep_data[type].to_dict(), dep_cpt[type].to_dict())


# Test A->B->C - check A, C dependencies match
def test_indep_abc_ok3():

    abc = read_bn(TESTDATA_DIR + "/discrete/tiny/abc.dsc")  # get A-->B-->C BN
    N = 10000
    data = Pandas(df=abc.generate_cases(N))

    abc = BN.fit(abc.dag, data)  # re-fit BN so CPTs match data

    # Check CI test p-values from data and CPT parameters match and show
    # Note the test statistics are not compared because they are
    # not expected to be exactly the same as the CPT parameters won't be
    # 'nice' fractions and so a huge sample size is needed for the data and
    # CPT derived frequencies to match exactly.

    dep_data = indep("A", "C", None, data.sample, types=TYPES)
    dep_cpt = indep("A", "C", None, None, abc, N, types=TYPES)
    for type in TYPES:
        assert values_same(dep_data[type].to_dict()["p_value"], 0.0)
        assert values_same(dep_cpt[type].to_dict()["p_value"], 0.0)


# Test A->B->C - check A, C given B independent
def test_indep_abc_ok4():

    abc = read_bn(TESTDATA_DIR + "/discrete/tiny/abc.dsc")  # get A-->B-->C BN
    N = 10000
    data = abc.generate_cases(N)  # generate data for 10000 cases

    # For conditional independence A ⊥ C | B, we expect the test to show
    # independence
    dep_data = indep("A", "C", "B", data, types=TYPES)
    # Since A->B->C, A and C should be independent given B
    for type in TYPES:
        # We expect high p-values (> 0.05) indicating independence
        assert dep_data[type].to_dict()["p_value"] > 0.01


# Test dependence in Lizards
def test_bnlearn_indep_lizards_1_ok():
    data = Pandas.read(
        TESTDATA_DIR + "/simple/lizards.csv", dstype="categorical"
    )

    # check CI statistics from this code and expected golden reference
    # values. Note they DO NOT cause rejection of independence

    dep_data = indep("Height", "Diameter", None, data.sample, types=TYPES)
    print(
        "\nLizards - Height, Diameter CI stats from real data:\n{}\n".format(
            dep_data
        )
    )
    # Golden reference values (previously from bnlearn_indep)
    expected = {
        "x2": {
            "df": 1.0,
            "statistic": 0.60519424065575,
            "p_value": 0.43660302271342777,
        },
        "mi": {
            "df": 1.0,
            "statistic": 0.6077051927518164,
            "p_value": 0.4356531562228757,
        },
    }
    for t in TYPES:
        assert dicts_same(expected[t], dep_data[t].to_dict())

    # check CI test statistics from CPTs learnt from data do show
    # independence at large sample sizes

    dag = read_dag(TESTDATA_DIR + "/bayesys/lizards.csv")
    bn = BN.fit(dag, data)
    dep_cpt = indep("Height", "Diameter", None, None, bn=bn, types=TYPES)
    print(
        "\nLizards - Height Species CI stats from CPTs with N=10**9:\n"
        "{}\n".format(dep_cpt)
    )
    for type in TYPES:
        assert values_same(dep_cpt[type].to_dict()["p_value"], 0)


# Test conditional independence in Lizards
def test_bnlearn_indep_lizards_2_ok():
    data = Pandas.read(
        TESTDATA_DIR + "/simple/lizards.csv", dstype="categorical"
    )

    # check CI statistics from this code and expected golden reference
    # values. Data does show independence

    dep_data = indep("Height", "Diameter", "Species", data.sample, types=TYPES)
    print(
        "\nLizards - Height, Diameter | Species CI stats from data:\n"
        "{}\n".format(dep_data)
    )
    # Golden reference values (previously from bnlearn_indep)
    expected = {
        "x2": {
            "df": 2.0,
            "statistic": 2.0173636681378735,
            "p_value": 0.3646993971873451,
        },
        "mi": {
            "df": 2.0,
            "statistic": 2.025646649876277,
            "p_value": 0.363192121302939,
        },
    }
    for type in TYPES:
        assert dicts_same(expected[type], dep_data[type].to_dict())

    # check CI test statistics from CPTs learnt from data show
    # independence

    dag = read_dag(TESTDATA_DIR + "/bayesys/lizards.csv")
    bn = BN.fit(dag, data)
    dep_cpt = indep("Height", "Diameter", "Species", None, bn=bn, types=TYPES)
    print(
        "\nLizards - Height, Diameter | Species CI stats from CPTs:\n"
        "{}\n".format(dep_cpt)
    )
    for type in TYPES:
        assert values_same(dep_cpt[type].to_dict()["p_value"], 1)


# Test conditional dependence in Lizards
def test_bnlearn_indep_lizards_ok3():
    data = Pandas.read(
        TESTDATA_DIR + "/simple/lizards.csv", dstype="categorical"
    )

    # check CI statistics from this code and expected golden reference values

    dep_data = indep("Species", "Diameter", "Height", data.sample, types=TYPES)
    print(
        "\nLizards - Species, Diameter | Height CI stats from data:\n"
        "{}\n".format(dep_data)
    )
    # Golden reference values (previously from bnlearn_indep)
    expected = {
        "x2": {
            "df": 2.0,
            "statistic": 13.781277646918205,
            "p_value": 0.0010172637814132024,
        },
        "mi": {
            "df": 2.0,
            "statistic": 14.024135550832945,
            "p_value": 0.0009009437117287744,
        },
    }
    for type in TYPES:
        assert dicts_same(expected[type], dep_data[type].to_dict())

    # check CI test statistics from CPTs learnt from data show dependence

    dag = read_dag(TESTDATA_DIR + "/bayesys/lizards.csv")
    bn = BN.fit(dag, data)
    dep_cpt = indep("Species", "Diameter", "Height", None, bn=bn, types=TYPES)
    print(
        "\nLizards - Species, Diameter | Heights CI stats from CPTs:\n"
        "{}\n".format(dep_cpt)
    )
    for type in TYPES:
        assert values_same(dep_cpt[type].to_dict()["p_value"], 0)


# Test independence in cancer BN
def test_bnlearn_indep_cancer_ok1():
    cancer = read_bn(TESTDATA_DIR + "/discrete/small/cancer.dsc")
    data = Pandas(df=cancer.generate_cases(1000))
    test = indep("Pollution", "Smoker", None, data.sample, types=TYPES)
    print("\nCancer - Pollution, Smoker:\n{}".format(test))
    # Golden reference values (previously from bnlearn_indep)
    expected = {
        "x2": {
            "df": 1.0,
            "statistic": 0.28793045679182255,
            "p_value": 0.5915498046048349,
        },
        "mi": {
            "df": 1.0,
            "statistic": 0.28454716409861014,
            "p_value": 0.593736215179713,
        },
    }
    for type in TYPES:
        assert dicts_same(expected[type], test[type].to_dict())


# Test dependence in cancer BN
def test_bnlearn_indep_cancer_ok2():
    cancer = read_bn(TESTDATA_DIR + "/discrete/small/cancer.dsc")
    data = Pandas(df=cancer.generate_cases(1000))
    test = indep("Smoker", "Cancer", None, data.sample, types=TYPES)
    print("\nCancer - Smoker, Cancer:\n{}".format(test))
    # Golden reference values (previously from bnlearn_indep)
    expected = {
        "x2": {
            "df": 1.0,
            "statistic": 25.784671167164277,
            "p_value": 3.8170800831238694e-07,
        },
        "mi": {
            "df": 1.0,
            "statistic": 24.59507462690446,
            "p_value": 7.073229180143414e-07,
        },
    }
    for type in TYPES:
        assert dicts_same(expected[type], test[type].to_dict())


# Test dependence in cancer BN
def test_bnlearn_indep_cancer_ok3():
    cancer = read_bn(TESTDATA_DIR + "/discrete/small/cancer.dsc")
    data = Pandas(df=cancer.generate_cases(1000))
    test = indep("Pollution", "Cancer", None, data.sample, types=TYPES)
    print("\nCancer - Pollution, Cancer:\n{}".format(test))
    # Golden reference values (previously from bnlearn_indep)
    expected = {
        "x2": {
            "df": 1.0,
            "statistic": 0.15318277061192007,
            "p_value": 0.6955121565789647,
        },
        "mi": {
            "df": 1.0,
            "statistic": 0.1694572972094579,
            "p_value": 0.6805945758087499,
        },
    }
    for type in TYPES:
        assert dicts_same(expected[type], test[type].to_dict())


# Test conditional independence in cancer BN, cond set = 1
def test_bnlearn_indep_cancer_ok4():
    cancer = read_bn(TESTDATA_DIR + "/discrete/small/cancer.dsc")
    data = Pandas(df=cancer.generate_cases(1000))
    test = indep("Xray", "Smoker", "Cancer", data.sample, types=TYPES)
    print("\nCancer - Xray, Smoker | Cancer:\n{}".format(test))
    # Golden reference values (previously from bnlearn_indep)
    expected = {
        "x2": {
            "df": 2.0,
            "statistic": 1.783626192413504,
            "p_value": 0.409911868934995,
        },
        "mi": {
            "df": 2.0,
            "statistic": 1.8919851191421189,
            "p_value": 0.38829397466885174,
        },
    }
    for type in TYPES:
        assert dicts_same(expected[type], test[type].to_dict())


# Test conditional independence in cancer BN, cond set = 2
def test_bnlearn_indep_cancer_ok5():
    cancer = read_bn(TESTDATA_DIR + "/discrete/small/cancer.dsc")
    data = Pandas(df=cancer.generate_cases(5000))
    cancer = BN.fit(cancer.dag, data)
    test = indep(
        "Xray", "Smoker", ["Cancer", "Pollution"], data.sample, types=TYPES
    )
    print("\nCancer - Xray, Smoker | Cancer, Pollution:\n{}".format(test))
    # Golden reference values (previously from bnlearn_indep)
    expected = {
        "x2": {
            "df": 4.0,
            "statistic": 1.3423098023411906,
            "p_value": 0.8541572606633745,
        },
        "mi": {
            "df": 4.0,
            "statistic": 1.9904877565588612,
            "p_value": 0.7375085551313881,
        },
    }
    print(
        indep(
            "Xray",
            "Smoker",
            ["Cancer", "Pollution"],
            None,
            cancer,
            5000,
            types=TYPES,
        )
    )
    for type in TYPES:
        assert dicts_same(expected[type], test[type].to_dict())


# Test conditional dependence in cancer BN
def test_bnlearn_indep_cancer_ok6():
    cancer = read_bn(TESTDATA_DIR + "/discrete/small/cancer.dsc")
    data = cancer.generate_cases(1000)
    test = indep("Smoker", "Pollution", "Cancer", data, types=TYPES)
    print("\nCancer - Smoker, Pollution | Cancer:\n{}".format(test))
    # Golden reference values (previously from bnlearn_indep)
    expected = {
        "x2": {
            "df": 2.0,
            "statistic": 14.67204068525291,
            "p_value": 0.0006516386661854501,
        },
        "mi": {
            "df": 2.0,
            "statistic": 7.864518219249568,
            "p_value": 0.019599345432099624,
        },
    }
    for type in TYPES:
        assert dicts_same(expected[type], test[type].to_dict())


# --- Independence tests with continuous data


# Test X1 and Y2 are unconditionally independent
def test_bnlearn_indep_xyzw_ok1():
    data = Pandas.read(
        TESTDATA_DIR + "/simple/xyzw.csv", dstype="categorical"
    ).df
    print(data.value_counts())
    test = indep("X1", "Y2", None, data, types=TYPES)
    print("\nXYZW - X1, Y2:\n{}".format(test))
    # Golden reference values (previously from bnlearn_indep)
    expected = {
        "x2": {"df": 1.0, "statistic": 0.0, "p_value": 1.0},
        "mi": {"df": 1.0, "statistic": 0.0, "p_value": 1.0},
    }
    for type in TYPES:
        assert dicts_same(expected[type], test[type].to_dict())


# Test X1 and W4 are dependent - 2x2 table
def test_bnlearn_indep_xyzw_ok2():
    data = Pandas.read(
        TESTDATA_DIR + "/simple/xyzw.csv", dstype="categorical"
    ).df
    test = indep("X1", "W4", None, data, types=TYPES)
    print("\nXYZW - X1, W4:\n{}".format(test))
    # Golden reference values (previously from bnlearn_indep)
    expected = {
        "x2": {
            "df": 1.0,
            "statistic": 0.1206090758329565,
            "p_value": 0.7283748343120175,
        },
        "mi": {
            "df": 1.0,
            "statistic": 0.12023923109641554,
            "p_value": 0.7287751688628149,
        },
    }
    for type in TYPES:
        assert dicts_same(expected[type], test[type].to_dict())


# Test X1 and Z3 are dependent - 2x3 table
def test_bnlearn_indep_xyzw_ok3():
    data = Pandas.read(
        TESTDATA_DIR + "/simple/xyzw.csv", dstype="categorical"
    ).df
    test = indep("X1", "Z3", None, data, types=TYPES)
    print("\nXYZW - X1, Z3:\n{}".format(test))
    # Golden reference values (previously from bnlearn_indep)
    expected = {
        "x2": {
            "df": 2.0,
            "statistic": 2.2651295836779703,
            "p_value": 0.32220580493646134,
        },
        "mi": {
            "df": 2.0,
            "statistic": 2.227437606857503,
            "p_value": 0.32833567207621794,
        },
    }
    for type in TYPES:
        assert dicts_same(expected[type], test[type].to_dict())


# Test single conditioning variable
def test_bnlearn_indep_xyzw_ok4():
    data = Pandas.read(
        TESTDATA_DIR + "/simple/xyzw.csv", dstype="categorical"
    ).df
    test = indep("X1", "Y2", "Z3", data, types=TYPES)
    print("\nXYZW - X1, Y2 | Z3:\n{}".format(test))
    # Golden reference values (previously from bnlearn_indep)
    expected = {
        "x2": {
            "df": 3.0,
            "statistic": 5.666981019966592,
            "p_value": 0.12898550393313235,
        },
        "mi": {
            "df": 3.0,
            "statistic": 6.318574137081791,
            "p_value": 0.09709875307235105,
        },
    }
    for type in TYPES:
        assert dicts_same(expected[type], test[type].to_dict())


# Test two conditioning variables
def test_bnlearn_indep_xyzw_ok5():
    data = Pandas.read(
        TESTDATA_DIR + "/simple/xyzw.csv", dstype="categorical"
    ).df
    test = indep("X1", "Y2", ["Z3", "W4"], data, types=TYPES)
    print("\nXYZW - X1, Y2 | Z3, W4:\n{}".format(test))
    # Golden reference values (previously from bnlearn_indep)
    expected = {
        "x2": {
            "df": 6.0,
            "statistic": 8.011940183999007,
            "p_value": 0.23722984212140874,
        },
        "mi": {
            "df": 6.0,
            "statistic": 8.792240952112762,
            "p_value": 0.18560383445524475,
        },
    }
    for type in TYPES:
        assert dicts_same(expected[type], test[type].to_dict())


# Test generate zero row in a contingency table
def test_bnlearn_indep_xyzw_ok6():
    data = Pandas.read(
        TESTDATA_DIR + "/simple/xyzw.csv", dstype="categorical"
    ).df
    indep("W4", "Z3", ["X1", "Y2"], data, types="x2")
    test = indep("Z3", "W4", ["X1", "Y2"], data, types=TYPES)
    print("\nXYZW - Z3, W4 | X1, Y2:\n{}".format(test))
    # Golden reference values (previously from bnlearn_indep)
    expected = {
        "x2": {
            "df": 8.0,
            "statistic": 2.033375774552245,
            "p_value": 0.9799715331308619,
        },
        "mi": {
            "df": 8.0,
            "statistic": 1.9535295075329975,
            "p_value": 0.9824035009383545,
        },
    }
    for type in TYPES:
        assert dicts_same(expected[type], test[type].to_dict())


# --- Test check_test_params


# Test alpha not a float
def test_check_test_params_type_error_1():
    with pytest.raises(TypeError):
        check_test_params({"alpha": 1})
    with pytest.raises(TypeError):
        check_test_params({"alpha": "wrong type"})


# Test alpha out of range
def test_check_test_params_value_error_1():
    with pytest.raises(ValueError):
        check_test_params({"alpha": 1e-20})
    with pytest.raises(ValueError):
        check_test_params({"alpha": 1.0})
    with pytest.raises(ValueError):
        check_test_params({"alpha": 0.0})


# Test sets defaults
def test_check_test_params_ok_1():
    assert check_test_params({}) == {"alpha": 0.05}


# Test accepts valid values
def test_check_test_params_ok_2():
    assert check_test_params({"alpha": 0.01}) == {"alpha": 0.01}
    assert check_test_params({"alpha": 0.1}) == {"alpha": 0.1}
    assert check_test_params({"alpha": 0.999}) == {"alpha": 0.999}
    assert check_test_params({"alpha": 1e-2}) == {"alpha": 1e-2}
    assert check_test_params({"alpha": 1e-3}) == {"alpha": 1e-3}
