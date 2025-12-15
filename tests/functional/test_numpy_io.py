# Test the read and write functions of NumPy concrete implementation of Data

from os import remove
from pathlib import Path
from random import random

import pytest
from causaliq_core.utils import FileFormatError
from numpy import array
from pandas import DataFrame

from causaliq_data import NumPy

TESTDATA_DIR = str(Path(__file__).parent.parent / "data" / "functional")

AB_3 = str(Path(TESTDATA_DIR) / "simple" / "ab_3.csv")
PQ_3 = str(Path(TESTDATA_DIR) / "simple" / "pq_3.csv")
YESNO_4 = str(Path(TESTDATA_DIR) / "simple" / "yesno_4.csv")


@pytest.fixture(scope="module")  # AB, 2 rows
def ab2():
    df = DataFrame({"A": ["0", "1"], "B": ["1", "1"]}, dtype="category")
    return NumPy.from_df(df=df, dstype="categorical", keep_df=True)


@pytest.fixture(scope="module")  # continuous XY, 2 rows
def xy2():
    data = array([[1.04, -0.00348], [132, 0.0000453]], dtype="float32")
    dstype = "continuous"
    col_values = {"X": None, "Y": None}
    return NumPy(data, dstype, col_values)


@pytest.fixture(scope="function")  # temp file, automatically removed
def tmpfile():
    _tmpfile = str(
        Path(TESTDATA_DIR) / "tmp" / "{}.csv".format(int(random() * 10000000))
    )
    yield _tmpfile
    remove(_tmpfile)


@pytest.fixture(scope="function")  # temp file, automatically removed
def tmpgzfile():
    _tmpfile = str(
        Path(TESTDATA_DIR)
        / "tmp"
        / "{}.csv.gz".format(int(random() * 10000000))
    )
    yield _tmpfile
    remove(_tmpfile)


# no arguments provided
def test_read_type_error_1_():
    with pytest.raises(TypeError):
        NumPy.read()


# filename bad arg type
def test_read_type_error_2_():
    with pytest.raises(TypeError):
        NumPy.read(None, dstype="continuous")
    with pytest.raises(TypeError):
        NumPy.read(True, dstype="continuous")
    with pytest.raises(TypeError):
        NumPy.read(1, dstype="categorical")
    with pytest.raises(TypeError):
        NumPy.read({AB_3}, dstype="categorical")


# dstype bad type
def test_read_type_error_3_():
    with pytest.raises(TypeError):
        NumPy.read(AB_3, dstype="invalid")
    with pytest.raises(TypeError):
        NumPy.read(AB_3, dstype=True)
    with pytest.raises(TypeError):
        NumPy.read(AB_3, dstype={"continuous"})
    with pytest.raises(TypeError):
        NumPy.read(AB_3, dstype=6)


# N bad type
def test_read_type_error_4_():
    with pytest.raises(TypeError):
        NumPy.read(AB_3, dstype="categorical", N=True)
    with pytest.raises(TypeError):
        NumPy.read(AB_3, dstype="categorical", N=1.2)
    with pytest.raises(TypeError):
        NumPy.read(AB_3, dstype="categorical", N=(3,))


# non-existent file
def test_read_filenotfound_error_1_():
    with pytest.raises(FileNotFoundError):
        NumPy.read(
            str(Path(TESTDATA_DIR) / "simple" / "nonexistent.csv"),
            dstype="categorical",
        )
    with pytest.raises(FileNotFoundError):
        NumPy.read(
            str(Path(TESTDATA_DIR) / "nonexistent" / "ab_3.csv"),
            dstype="categorical",
        )


# mixed datasets not supported yet
def test_read_value_error_1_():
    with pytest.raises(ValueError):
        NumPy.read(AB_3, dstype="mixed")


# N bad value
def test_read_value_error_2_():
    with pytest.raises(ValueError):
        NumPy.read(AB_3, dstype="categorical", N=1)
    with pytest.raises(ValueError):
        NumPy.read(AB_3, dstype="categorical", N=-1)
    with pytest.raises(ValueError):
        NumPy.read(AB_3, dstype="categorical", N=0)


# N more than number of rows in file
def test_read_value_error_3_():
    with pytest.raises(ValueError):
        NumPy.read(AB_3, dstype="categorical", N=4)


# File only contains one column
def test_read_value_error_5_():
    with pytest.raises(ValueError):
        NumPy.read(
            str(Path(TESTDATA_DIR) / "simple" / "a_2.csv"),
            dstype="categorical",
        )


# File only contains one row
def test_read_value_error_6_():
    with pytest.raises(ValueError):
        NumPy.read(
            str(Path(TESTDATA_DIR) / "simple" / "ab_1.csv"),
            dstype="categorical",
        )


# file categorical, dstype cont
def test_read_value_error_7_():
    with pytest.raises(ValueError):
        NumPy.read(PQ_3, dstype="continuous")


# an empty plain file
def test_read_fileformat_error_1_():
    with pytest.raises(FileFormatError):
        NumPy.read(
            str(Path(TESTDATA_DIR) / "misc" / "empty.txt"),
            dstype="categorical",
        )


# an empty compressed file
def test_read_fileformat_error_2_():
    with pytest.raises(FileFormatError):
        NumPy.read(
            str(Path(TESTDATA_DIR) / "misc" / "empty.pkl.gz"),
            dstype="categorical",
        )


# reading a binary file
def test_read_fileformat_error_3_():
    with pytest.raises(FileFormatError):
        NumPy.read(
            str(Path(TESTDATA_DIR) / "misc" / "null.sys"), dstype="categorical"
        )


# read AB categorical data successfully
def test_read_ab_ok_1_():
    data = NumPy.read(AB_3, dstype="categorical")
    assert isinstance(data, NumPy)


# read YESNO categorical data successfully
def test_read_yesno_ok_1_():
    data = NumPy.read(YESNO_4, dstype="categorical")
    assert isinstance(data, NumPy)


# Asia, 1K rows
def test_read_asia_ok_1_():
    data = NumPy.read(
        str(Path(TESTDATA_DIR) / "experiments" / "datasets" / "asia.data.gz"),
        dstype="categorical",
    )
    assert isinstance(data, NumPy)
    assert data.N == 1000


# XYZ, 1K rows
def test_read_xyz10_ok_1_():
    data = NumPy.read(
        str(Path(TESTDATA_DIR) / "simple" / "xyz_10.csv"), dstype="continuous"
    )
    assert isinstance(data, NumPy)
    assert data.N == 10


# Pandas write tests


# write with no arguments
def test_write_type_error_1():
    with pytest.raises(TypeError):
        NumPy.write()


# write with bad filename type
def test_write_type_error_2():
    with pytest.raises(TypeError):
        NumPy.write(3)
    with pytest.raises(TypeError):
        NumPy.write(None, compress=True)
    with pytest.raises(TypeError):
        NumPy.write([42])
    with pytest.raises(TypeError):
        NumPy.write("invalid")


# write with bad compress type
def test_write_type_error_3(ab2):
    with pytest.raises(TypeError):
        ab2.write(
            str(Path(TESTDATA_DIR) / "tmp" / "wont_create.csv"), compress=None
        )
    with pytest.raises(TypeError):
        ab2.write(
            str(Path(TESTDATA_DIR) / "tmp" / "wont_createt.csv"), compress=1
        )


# write with bad sf type
def test_write_type_error_4(ab2):
    with pytest.raises(TypeError):
        ab2.write(str(Path(TESTDATA_DIR) / "tmp" / "wont_create.csv"), sf=None)
    with pytest.raises(TypeError):
        ab2.write(
            str(Path(TESTDATA_DIR) / "tmp" / "wont_create.csv"), sf="one"
        )
    with pytest.raises(TypeError):
        ab2.write(str(Path(TESTDATA_DIR) / "tmp" / "wont_create.csv"), sf=(1,))
    with pytest.raises(TypeError):
        ab2.write(str(Path(TESTDATA_DIR) / "tmp" / "wont_create.csv"), sf=0.2)
    with pytest.raises(TypeError):
        ab2.write(str(Path(TESTDATA_DIR) / "tmp" / "wont_create.csv"), sf=True)


# write with bad zero type
def test_write_type_error_5(ab2):
    with pytest.raises(TypeError):
        ab2.write(
            str(Path(TESTDATA_DIR) / "tmp" / "wont_create.csv"),
            sf=2,
            zero=False,
        )
    with pytest.raises(TypeError):
        ab2.write(
            str(Path(TESTDATA_DIR) / "tmp" / "wont_create.csv"), sf=2, zero=1
        )


# write with bad preserve type
def test_write_type_error_6(ab2):
    with pytest.raises(TypeError):
        ab2.write(
            str(Path(TESTDATA_DIR) / "tmp" / "wont_create.csv"), preserve=1
        )


# sf not between 2 and 10
def test_write_value_error_1(ab2):
    with pytest.raises(ValueError):
        ab2.write(str(Path(TESTDATA_DIR) / "tmp" / "wont_create.csv"), sf=1)
    with pytest.raises(ValueError):
        ab2.write(str(Path(TESTDATA_DIR) / "tmp" / "wont_create.csv"), sf=0)
    with pytest.raises(ValueError):
        ab2.write(str(Path(TESTDATA_DIR) / "tmp" / "wont_create.csv"), sf=-1)
    with pytest.raises(ValueError):
        ab2.write(str(Path(TESTDATA_DIR) / "tmp" / "wont_create.csv"), sf=11)


# zero not between 1E-20 and 1E-1
def test_write_value_error_2(ab2):
    with pytest.raises(ValueError):
        ab2.write(
            str(Path(TESTDATA_DIR) / "tmp" / "wont_create.csv"), sf=2, zero=0.2
        )
    with pytest.raises(ValueError):
        ab2.write(
            str(Path(TESTDATA_DIR) / "tmp" / "wont_create.csv"),
            sf=2,
            zero=-0.01,
        )
    with pytest.raises(ValueError):
        ab2.write(
            str(Path(TESTDATA_DIR) / "tmp" / "wont_create.csv"),
            sf=2,
            zero=1e-21,
        )


# bad directory
def test_write_filenotfound_error_1(ab2):
    with pytest.raises(FileNotFoundError):
        ab2.write(str(Path(TESTDATA_DIR) / "nonexistent" / "bad.csv"))


# write a non-compressed file OK
def test_write_ab2_1_ok(ab2, tmpfile):
    ab2.write(tmpfile)
    check = NumPy.read(tmpfile, dstype="categorical")
    assert check.as_df().to_dict() == ab2.as_df().to_dict()


# write a compressed file OK
def test_write_ab2_2_ok(ab2, tmpgzfile):
    ab2.write(tmpgzfile, compress=True)
    check = NumPy.read(tmpgzfile, dstype="categorical")
    assert check.as_df().to_dict() == ab2.as_df().to_dict()


# write a cont file OK
def test_write_xy2_1_ok(xy2, tmpgzfile):
    xy2.write(tmpgzfile, compress=True)
    check = NumPy.read(tmpgzfile, dstype="continuous")
    assert check.as_df().to_dict() == xy2.as_df().to_dict()


# check round to 2 s.f., zero 0.01
def test_numpy_xy2_2_ok(xy2, tmpgzfile):
    xy2.write(tmpgzfile, compress=True, sf=2)
    check = NumPy.read(tmpgzfile, dstype="continuous")
    assert (check.data == array([[1.0, 0.0], [130.0, 0.0]])).all().all()


# check rounding to 2 s.f., zero 1E-5
def test_numpy_xy2_3_ok(xy2, tmpfile):
    xy2.write(tmpfile, compress=False, sf=2, zero=1e-5)
    check = NumPy.read(tmpfile, dstype="continuous")
    assert (
        (
            check.data
            == array([[1.0, -0.0035], [130.0, 0.000045]], dtype="float32")
        )
        .all()
        .all()
    )


# check rounding to 3 s.f., zero 1E-6
def test_write_xy2_4_ok(xy2, tmpfile):
    xy2.write(tmpfile, compress=False, sf=3, zero=1e-6)
    check = NumPy.read(tmpfile, dstype="continuous")
    assert (
        (
            check.data
            == array([[1.04, -0.00348], [132.0, 0.0000453]], dtype="float32")
        )
        .all()
        .all()
    )


# check rounding to 7 s.f.,
def test_write_xy3_1_ok(tmpfile):
    data = array(
        [[1.04, 132], [-0.00348066, 0.00045], [1e-5, 43.12345]],
        dtype="float32",
    )
    data = NumPy(data, "continuous", {"X": None, "Y": None})
    data.write(tmpfile, compress=False, sf=3, zero=1e-6)
    check = NumPy.read(tmpfile, dstype="continuous")
    assert (
        (
            check.data
            == array(
                [[-0.00348, 0.00045], [1e-5, 43.1], [1.04, 132.0]],
                dtype="float32",
            )
        )
        .all()
        .all()
    )
