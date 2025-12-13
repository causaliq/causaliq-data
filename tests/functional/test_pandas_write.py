# Test Pandas write() method

from os import remove
from random import random

import pytest
from causaliq_core.utils import dicts_same
from pandas import DataFrame

from causaliq_data import Pandas
from tests.conftest import TESTDATA_DIR


@pytest.fixture(scope="function")
def tmpfile():
    """Temporary file, automatically removed after test."""
    _tmpfile = TESTDATA_DIR + "/tmp/{}.csv".format(int(random() * 10000000))
    yield _tmpfile
    remove(_tmpfile)


@pytest.fixture(scope="function")
def tmpgzfile():
    """Temporary gzipped file, automatically removed after test."""
    _tmpfile = TESTDATA_DIR + "/tmp/{}.csv.gz".format(int(random() * 10000000))
    yield _tmpfile
    remove(_tmpfile)


# write() requires filename argument
def test_pandas_write_type_error_1():
    """Write with no arguments."""
    with pytest.raises(TypeError):
        Pandas.write()


# write() requires string filename
def test_pandas_write_type_error_2():
    """Write with bad filename type."""
    with pytest.raises(TypeError):
        Pandas.write(3)
    with pytest.raises(TypeError):
        Pandas.write(None, compress=True)
    with pytest.raises(TypeError):
        Pandas.write([42])
    with pytest.raises(TypeError):
        Pandas.write("invalid")


# write() compress parameter must be boolean
def test_pandas_write_type_error_3():
    """Write with bad compress type."""
    data = Pandas(
        df=DataFrame({"A": ["1", "1"], "B": ["1", "0"]}, dtype="category")
    )
    with pytest.raises(TypeError):
        data.write(TESTDATA_DIR + "/tmp/wont_create.csv", compress=None)
    with pytest.raises(TypeError):
        data.write(TESTDATA_DIR + "/tmp/wont_createt.csv", compress=1)


# write() sf parameter must be integer
def test_pandas_write_type_error_4():
    """Write with bad sf type."""
    data = Pandas(
        df=DataFrame({"A": ["1", "1"], "B": ["1", "0"]}, dtype="category")
    )
    with pytest.raises(TypeError):
        data.write(TESTDATA_DIR + "/tmp/wont_create.csv", sf=None)
    with pytest.raises(TypeError):
        data.write(TESTDATA_DIR + "/tmp/wont_create.csv", sf="one")
    with pytest.raises(TypeError):
        data.write(TESTDATA_DIR + "/tmp/wont_create.csv", sf=(1,))
    with pytest.raises(TypeError):
        data.write(TESTDATA_DIR + "/tmp/wont_create.csv", sf=0.2)
    with pytest.raises(TypeError):
        data.write(TESTDATA_DIR + "/tmp/wont_create.csv", sf=True)


# write() zero parameter must be float if provided
def test_pandas_write_type_error_5():
    """Write with bad zero type."""
    data = Pandas(
        df=DataFrame({"A": ["1", "1"], "B": ["1", "0"]}, dtype="category")
    )
    with pytest.raises(TypeError):
        data.write(TESTDATA_DIR + "/tmp/wont_create.csv", sf=2, zero=False)
    with pytest.raises(TypeError):
        data.write(TESTDATA_DIR + "/tmp/wont_create.csv", sf=2, zero=1)


# write() preserve parameter must be boolean
def test_pandas_write_type_error_6():
    """Write with bad preserve type."""
    data = Pandas(
        df=DataFrame({"A": ["1", "1"], "B": ["1", "0"]}, dtype="category")
    )
    with pytest.raises(TypeError):
        data.write(TESTDATA_DIR + "/tmp/wont_create.csv", preserve=1)


# write() sf must be between 2 and 10
def test_pandas_write_value_error_1():
    """sf not between 2 and 10."""
    data = Pandas(
        df=DataFrame({"A": ["1", "1"], "B": ["1", "0"]}, dtype="category")
    )
    with pytest.raises(ValueError):
        data.write(TESTDATA_DIR + "/tmp/wont_create.csv", sf=1)
    with pytest.raises(ValueError):
        data.write(TESTDATA_DIR + "/tmp/wont_create.csv", sf=0)
    with pytest.raises(ValueError):
        data.write(TESTDATA_DIR + "/tmp/wont_create.csv", sf=-1)
    with pytest.raises(ValueError):
        data.write(TESTDATA_DIR + "/tmp/wont_create.csv", sf=11)


# write() zero must be between 1E-20 and 0.1
def test_pandas_write_value_error_2():
    """zero not between 1E-20 and 1E-1."""
    data = Pandas(
        df=DataFrame({"A": ["1", "1"], "B": ["1", "0"]}, dtype="category")
    )
    with pytest.raises(ValueError):
        data.write(TESTDATA_DIR + "/tmp/wont_create.csv", sf=2, zero=0.2)
    with pytest.raises(ValueError):
        data.write(TESTDATA_DIR + "/tmp/wont_create.csv", sf=2, zero=-0.01)
    with pytest.raises(ValueError):
        data.write(TESTDATA_DIR + "/tmp/wont_create.csv", sf=2, zero=1e-21)


# write() raises FileNotFoundError for non-existent directory
def test_pandas_write_filenotfound_error_1():
    """Bad directory."""
    data = Pandas(
        df=DataFrame({"A": ["1", "1"], "B": ["1", "0"]}, dtype="category")
    )
    with pytest.raises(FileNotFoundError):
        data.write(TESTDATA_DIR + "/nonexistent/bad.csv")


# write() can write discrete non-compressed file
def test_pandas_write_1_ok(tmpfile):
    """Write a discrete non-compressed file OK."""
    data = Pandas(
        df=DataFrame({"A": ["1", "1"], "B": ["1", "0"]}, dtype="category")
    )
    data.write(tmpfile)
    check = Pandas.read(tmpfile, dstype="categorical")
    assert check.df.to_dict() == data.df.to_dict()


# write() can write discrete compressed file
def test_pandas_write_2_ok(tmpgzfile):
    """Write a discrete compressed file OK."""
    data = Pandas(
        df=DataFrame({"A": ["1", "1"], "B": ["1", "0"]}, dtype="category")
    )
    data.write(tmpgzfile, compress=True)
    check = Pandas.read(tmpgzfile, dstype="categorical")
    assert check.df.to_dict() == data.df.to_dict()


# write() can write continuous compressed file
def test_pandas_write_3_ok(tmpgzfile):
    """Write a cont compressed file OK."""
    data = Pandas(
        df=DataFrame({"A": [1.0, -1.0], "B": [1.5, 0.0]}, dtype="float32")
    )
    data.write(tmpgzfile, compress=True, preserve=True)
    check = Pandas.read(tmpgzfile, dstype="continuous")
    assert check.df.to_dict() == data.df.to_dict()


# write() rounds to specified significant figures with zero threshold
def test_pandas_write_4_ok(tmpgzfile):
    """Check rounding to 2 s.f., zero 0.01."""
    data = Pandas(
        df=DataFrame(
            {"A": [1.04, -0.00348], "B": [132, 0.0000453]}, dtype="float32"
        )
    )
    data.write(tmpgzfile, compress=True, sf=2)
    check = Pandas.read(tmpgzfile, dstype="continuous")
    assert check.df.to_dict(orient="list") == {
        "A": [1.0, 0.0],
        "B": [130.0, 0.0],
    }


# write() respects custom zero threshold
def test_pandas_write_5_ok(tmpfile):
    """Check rounding to 2 s.f., zero 1E-5."""
    data = [{"A": 1.04, "B": 132}, {"A": -0.00348066, "B": 0.00045}]
    df = DataFrame.from_records(data).astype("float32")
    Pandas(df).write(tmpfile, compress=False, sf=2, zero=10e-5)
    check = Pandas.read(tmpfile, dstype="continuous")
    print("\nRounded df read back in:\n{}\n".format(check.df))
    check = check.df.to_dict(orient="records")
    for row in zip(data, check):
        assert dicts_same(row[0], row[1], sf=2)


# write() handles 3 significant figures correctly
def test_pandas_write_6_ok(tmpfile):
    """Check rounding to 3 s.f., zero 1E-6."""
    data = [
        {"A": 1.04, "B": 132},
        {"A": -0.00348066, "B": 0.00045},
        {"A": 1e-5, "B": 43.12345},
    ]
    df = DataFrame.from_records(data).astype("float32")
    Pandas(df).write(tmpfile, compress=False, sf=3, zero=1e-6)
    check = Pandas.read(tmpfile, dstype="continuous")
    print("\nRounded df read back in:\n{}\n".format(check.df))
    check = check.df.to_dict(orient="records")
    for row in zip(data, check):
        assert dicts_same(row[0], row[1], sf=3)


# write() handles 7 significant figures correctly
def test_pandas_write_7_ok(tmpfile):
    """Check rounding to 7 s.f."""
    data = [
        {"A": 1.04, "B": 132},
        {"A": -0.00348066, "B": 0.00045},
        {"A": 1e-5, "B": 43.12345},
    ]
    df = DataFrame.from_records(data).astype("float32")
    Pandas(df).write(tmpfile, compress=False, sf=7)
    check = Pandas.read(tmpfile, dstype="continuous")
    print("\nRounded df read back in:\n{}\n".format(check.df))
    check = check.df.to_dict(orient="records")
    for row in zip(data, check):
        assert dicts_same(row[0], row[1], sf=7)
