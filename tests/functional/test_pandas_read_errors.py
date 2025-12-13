# Test Pandas read() method - error cases

import pytest
from causaliq_core.utils import FileFormatError

from causaliq_data import Pandas
from tests.conftest import TESTDATA_DIR


# read() requires filename argument
def test_pandas_read_type_error_1():
    """Fails with no arguments."""
    with pytest.raises(TypeError):
        Pandas.read()


# read() requires string filename
def test_pandas_read_type_error_2():
    """Fails with bad filename type."""
    with pytest.raises(TypeError):
        Pandas.read(666)
    with pytest.raises(TypeError):
        Pandas.read(["should not be an array"])


# read() dstype must be valid DatasetType or None
def test_pandas_read_type_error_3():
    """Fails with bad DatasetType."""
    with pytest.raises(TypeError):
        Pandas.read(TESTDATA_DIR + "/simple/ab_3.csv", dstype=True)
    with pytest.raises(TypeError):
        Pandas.read(TESTDATA_DIR + "/simple/ab_3.csv", dstype=37)
    with pytest.raises(TypeError):
        Pandas.read(TESTDATA_DIR + "/simple/ab_3.csv", dstype="invalid")


# read() N parameter must be integer if provided
def test_pandas_read_type_error_4():
    """Fails with bad N type."""
    with pytest.raises(TypeError):
        Pandas.read(TESTDATA_DIR + "/simple/ab_3.csv", N="invalid")
    with pytest.raises(TypeError):
        Pandas.read(TESTDATA_DIR + "/simple/ab_3.csv", N=True)
    with pytest.raises(TypeError):
        Pandas.read(TESTDATA_DIR + "/simple/ab_3.csv", N=[1])


# read() raises FileNotFoundError for non-existent files
def test_pandas_read_filenotfound_error():
    """Fails with nonexistent file."""
    with pytest.raises(FileNotFoundError):
        Pandas.read("nonexistent.txt")


# read() raises FileFormatError for invalid file formats
def test_pandas_read_fileformat_error():
    """Fails with binary file."""
    with pytest.raises(FileFormatError):
        Pandas.read(TESTDATA_DIR + "/misc/null.sys")


# read() cannot coerce non-numeric data to continuous
def test_pandas_read_value_error_1():
    """Invalid coercion to float."""
    with pytest.raises(ValueError):
        Pandas.read(TESTDATA_DIR + "/simple/mix_2.csv", dstype="continuous")


# read() requires N >= 2
def test_pandas_read_value_error_2():
    """Invalid N values."""
    with pytest.raises(ValueError):
        Pandas.read(TESTDATA_DIR + "/simple/mix_2.csv", N=1)
    with pytest.raises(ValueError):
        Pandas.read(TESTDATA_DIR + "/simple/mix_2.csv", N=0)
    with pytest.raises(ValueError):
        Pandas.read(TESTDATA_DIR + "/simple/mix_2.csv", N=-1)


# read() N cannot exceed dataset size
def test_pandas_read_value_error_3():
    """Invalid N bigger than dataset."""
    with pytest.raises(ValueError):
        Pandas.read(TESTDATA_DIR + "/simple/mix_2.csv", N=3)
