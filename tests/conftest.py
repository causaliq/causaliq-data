# Shared test configuration and fixtures

import os

import pytest
from pandas import read_csv

from causaliq_data import Pandas

# Test data directory location
TESTDATA_DIR = os.path.join(os.path.dirname(__file__), "data", "functional")


@pytest.fixture(scope="module")
def xyz_data():
    """XYZ continuous dataset with 10 rows for values() tests."""
    return Pandas(
        df=read_csv(
            os.path.join(TESTDATA_DIR, "simple/xyz_10.csv"), dtype="float32"
        )
    )
