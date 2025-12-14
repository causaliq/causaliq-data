"""
Tests for NumPy data adapter DataFrame conversion functionality.

These tests validate the as_df() method that converts NumPy data structures
back to pandas DataFrames.
"""

import pytest
from numpy import array

from causaliq_data import NumPy


@pytest.fixture(scope="module")  # categorical AB, 3 rows
def ab3():
    data = array([[1, 1], [1, 0], [0, 0]], dtype="uint8")
    dstype = "categorical"
    col_values = {"A": ("1", "0"), "B": ("1", "0")}
    return {"d": data, "t": dstype, "v": col_values}


@pytest.fixture(scope="module")  # continuous XY, 3 rows
def xy3():
    data = array([[1, 1], [1, 0], [0, 0]], dtype="float32")
    dstype = "continuous"
    col_values = {"X": None, "Y": None}
    return {"d": data, "t": dstype, "v": col_values}


# Test as_df() converts categorical data to DataFrame with proper categories
def test_as_df_ab3_1_ok(ab3):
    data = NumPy(ab3["d"], ab3["t"], ab3["v"])

    df = data.as_df()

    assert (df == data.as_df()).all().all()
    df = df.to_dict(orient="list")

    print("\n\nab3 NumPy as dataframe: {}\n".format(df))

    assert df == {"A": ["0", "0", "1"], "B": ["0", "1", "1"]}


# Test as_df() converts continuous data to DataFrame with numeric types
def test_as_df_xy3_1_ok(xy3):
    data = NumPy(xy3["d"], xy3["t"], xy3["v"])

    df = data.as_df().to_dict(orient="list")
    print("\n\nxy3 NumPy as dataframe: {}\n".format(df))

    assert df == {"X": [0.0, 1.0, 1.0], "Y": [0.0, 0.0, 1.0]}
