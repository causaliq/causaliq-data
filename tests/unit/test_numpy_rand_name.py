"""
Tests for NumPy data adapter randomise_names method.

These tests validate name randomization functionality
without requiring external file dependencies.
"""

import pytest
from numpy import array

from causaliq_data import NumPy


@pytest.fixture(scope="module")  # categorical ABC, 36 rows
def abc36():
    data = array(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 1],
            [0, 1, 1],
            [0, 1, 1],
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 1],
            [1, 0, 1],
            [1, 0, 1],
            [1, 0, 1],
            [1, 0, 1],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 0],
            [1, 1, 0],
            [1, 1, 0],
            [1, 1, 0],
            [1, 1, 0],
            [1, 1, 0],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
        ],
        dtype="uint8",
    )
    dstype = "categorical"
    col_values = {"A": ("0", "1"), "B": ("0", "1"), "C": ("0", "1")}
    return {"d": data, "t": dstype, "v": col_values}


# Test randomise_names() with categorical dataset
def test_rand_name_abc36_1_ok(abc36):
    data = NumPy(abc36["d"], abc36["t"], abc36["v"])
    print("\n\nOriginal ABC names:\n{}".format(data.as_df().head()))
    assert data.get_order() == ("A", "B", "C")
    assert data.node_values == {
        "A": {"0": 10, "1": 26},
        "B": {"0": 14, "1": 22},
        "C": {"0": 16, "1": 20},
    }
    assert data.node_types == {
        "A": "category",
        "B": "category",
        "C": "category",
    }

    data.randomise_names(seed=0)
    print("\n\nRandom names, seed 0:\n{}".format(data.as_df().head()))
    assert tuple(data.as_df().columns) == ("X001A", "X002B", "X000C")
    assert data.get_order() == ("X001A", "X002B", "X000C")
    assert data.node_values == {
        "X001A": {"0": 10, "1": 26},
        "X002B": {"0": 14, "1": 22},
        "X000C": {"0": 16, "1": 20},
    }
    assert data.node_types == {
        "X001A": "category",
        "X002B": "category",
        "X000C": "category",
    }
    assert (data.data == abc36["d"]).all().all()

    data.randomise_names(seed=1)
    print("\n\nRandom names, seed 1:\n{}".format(data.as_df().head()))
    assert tuple(data.as_df().columns) == ("X002A", "X000B", "X001C")
    assert data.get_order() == ("X002A", "X000B", "X001C")
    assert data.node_values == {
        "X002A": {"0": 10, "1": 26},
        "X000B": {"0": 14, "1": 22},
        "X001C": {"0": 16, "1": 20},
    }
    assert data.node_types == {
        "X002A": "category",
        "X000B": "category",
        "X001C": "category",
    }
    assert (data.data == abc36["d"]).all().all()

    data.randomise_names(seed=0)
    print("\n\nRandom names, seed 0:\n{}".format(data.as_df().head()))
    assert tuple(data.as_df().columns) == ("X001A", "X002B", "X000C")
    assert data.get_order() == ("X001A", "X002B", "X000C")
    assert data.node_values == {
        "X001A": {"0": 10, "1": 26},
        "X002B": {"0": 14, "1": 22},
        "X000C": {"0": 16, "1": 20},
    }
    assert data.node_types == {
        "X001A": "category",
        "X002B": "category",
        "X000C": "category",
    }
    assert (data.data == abc36["d"]).all().all()

    data.randomise_names()
    print("\n\nRandom names, seed None:\n{}".format(data.as_df().head()))
    assert tuple(data.as_df().columns) == ("A", "B", "C")
    assert data.get_order() == ("A", "B", "C")
    assert data.node_values == {
        "A": {"0": 10, "1": 26},
        "B": {"0": 14, "1": 22},
        "C": {"0": 16, "1": 20},
    }
    assert data.node_types == {
        "A": "category",
        "B": "category",
        "C": "category",
    }
    assert (data.data == abc36["d"]).all().all()
