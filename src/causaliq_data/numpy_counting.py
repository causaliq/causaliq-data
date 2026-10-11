# Counting helpers for the NumPy data adapter
#
# Stateless helpers used by causaliq_data.numpy, extracted so that the
# adapter module stays within the module size limits. Behaviour is
# unchanged; the adapter passes all state in as arguments.

from math import prod
from typing import Dict, Tuple

from numpy import array, bincount, empty, ndarray, nonzero
from numpy import unique as npunique
from numpy import zeros


def combinations_from_binning(
    sample_cols: ndarray,
    num_vals: ndarray,
    n_reqd: int,
    minlength: int,
) -> Tuple[ndarray, ndarray]:
    """Count unique combinations of values by packing them into integers.

    Args:
        sample_cols: Sample values for the required columns.
        num_vals: Number of values in each of those columns.
        n_reqd: Number of columns required.
        minlength: Maximum possible number of value combinations.

    Returns:
        Tuple of unique value combinations and their counts.
    """
    multipliers = array([prod((num_vals[:i]).tolist()) for i in range(n_reqd)])
    packed = sample_cols @ multipliers

    # Count the frequency of unique packed integers, removing all
    # entries with zero counts

    counts = bincount(packed, minlength=minlength)
    packed = nonzero(counts)[0]
    counts = counts[packed]

    # Unpack integers back into their combinations of values using
    # the same multipliers used to pack them into one integer

    combos: ndarray = empty((len(packed), len(multipliers)), dtype=int)
    for jj, r in enumerate(reversed(multipliers)):
        combos[:, len(multipliers) - jj - 1] = packed // r
        packed = packed % r

    return combos, counts


def categorical_value_counts(
    sample: ndarray,
    categories: ndarray,
    nodes: Tuple[str, ...],
    orig_to_ext: Dict[str, str],
) -> Dict[str, Dict[str, int]]:
    """Return value counts for each categorical column of a sample.

    Args:
        sample: Sample data array.
        categories: Category values for each column.
        nodes: Internal (i.e. original) node names.
        orig_to_ext: Map from original to external names.

    Returns:
        Value counts for each categorical node.
    """
    node_values: Dict[str, Dict[str, int]] = {}
    for j in range(sample.shape[1]):
        counts = {
            categories[j][v]: int(c)
            for v, c in enumerate(bincount(sample[:, j]))
        }
        counts = {v: counts[v] for v in sorted(counts)}
        node_values[orig_to_ext[nodes[j]]] = counts

    return node_values


def crosstab_counts(
    combos: ndarray, combo_counts: ndarray, n_child_values: int
) -> Tuple[ndarray, ndarray]:
    """Return a counts matrix of child values against parental combos.

    Args:
        combos: Unique value combinations, with child value first.
        combo_counts: Count of each unique value combination.
        n_child_values: Number of distinct child values.

    Returns:
        Tuple of counts matrix and unique parental combinations.
    """
    p_combos = npunique(combos[:, 1:], axis=0)
    c_value_to_i = {v: i for i, v in enumerate(range(n_child_values))}
    p_combo_to_j = {tuple(c): j for j, c in enumerate(p_combos)}
    counts = zeros((n_child_values, len(p_combos)), dtype="int32")
    for idx, (c_value, *p_combo) in enumerate(combos):
        counts[c_value_to_i[c_value], p_combo_to_j[tuple(p_combo)]] = (
            combo_counts[idx]
        )

    return counts, p_combos
