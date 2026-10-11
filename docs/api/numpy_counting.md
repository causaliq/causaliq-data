# NumPy Adapter - Counting Helpers

Stateless counting helpers used by the [`NumPy`](numpy.md) adapter. They were
extracted into this sibling module so that `causaliq_data.numpy` stays within
the module size limits described in the
[development policy](https://github.com/causaliq/causaliq/blob/main/docs/development_policy/01-code.md).

The adapter passes all state to these functions as arguments, so the
`NumPy.MAX_BINCOUNT` class attribute remains the single place that selects
the fast packing path over the slower NumPy `unique` path.

## Combination Counting

::: causaliq_data.numpy_counting.combinations_from_binning
    options:
        show_root_heading: true
        show_source: false
        heading_level: 3

::: causaliq_data.numpy_counting.crosstab_counts
    options:
        show_root_heading: true
        show_source: false
        heading_level: 3

## Categorical Counts

::: causaliq_data.numpy_counting.categorical_value_counts
    options:
        show_root_heading: true
        show_source: false
        heading_level: 3
