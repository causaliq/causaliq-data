# NumPy Adapter - Category Helpers

Category encoding and decoding helpers used by the [`NumPy`](numpy.md)
adapter. They were extracted into this sibling module so that
`causaliq_data.numpy` stays within the module size limits described in the
[development policy](https://github.com/causaliq/causaliq/blob/main/docs/development_policy/01-code.md).

The category to code mapping is captured when data is read in, and applied in
reverse when a dataframe is requested, so round trips through
`NumPy.from_df()` and `NumPy.as_df()` are unchanged.

## Encoding Categories

::: causaliq_data.numpy_categories.category_codes
    options:
        show_root_heading: true
        show_source: false
        heading_level: 3

## Decoding Categories

::: causaliq_data.numpy_categories.restore_categories
    options:
        show_root_heading: true
        show_source: false
        heading_level: 3
