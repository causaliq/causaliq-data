# Score Functions - Parameters and Constants

Score parameter constants, validation and graph size metrics used by the
[`score`](score.md) module. They were extracted into this sibling module so
that `causaliq_data.score` stays within the module size limits described in
the
[development policy](https://github.com/causaliq/causaliq/blob/main/docs/development_policy/01-code.md).
Every name is re-exported from `causaliq_data.score`, so import paths are
unchanged.

The score constants live in this module because the score kernels need them,
and importing them from `causaliq_data.score` would create a circular import.

## Parameter Checking

::: causaliq_data.score_params.check_score_params
    options:
        show_root_heading: true
        show_source: false
        heading_level: 3

## Free Parameters

::: causaliq_data.score_params.free_params
    options:
        show_root_heading: true
        show_source: false
        heading_level: 3
