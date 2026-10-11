# Score Functions - Categorical Kernels

Categorical score kernels used by the [`score`](score.md) module. They were
extracted into this sibling module so that `causaliq_data.score` stays within
the module size limits described in the
[development policy](https://github.com/causaliq/causaliq/blob/main/docs/development_policy/01-code.md).
Every name is re-exported from `causaliq_data.score`, so import paths are
unchanged.

## Bayesian Scores

::: causaliq_data.score_categorical.bayesian_score
    options:
        show_root_heading: true
        show_source: false
        heading_level: 3

## Entropy Scores

::: causaliq_data.score_categorical.entropy_scores
    options:
        show_root_heading: true
        show_source: false
        heading_level: 3
