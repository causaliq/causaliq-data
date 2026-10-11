# refactor causaliq-data modules according to module and method rules

All `causaliq-data` modules were refactored to obey the development policies.

## Constraints learned from other package refactors

- **Behaviour must stay byte-identical**
- **Preserve import binding sites**
- **No behaviour-free bulk test rewrites**: add focused tests
  only where coverage gaps appear

## Commits

Stage the work as **between two and four commits** which the user will do:

1. **Code and tests** — one to three commits
2. **Documentation** — one documentation commit

Do not mix documentation-only changes into the code commit.

## Gates after each slice (as in `scripts/check_ci.ps1`)

- `black --check src tests`
- `isort --check-only src tests`
- `flake8 src tests`
- `mypy src/`
- `python scripts/check_code_rules.py --strict src/causaliq_data`

100 % coverage is maintained on the new modules and the bnlearn score
replication tests in `tests/functional/test_score_bnlearn.py` stay green.

## Outcome

The refactor is complete. `python scripts/check_code_rules.py` reports zero
breaches across all 14 modules in `src/causaliq_data`, the full test suite
passes with 100 % coverage, and behaviour is unchanged.

| Stage | Modules refactored | Breaches reported |
| --- | --- | --- |
| Starting point | – | 19 |
| Slice 1 | `__init__`, `indep`, `pandas` | 13 |
| Slice 2 | `numpy`, `numpy_counting`, `numpy_categories` | 9 |
| Slice 3 | `score`, `score_categorical`, `score_gaussian`, `score_params` | 0 |

### What changed

- Long methods and functions were split into private helpers, keeping every
  method within the statement and nesting depth limits.
- Modules that exceeded the module statement limit were decomposed into
  sibling sub-modules which the original module re-exports, so public API,
  module paths and import sites are unchanged. Details of the decomposition
  are recorded in the [architecture overview](architecture/overview.md).
- Modules whose functions are substituted in tests with
  `monkeypatch.setattr` were flattened in place (`__init__`,
  `indep`, `pandas.read`, `pandas.marginals`, `numpy.read`, `numpy.from_df`,
  `numpy.set_N`, `numpy.marginals`, `numpy.unique` and the whole `score`
  dispatch layer), so patches such as
  `causaliq_data.score.gaussian_node_score`,
  `causaliq_data.score.categorical_node_score`,
  `causaliq_data.numpy.MAX_CATEGORY` and
  `causaliq_data.numpy.NumPy.MAX_BINCOUNT` still intercept the calls.
- Floating point accumulation order was preserved: helpers return
  intermediates and the caller accumulates them in the original sequence
  (BGE, entropy scores, Bayesian scores and the chi-squared/MI statistics).
- No test was rewritten or moved. The unchanged suite still reports 100 %
  coverage for every new and re-structured module.

### Verification

Every slice was checked with `black --check`, `isort --check-only`, `flake8`,
`mypy src/` and the strict rules report, followed by the full test suite at
100 % coverage on Python 3.9 to 3.13 in CI.

| Stage | Statements (coverage) | Coverage |
| --- | --- | --- |
| Starting point | 879 | 100 % |
| Refactored | 1059 | 100 % |

