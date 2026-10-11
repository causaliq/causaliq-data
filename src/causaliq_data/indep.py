#
#   Functions providing probabilistic independence tests
#

from typing import Any, Dict, List, Optional, Tuple, Union, cast

from causaliq_core.bn import BN
from numpy import log
from pandas import DataFrame, crosstab
from scipy import stats  # type: ignore

from causaliq_data.pandas import Pandas

TESTS = ["mi", "x2"]
TEST_PARAMS = {"alpha": 0.05}

MIN_P_VALUE = 1e-30  # p-values below this are considered to be zero


def check_test_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Checks independence test parameters and sets defaults.

    Args:
        params (dict): Test parameters to check, supports:
            - alpha: float test p-value threshold

    Returns:
        dict: Of test params and values including defaults.

    Raises:
        TypeError: If test_params have wrong type.
        ValueError: If test_params have wrong values.
    """
    params = params.copy()
    defaults = {p: v for p, v in TEST_PARAMS.items() if p not in params}
    params.update(defaults)

    if not isinstance(params["alpha"], float):
        raise TypeError("test alpha not a float")

    if params["alpha"] < 1e-10 or params["alpha"] > (1 - 1e-10):
        raise ValueError("Invalid value for test alpha")

    return params


def _accumulate_results(
    actuals: List[List[int]],
    types: List[str],
    results: Dict[str, Dict[str, Union[int, float]]],
) -> Dict[str, Dict[str, Union[int, float]]]:
    """Accumulate test statistics for one contingency table."""
    for type in types:
        df, statistic = _statistic(actuals, type)
        results[type]["df"] += df
        results[type]["statistic"] += statistic
    return results


def _combination_row(
    contingency: DataFrame,
    col: Any,
    missing_y_values: Optional[List[int]],
) -> Optional[List[int]]:
    """Return the contingency row for a column, or the missing values."""
    if col in contingency.columns:
        return list(contingency[col].to_dict().values())
    return missing_y_values


def _actuals_for_combination(
    contingency: DataFrame,
    pv: Any,
    y_values: Any,
    missing_y_values: Optional[List[int]],
) -> List[List[int]]:
    """Return the contingency rows for one conditioning value combo."""
    actuals = []
    for y in y_values:
        col = (pv, y) if isinstance(pv, str) else pv + (y,)
        row = _combination_row(contingency, col, missing_y_values)
        if row is not None:
            actuals.append(row)
    return actuals


def _conditional_results(
    contingency: DataFrame,
    z: List[str],
    y_values: Any,
    missing_y_values: Optional[List[int]],
    types_list: List[str],
    results: Dict[str, Dict[str, Union[int, float]]],
) -> Dict[str, Dict[str, Union[int, float]]]:
    """Accumulate statistics for each conditioning value combination."""
    pvs = contingency.columns.to_frame(index=False).groupby(z).size().index
    for pv in pvs:
        actuals = _actuals_for_combination(
            contingency, pv, y_values, missing_y_values
        )
        results = _accumulate_results(actuals, types_list, results)
    return results


def _unconditional_results(
    contingency: DataFrame,
    types_list: List[str],
    results: Dict[str, Dict[str, Union[int, float]]],
) -> Dict[str, Dict[str, Union[int, float]]]:
    """Accumulate statistics for the whole contingency table."""
    actuals = [list(contingency.iloc[i]) for i in range(len(contingency))]
    return _accumulate_results(actuals, types_list, results)


def _contingency_from_data(
    data: DataFrame, x: str, y: str, z: List[str]
) -> Tuple[DataFrame, Any, List[int]]:
    """Return contingency table and y values derived from data.

    All required contingency tables are generated in one go: rows are values
    of x, columns are a multi-series with different conditioning values,
    then y value.
    """
    cols = [data[col] for col in z]
    cols.append(data[y])
    contingency = crosstab(data[x], cols)

    y_values = data[y].unique()  # used to order y values consistently
    missing_y_values: List[int] = [0] * len(
        data[x].unique()
    )  # filler for missing y values
    return contingency, y_values, missing_y_values


def _contingency_from_bn(
    bn: BN, x: str, y: str, z: List[str], N: int
) -> Tuple[DataFrame, Any, List[int]]:
    """Return contingency table and y values derived from BN parameters."""
    contingency = (
        bn.marginal_distribution(x, sorted(z) + [y])
        .apply(lambda x: round(N * x))
        .astype(int)
    )
    y_values = bn.cnds[y].node_values()
    # Force typing consistency - won't be used in BN case
    missing_y_values = [0] * len(bn.dag.nodes)
    return contingency, y_values, missing_y_values


def _add_p_values(
    results: Dict[str, Dict[str, Union[int, float]]],
    types_list: List[str],
) -> None:
    """Add the p-value of each statistic to the results."""
    for type in types_list:
        p_value = 1.0 - stats.chi2.cdf(
            results[type]["statistic"], results[type]["df"]
        )
        results[type]["p_value"] = p_value if p_value > MIN_P_VALUE else 0.0


def indep(
    x: str,
    y: str,
    z: Union[str, List[str], None],
    data: Union[DataFrame, str, None],
    bn: Optional[BN] = None,
    N: int = 1000000000,
    types: Union[str, List[str]] = "mi",
) -> DataFrame:
    """Checks whether variables x and y are independent.

    Optionally conditionally on z variables, in specified data or using
    BN parameters.

    Args:
        x (str): Name of first variable.
        y (str): Name of second variable.
        z (str/list/None): Name(s) of any conditioning variables.
        data (DataFrame/str/None): Data or file containing data.
        bn (BN/None): Bayesian Network to use parameters from.
        N (int): Sample size to use with BN parameters.
        types (str/list): Type/types of independence test to perform.

    Returns:
        DataFrame: Of independence test results, columns are test names,
            rows are statistic, deg. of freedom, p-value.

    Raises:
        TypeError: If arg types incorrect.
        ValueError: If arg values invalid.
        FileNotFoundError: If data file not found.
    """
    z, data, _, types_list = check_indep_args(x, y, z, data, bn, N, types)

    if data is not None:
        contingency, y_values, missing_y_values = _contingency_from_data(
            data, x, y, z
        )
    else:
        assert bn is not None  # For mypy - we know bn is not None here
        contingency, y_values, missing_y_values = _contingency_from_bn(
            bn, x, y, z, N
        )

    results: Dict[str, Dict[str, Union[int, float]]] = {
        t: {"df": 0, "statistic": 0.0} for t in types_list
    }

    # accumulate statistics, conditioning on z if provided

    if len(z):  # conditional independence
        results = _conditional_results(
            contingency, z, y_values, missing_y_values, types_list, results
        )
    else:  # unconditional independence
        results = _unconditional_results(contingency, types_list, results)

    # compute the p-value for each statistic

    _add_p_values(results, types_list)

    # return independence test results as a DataFrame

    return DataFrame.from_dict(results)  # type: ignore[no-any-return]


def _check_indep_actuals(actuals: List[List[int]], type: str) -> None:
    """Check the contingency table and statistic type arguments."""
    if (
        not isinstance(actuals, list)
        or not isinstance(type, str)
        or any([not isinstance(a, list) for a in actuals])
    ):
        raise TypeError("_indep_tests bad arg types")

    if any([len(a) != len(actuals[0]) for a in actuals]):
        raise TypeError("_indep_tests misshapen actuals arg")


def _cell_contribution(actual: int, expected: float, type: str) -> float:
    """Return the statistic contribution of one contingency table cell."""
    if expected == 0.0:
        return 0.0

    if type == "x2":
        return ((actual - expected) * (actual - expected)) / expected

    if type == "mi" and actual != 0:
        return cast(float, 2.0 * actual * log(actual / expected))

    return 0.0


def _statistic(actuals: List[List[int]], type: str) -> Tuple[int, float]:
    """Returns variance (from independence) statistic.

    Returns chi-squared or MI for a 2 variable contingency table.

    Args:
        actuals (list): Array of frequency counts of variables as a list
            of lists (i.e. contingency table).
        type (str): Type of variance statistic to generate e.g. x2 or mi.

    Returns:
        tuple: (degrees of freedom, variance statistic).

    Raises:
        TypeError: For any illegal argument types.
    """
    _check_indep_actuals(actuals, type)

    x_cardinality = len(actuals)
    y_cardinality = len(actuals[0])

    # print('{}x{} actuals table'.format(x_cardinality, y_cardinality))
    N = sum([sum(a) for a in actuals])
    df = (x_cardinality - 1) * (y_cardinality - 1)
    if N == 0:  # no counts at all in table, assume independence
        return (df, 0.0)

    statistic = 0.0
    for i in range(x_cardinality):
        y_marginal = sum(actuals[i])
        for j in range(y_cardinality):
            x_marginal = sum([actuals[i][j] for i in range(x_cardinality)])
            expected = x_marginal * y_marginal / N
            # print('Actual {}, expected {}'.format(actual, expected))
            statistic += _cell_contribution(actuals[i][j], expected, type)

    return (df, statistic)


def _check_indep_arg_types(
    x: str,
    y: str,
    z: Union[str, List[str], None],
    types: Union[str, List[str]],
    bn: Optional[BN],
    data: Union[DataFrame, str, None],
    N: int,
) -> None:
    """Check the primary argument types for an independence test."""
    if (
        not isinstance(x, str)
        or not isinstance(y, str)
        or (
            not (
                isinstance(z, list) and all([isinstance(_z, str) for _z in z])
            )
        )
        or (
            not (
                isinstance(types, list)
                and all([isinstance(t, str) for t in types])
            )
        )
        or (bn is not None and data is not None)
        or (
            bn is None
            and (not isinstance(data, DataFrame) and not isinstance(data, str))
        )
        or (
            data is None
            and (
                not isinstance(bn, BN)
                or not isinstance(N, int)
                or isinstance(N, bool)
            )
        )
    ):
        raise TypeError("bnlearn_indep called with bad arg types")


def _indep_data_from_args(
    data: Union[DataFrame, str, None],
) -> Tuple[Optional[DataFrame], Optional[str]]:
    """Return processed data and file path from the data argument."""
    data_file_path: Optional[str] = (
        None if isinstance(data, DataFrame) else data
    )
    processed_data: Optional[DataFrame] = None
    if data_file_path:
        processed_data = Pandas.read(data_file_path, dstype="categorical").df
    elif isinstance(data, DataFrame):
        processed_data = data
    return processed_data, data_file_path


def _check_indep_variables(
    varlist: List[str],
    processed_data: Optional[DataFrame],
    bn: Optional[BN],
) -> None:
    """Check the variables tested exist in the data or the BN."""
    if len(varlist) != len(set(varlist)):
        raise ValueError("Duplicate variable names")

    if processed_data is not None:
        if any([v not in processed_data.columns for v in varlist]):
            raise ValueError("Variables not in data")
    else:
        if bn is not None and any([v not in bn.dag.nodes for v in varlist]):
            raise ValueError("Variables not in BN")


def _check_indep_tests(N: int, types: List[str]) -> None:
    """Check the sample size and requested test names."""
    if N < 0:
        raise ValueError("indep() sample size is negative")

    if (
        not len(types)
        or len(types) != len(set(types))
        or any([t not in TESTS for t in types])
    ):
        raise ValueError("Empty, duplicated or unsupported tests specified")


def check_indep_args(
    x: str,
    y: str,
    z: Union[str, List[str], None],
    data: Union[DataFrame, str, None],
    bn: Optional[BN] = None,
    N: int = 0,
    types: Union[str, List[str]] = "mi",
) -> Tuple[List[str], Optional[DataFrame], Optional[str], List[str]]:
    """Check arguments supplied for independence test functions.

    Args:
        x (str): Name of first variable.
        y (str): Name of second variable.
        z (str/list/None): Name(s) of any conditioning variables.
        data (DataFrame/str): Data or file containing data.
        bn (BN/None): Bayesian Network to use parameters from.
        N (int): Sample size to use with BN parameters.
        types (str/list): Type/types of independence test to perform.

    Returns:
        tuple: Of standardised arguments: z as a list, data as a
            DataFrame, data_file as string or None, and types as list.

    Raises:
        TypeError: If arg types incorrect.
        ValueError: If arg values invalid.
        FileNotFoundError: If data file not found.
    """

    # check primary arg types

    z = [] if z is None else ([z] if isinstance(z, str) else z)
    types = [types] if isinstance(types, str) else types
    _check_indep_arg_types(x, y, z, types, bn, data, N)

    # attempt to read in data if data file specified

    processed_data, data_file_path = _indep_data_from_args(data)

    # check no duplicate variable names across x, y and z, and that the
    # variables are present in the data or BN

    varlist = [x] + [y] + z
    _check_indep_variables(varlist, processed_data, bn)

    # check the sample size and requested test names are valid

    _check_indep_tests(N, types)

    return (z, processed_data, data_file_path, types)
