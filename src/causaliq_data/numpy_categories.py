# Category encoding helpers for the NumPy data adapter
#
# Stateless helpers used by causaliq_data.numpy, extracted so that the
# adapter module stays within the module size limits. Behaviour is
# unchanged; they operate on the dataframe passed in.

from typing import Any, Dict, Optional, Tuple

from numpy import ndarray
from pandas import Categorical, DataFrame, factorize


def _unique_values(uniques: Any) -> Any:
    """Return the unique values of a factorized categorical column.

    Handles the case where uniques might be an Index or CategoricalIndex.
    """
    if hasattr(uniques, "categories") and hasattr(uniques, "codes"):
        # CategoricalIndex case
        return uniques.categories[uniques.codes].unique()
    # Regular Index case
    return uniques.unique()


def category_codes(
    df: DataFrame, max_category: int
) -> Dict[str, Optional[Tuple[str, ...]]]:
    """Convert categorical columns to integer codes, in place.

    Captures the code to value mapping as a tuple for each node, i.e.
    ('yes', 'no') implies integer code 0 maps to 'yes', 1 to 'no'.

    Args:
        df: Dataframe whose categorical columns are converted.
        max_category: Maximum number of different category values.

    Returns:
        Code to value mapping for each column.

    Raises:
        ValueError: If a column has too many category values.
    """
    col_values: Dict[str, Optional[Tuple[str, ...]]] = {}
    for col in df.columns:
        df[col], uniques = factorize(df[col])
        if len(uniques) > max_category:
            raise ValueError("data.read() too many categories")
        col_values[col] = tuple(_unique_values(uniques))

    return col_values


def restore_categories(df: DataFrame, categories: ndarray) -> None:
    """Convert integer codes in a dataframe back to categories, in place.

    Args:
        df: Dataframe whose integer codes are converted.
        categories: Category values for each column.
    """
    for j in range(len(df.columns)):
        # Extract the integer codes first to avoid dtype warnings
        codes = df.iloc[:, j].values
        categorical_data = Categorical.from_codes(
            codes,  # type: ignore[arg-type]
            categories=categories[j],
        )
        # Use column assignment instead of iloc to avoid dtype warnings
        df[df.columns[j]] = categorical_data
