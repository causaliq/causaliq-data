"""
causaliq-data: Data handling for causal discovery and BN fitting.
"""

from causaliq_data.data import Data, DatasetType, VariableType
from causaliq_data.numpy import NumPy
from causaliq_data.oracle import Oracle
from causaliq_data.pandas import Pandas

__version__ = "0.4.0.dev1"
__author__ = "CausalIQ"
__email__ = "info@causaliq.com"

# Package metadata
__title__ = "causaliq-data"
__description__ = "Data handling for causal discovery and BN fitting"

__url__ = "https://github.com/causaliq/causaliq-data"
__license__ = "MIT"


def _parse_version(version_str: str) -> tuple:
    """Parse version string to tuple, handling pre-release identifiers.

    Extracts numeric parts from version strings like "0.3.0" or "0.3.0.dev1".
    Pre-release identifiers (dev, alpha, beta, rc) are ignored for the tuple.

    Args:
        version_str: Version string (e.g., "0.3.0.dev1").

    Returns:
        Tuple of integers (e.g., (0, 3, 0)).
    """
    parts = []
    for part in version_str.split("."):
        try:
            parts.append(int(part))
        except ValueError:
            break
    return tuple(parts)


# Version tuple for programmatic access
VERSION = _parse_version(__version__)

__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "VERSION",
    "Data",
    "DatasetType",
    "VariableType",
    "Pandas",
    "NumPy",
    "Oracle",
]
