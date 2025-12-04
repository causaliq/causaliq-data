"""Abstract data interfaces for causal discovery and BN fitting."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import numpy as np
from causaliq_core.utils.random import (  # type: ignore[import-untyped]
    RandomIntegers,
)
from strenum import StrEnum


class DatasetType(StrEnum):
    """Type of dataset based on variable types."""

    CATEGORICAL = "categorical"  # all categorical variables
    CONTINUOUS = "continuous"  # all float variables
    MIXED = "mixed"  # mixed categorical, float or numeric


class VariableType(StrEnum):
    """Data type for individual variables."""

    INT16 = "int16"
    INT32 = "int32"
    INT64 = "int64"
    FLOAT32 = "float32"
    FLOAT64 = "float64"
    CATEGORY = "category"


class BNFit(ABC):
    """
    Minimal interface for Bayesian Network parameter estimation.

    This interface provides the essential methods required for fitting
    conditional probability tables (CPT) and linear Gaussian models
    in Bayesian Networks.
    """

    @abstractmethod
    def marginals(
        self, node: str, parents: Dict, values_reqd: bool = False
    ) -> Tuple:
        """
        Return marginal counts for a node and its parents.

        :param str node: node for which marginals required
        :param dict parents: {node: parents} parents of non-orphan nodes
        :param bool values_reqd: whether parent and child values required

        :raises TypeError: for bad argument types

        :returns tuple: of counts, and optionally, values:
                        - ndarray counts: 2D, rows=child, cols=parents
                        - int maxcol: maximum number of parental values
                        - tuple rowval: child values for each row
                        - tuple colval: parent combo (dict) for each col
        """
        pass

    @abstractmethod
    def values(self, nodes: Tuple[str, ...]) -> np.ndarray:
        """
        Return the (float) values for the specified set of nodes.

        Suitable for passing into e.g. linearRegression fitting function

        :param tuple nodes: nodes for which data required

        :raises TypeError: if bad arg type
        :raises ValueError: if bad arg value

        :returns ndarray: Numpy array of values, each column for a node
        """
        pass

    @property
    @abstractmethod
    def N(self) -> int:
        """
        Total sample size.

        :returns int: current sample size being used
        """
        pass

    @property
    @abstractmethod
    def node_values(self) -> Dict[str, Dict]:
        """
        Node value counts for categorical variables.

        :returns dict: values and their counts of categorical nodes
                       in sample {n1: {v1: c1, v2: ...}, n2 ...}
        """
        pass


class Data(BNFit):
    """
    Top level data object for structure learning.

    Extends BNFit interface with additional methods needed for
    causal structure learning algorithms.

    :param DataFrame/None df: data supplied as Pandas DataFrame
    :param BN/None bn: data specified as BN (Oracle data)
    :param int/None N: sample size

    :ivar tuple nodes: internal (i.e. original) node names
    :ivar tuple order: order in which nodes should be processed
    :ivar dict ext_to_orig: map from external to original names
    :ivar dict orig_to_ext: map from original to external names
    :ivar int N: current sample size being used by the algorithm
    :ivar dict node_types: node types {n1: t1, n2: ....}

    :raises TypeError: if bad types supplied
    """

    elapsed: float = 0.0
    nodes: Tuple[str, ...]
    order: Tuple[int, ...]
    ext_to_orig: Dict[str, str]
    orig_to_ext: Dict[str, str]
    node_types: Dict[str, str]
    dstype: str

    def __init__(self) -> None:
        pass

    def _update_sample(self) -> None:
        """Update sample - implemented in subclasses."""
        pass

    def set_order(self, order: Tuple[str, ...]) -> None:
        """
        Set the process order of the nodes to specified one.

        :param tuple order: new process order

        :raises TypeError: for bad argument types
        :raises ValueError: for bad argument values
        """
        if not isinstance(order, tuple) or any(
            not isinstance(n, str) for n in order
        ):
            raise TypeError("Data.set_order() bad arg type")

        if set(order) != set(self.ext_to_orig.keys()):
            raise ValueError("Data.set_order() bad arg value")

        self.order = tuple(
            self.nodes.index(self.ext_to_orig[n]) for n in order
        )
        if self.__class__.__name__ == "Pandas":
            self._update_sample()

    def get_order(self) -> Tuple[str, ...]:
        """
        Get the current process order.

        :returns tuple: of external names of nodes in process order
        """
        return tuple(self.orig_to_ext[self.nodes[i]] for i in self.order)

    def randomise_order(self, seed: int) -> None:
        """
        Randomise the process order of the nodes.

        :param int seed: randomisation seed

        :raises TypeError: for bad argument types
        :raises ValueError: for bad argument values
        """
        if not isinstance(seed, int):
            raise TypeError("Data.randomise_order() bad arg type")

        if seed < 0:
            raise ValueError("Data.randomise_order() bad arg value")

        self.order = tuple(RandomIntegers(len(self.nodes), seed))
        if self.__class__.__name__ == "Pandas":
            self._update_sample()

    def _set_dstype(self) -> None:
        """Determine overall dataset type from individual node types."""
        n_floats = sum(
            1 if v in {"float32", "float64"} else 0
            for v in self.node_types.values()
        )
        n_cats = sum(
            1 if v == "category" else 0 for v in self.node_types.values()
        )
        self.dstype = (
            "continuous"
            if n_floats == len(self.nodes)
            else ("categorical" if n_cats == len(self.nodes) else "mixed")
        )

    def _generate_random_names(self, seed: Optional[int]) -> None:
        """Generate randomised external names for nodes."""
        if seed is None:
            self.ext_to_orig = {n: n for n in self.nodes}
            self.orig_to_ext = {n: n for n in self.nodes}
        else:
            ints = [i for i in RandomIntegers(len(self.nodes), seed)]
            self.ext_to_orig = {
                "X{:03d}{}".format(ints[i], n[:6]): n
                for i, n in enumerate(self.nodes)
            }
            self.orig_to_ext = {
                orig: ext for ext, orig in self.ext_to_orig.items()
            }

    @abstractmethod
    def set_N(
        self,
        N: int,
        seed: Optional[int] = None,
        random_selection: bool = False,
    ) -> None:
        """
        Set current working sample size.

        :param int N: current working sample size
        :param int/None seed: seed for row order randomisation if reqd
        :param bool random_selection: whether rows selected is also
                                      randomised

        :raises TypeError: if bad argument type
        :raises ValueError: if bad argument value
        """
        pass

    @abstractmethod
    def randomise_names(self, seed: Optional[int]) -> None:
        """
        Randomise the node names that the learning algorithm uses.

        So sensitivity to these names can be assessed.

        :param int/None seed: randomisation seed (if None, names revert
                              back to original names)

        :raises TypeError: for bad argument types
        :raises ValueError: for bad argument values
        """
        pass

    @abstractmethod
    def as_df(self) -> Any:
        """
        Return the data as a Pandas dataframe.

        Returns data with current sample size and column order.

        :returns DataFrame: data as Pandas
        """
        pass
