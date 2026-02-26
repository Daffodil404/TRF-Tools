"""
Boosting estimator for TRF pipeline.

Use in experiment definition:

    estimators = {
        'boosting': BoostingEstimator(basis=0.050),
        'boosting-l2': BoostingEstimator(error='l2', partitions=5),
    }
    # Then: e.load_trf('acoustic_envelop', 0, 0.5, estimator='boosting')

tstart & tstop are parameters of load_trf/load_trfs, not of the estimator.
"""

from typing import Any, Dict, Optional
from .estimator import Estimator

class BoostingEstimator(Estimator):
    def __init__(
        self,
        *,
        delta: float = 0.005,
        mindelta: Optional[float] = None,
        error: str = "l1",
        basis: float = 0.050,
        partitions: Optional[int] = None,
        cv: bool = True,
        selective_stopping: int = 0,
        partition_results: bool = False,
        name: str = "",
    ):
        self.delta = delta
        self.mindelta = mindelta
        self.error = error
        self.basis = basis
        self.partitions = partitions
        self.cv = cv
        self.selective_stopping = selective_stopping
        self.partition_results = partition_results
        self.name = name or "boosting"

    # def to_partial_kwargs(self) -> Dict[str, Any]:
    #     """Keyword arguments for partial(boosting, y, xs, tstart, tstop, 'inplace', delta, mindelta, error, basis, **kwargs)."""
    #     return {
    #         "partitions": self.partitions,
    #         "test": self.cv,
    #         "selective_stopping": self.selective_stopping,
    #         "partition_results": self.partition_results,
    #     }

    # def positional_args_for_partial(self) -> tuple:
    #     """Positional args after (y, xs, tstart, tstop): ('inplace', delta, mindelta, error, basis)."""
    #     return ("inplace", self.delta, self.mindelta, self.error, self.basis)
