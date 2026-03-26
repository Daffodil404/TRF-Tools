from .estimator import Estimator
from typing import Dict, Any


class NCRFEstimator(Estimator):
    name: str = "ncrf"

    def __init__(self, *, mu: float = 'auto', n_iter: int = None, n_iterf: int = None, n_iterc: int = None, normalize: bool = True, in_place: bool = True):
        self.mu = mu
        self.n_iter = n_iter
        self.n_iterf = n_iterf
        self.n_iterc = n_iterc
        self.normalize = normalize
        self.in_place = in_place
    
    def parameters_for_partial(self) -> Dict[str, Any]:
        params = {
            "mu": self.mu,
            "n_iter": self.n_iter,
            "n_iterf": self.n_iterf,
            "n_iterc": self.n_iterc,
            "normalize": self.normalize,
            "in_place": self.in_place,
        }
        return {key: value for key, value in params.items() if value is not None}
