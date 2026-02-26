from .estimator import Estimator

class NCRFEstimator(Estimator):
    name: str = "ncrf"

    def __init__(self, *, mu: float = 'auto', n_iter: int = None, n_iterf: int = None, n_iterc: int = None, normalize: bool = True, in_place: bool = True):
        self.mu = mu
        self.n_iter = n_iter
        self.n_iterf = n_iterf
        self.n_iterc = n_iterc
        self.normalize = normalize
        self.in_place = in_place