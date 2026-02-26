# Base class for TRF estimators
from typing import Any, Dict


class Estimator:
    """Base for TRF estimator configs. Subclasses override parameters_for_partial()."""

    name: str = ""

    def parameters_for_partial(self) -> Dict[str, Any]:
        """
        Return kwargs for partial(fitter, ..., **kwargs).
        Subclasses override this to return their estimator-specific parameters.
        """
        return {}