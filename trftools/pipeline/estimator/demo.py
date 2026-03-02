from pathlib import Path
import numpy as np
from eelbrain import save, NDVar, UTS
from eelbrain.pipeline import RawFilter, PrimaryEpoch, LabelVar

from trftools.pipeline import TRFExperiment, FilePredictor
from trftools.pipeline.estimator import BoostingEstimator, NCRFEstimator


DATA_ROOT = "~/Data/BIDS"  # set to your data root

# Only run sub-01 (exclude the rest so pipeline is fast).
# MNE-BIDS get_entity_vals returns subject *values* (no "sub-" prefix): "01", "02", "emptyroom".
IGNORE_SUBJECTS = [
    "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12",
    "emptyroom",
]


class AppleSeed(TRFExperiment):
    data_dir = "meg"
    subject_re = r"sub-\d+"  # BIDS: sub-01, sub-02, ...
    sessions = ["Appleseed"]  # must match BIDS task name in filenames
    # MNE-BIDS get_entity_vals() expects "ignore_subjects", not "subject"
    ignore_entities = {"ignore_subjects": IGNORE_SUBJECTS}

    raw = {"0.5-20": RawFilter("raw", 0.5, 20, cache=False)}

    # At least one epoch required for load_trf. Task name must match BIDS (e.g. task-Appleseed in filenames).
    epochs = {
        "Appleseed": PrimaryEpoch("Appleseed", None, samplingrate=100),
    }
    defaults = {"epoch": "Appleseed"}

    # Predictor "acoustic_envelop" must exist as derivatives/predictors/{stimulus}~acoustic_envelop.pickle
    predictors = {
        "acoustic_envelop": FilePredictor(),
    }

    # Pipeline needs a "stimulus" column to load predictors. Create it from BIDS "task" (exists in events).
    variables = {
        "stimulus": LabelVar("task", {"Appleseed": "Appleseed", "Tone": "Tone"}),
    }
    tests = {}

    estimators = {
        # partitions required when n_cases not in 3..10 (e.g. 22 trials)
        "boosting": BoostingEstimator(basis=0.050, partitions=5),
        "boosting-l2": BoostingEstimator(error="l2", partitions=5),
        "ncrf": NCRFEstimator(mu=0.1, n_iter=100),
        "ncrf-fast": NCRFEstimator(mu=0.01, n_iter=10),
    }

# example usage of estimators
def example_usage():
    e = AppleSeed(DATA_ROOT)

    # Use named estimator (parameters come from e.estimators['boosting'])
    boosting_trf = e.load_trf("acoustic_envelop", 0, 0.500, estimator="boosting")

    # Different estimator → different cache file
    boosting_l2_trf = e.load_trf("acoustic_envelop", 0, 0.500, estimator="boosting-l2")

    # NCRF estimators (when using source-space / inv)
    # ncrf_trf = e.load_trf("acoustic_envelop", 0, 0.500, estimator="ncrf")
    # ncrf_fast_trf = e.load_trf("acoustic_envelop", 0, 0.500, estimator="ncrf-fast")

# original usage
def legacy_usage():
    e = AppleSeed(DATA_ROOT)
    # All params passed explicitly; no .estimators used
    trf = e.load_trf(
        "acoustic_envelop",
        0,
        0.500,
        basis=0.050,
        delta=0.005,
        error="l1",
        partitions=None,
        cv=True,
    )


def _make_synthetic_ndvar():
    """NDVar from -5000s to +5000s so any epoch tmin/tmax falls inside (avoids pad zeros -> flat)."""
    tstep = 0.01
    tmin = -5000.0
    n = 1000000  # 10000 s total
    rs = np.random.RandomState(42)
    t = tmin + np.arange(n, dtype=np.float64) * tstep
    data = 2.0 + 0.5 * np.sin(2 * np.pi * 0.5 * t) + 0.5 * rs.randn(n)
    uts = UTS(tmin, tstep, n)
    return NDVar(data.astype(np.float64), uts, name="acoustic_envelop")


def _ensure_synthetic_envelope(e):
    """Bypass: write synthetic envelopes to e.get('predictor-dir') for all stimuli. Not for real analysis."""
    pred_dir = Path(e.get("predictor-dir"))
    pred_dir.mkdir(parents=True, exist_ok=True)
    x = _make_synthetic_ndvar()
    for stimulus in ["Appleseed", "Tone"]:
        path = pred_dir / f"{stimulus}~acoustic_envelop.pickle"
        save.pickle(x, path)
    print(f"Wrote synthetic envelopes to {pred_dir} (bypass only).")


def run_sub01_only():
    """Fastest path: only sub-01, one TRF, make=True to compute if missing."""
    e = AppleSeed(DATA_ROOT)
    e.set(subject="01")
    _ensure_synthetic_envelope(e)  # use experiment's predictor-dir
    print("Running TRF for sub-01 (make=True)...")
    # Use sensor space (meg) so no parcellation mask is required
    trf = e.load_trf("acoustic_envelop", 0, 0.500, estimator="ncrf", make=True, data="meg")
    print("Done:", trf)
    return trf


if __name__ == "__main__":
    run_sub01_only()
