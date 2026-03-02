from eelbrain.pipeline import RawFilter

from trftools.pipeline import TRFExperiment
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
    sessions = ["alice"]
    # MNE-BIDS get_entity_vals() expects "ignore_subjects", not "subject"
    ignore_entities = {"ignore_subjects": IGNORE_SUBJECTS}

    raw = {"0.5-20": RawFilter("raw", 0.5, 20, cache=False)}

    variables = {}
    tests = {}

    estimators = {
        "boosting": BoostingEstimator(basis=0.050),
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


def run_sub01_only():
    """Fastest path: only sub-01, one TRF, make=True to compute if missing."""
    e = AppleSeed(DATA_ROOT)
    e.set(subject="01")
    print("Running TRF for sub-01 (make=True)...")
    trf = e.load_trf("acoustic_envelop", 0, 0.500, estimator="boosting", make=True)
    print("Done:", trf)
    return trf


if __name__ == "__main__":
    run_sub01_only()
