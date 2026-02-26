
from trftools.pipeline import TRFExperiment
from trftools.pipeline.estimator import BoostingEstimator, NCRFEstimator


DATA_ROOT = "~/Data/Alice"  # set to your data root or use a test path


class Alice(TRFExperiment):
    data_dir = "eeg"
    subject_re = r"S\d\d"
    sessions = ["alice"]

    raw = {"0.5-20": None}  # placeholder; use RawFilter in real setup

    variables = {}
    tests = {}

    # Named estimators: use these keys in load_trf(..., estimator='...')
    estimators = {
        "boosting": BoostingEstimator(basis=0.050),
        "boosting-l2": BoostingEstimator(error="l2", partitions=5),
        "ncrf": NCRFEstimator(mu=0.1, n_iter=100),
        "ncrf-fast": NCRFEstimator(mu=0.01, n_iter=10),
    }

# example usage of estimators
def example_usage():
    e = Alice(DATA_ROOT)

    # Use named estimator (parameters come from e.estimators['boosting'])
    boosting_trf = e.load_trf("acoustic_envelop", 0, 0.500, estimator="boosting")

    # Different estimator → different cache file
    boosting_l2_trf = e.load_trf("acoustic_envelop", 0, 0.500, estimator="boosting-l2")

    # NCRF estimators (when using source-space / inv)
    # ncrf_trf = e.load_trf("acoustic_envelop", 0, 0.500, estimator="ncrf")
    # ncrf_fast_trf = e.load_trf("acoustic_envelop", 0, 0.500, estimator="ncrf-fast")

# original usage
def legacy_usage():
    e = Alice(DATA_ROOT)
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


if __name__ == "__main__":
    print("Estimator demo – define Alice and call example_usage() with real DATA_ROOT to run.")
    # example_usage()
