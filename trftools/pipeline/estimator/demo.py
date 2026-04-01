from pathlib import Path
from eelbrain.pipeline import RawFilter, PrimaryEpoch, LabelVar

from trftools.pipeline import TRFExperiment, FilePredictor
from trftools.pipeline.estimator import BoostingEstimator, NCRFEstimator


def _log(msg: str):
    print(f"[demo] {msg}")


def _ensure_demo_predictor(e):
    """Check that the real demo predictor exists."""
    pred_dir = Path(e.get("predictor-dir"))
    target = pred_dir / "Appleseed~acoustic_envelop.pickle"
    if not target.exists():
        raise FileNotFoundError(f"Required predictor file not found: {target}")
    _log(f"Using predictor {target.name}.")


DATA_ROOT = "/Users/yanyuwoo/Data/Appleseed_BIDS_20251216"

# Only run sub-01 (exclude the rest so pipeline is fast).
# MNE-BIDS get_entity_vals returns subject *values* (no "sub-" prefix): "01", "02", "emptyroom".
IGNORE_SUBJECTS = [
    "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12",
    "emptyroom",
]


class AppleSeed(TRFExperiment):
    data_dir = "meg"
    subject_re = r"sub-[A-Za-z0-9]+"  # BIDS subjects, e.g. sub-01 or sub-R2677
    sessions = ["Appleseed"]  # must match BIDS task name in filenames
    # MNE-BIDS get_entity_vals() expects "ignore_subjects", not "subject"
    ignore_entities = {"ignore_subjects": IGNORE_SUBJECTS}

    raw = {"0.5-20": RawFilter("raw", 0.5, 20, cache=False)}

    # At least one epoch required for load_trf. Task name must match BIDS (e.g. task-Appleseed in filenames).
    epochs = {
        "Appleseed": PrimaryEpoch("Appleseed", None, samplingrate=100),
        # Minimal covariance epoch for source-space/NCRF demos.
        "cov": PrimaryEpoch("Appleseed", None, tmin=-0.100, tmax=0.0, samplingrate=100),
    }
    defaults = {"epoch": "Appleseed"}

    # Predictor must exist as derivatives/predictors/{stimulus}~acoustic_envelop.pickle
    predictors = {
        "acoustic_envelop": FilePredictor(),
    }

    models = {
        "acoustic_envelop": "acoustic_envelop",
    }

    # Pipeline needs a "stimulus" column to load predictors. Eelbrain derives
    # events from the raw stimulus channel here, so use trigger codes rather
    # than relying on columns from the BIDS events.tsv.
    variables = {
        "stimulus": LabelVar("trigger", {(162, 167): "Appleseed"}),
    }
    tests = {}

    # Example estimators
    estimators = {
        # partitions required when n_cases not in 3..10 (e.g. 22 trials)
        "boosting": BoostingEstimator(basis=0.050, partitions=5),
        "boosting-l2": BoostingEstimator(error="l2", partitions=5),
        "ncrf": NCRFEstimator(mu=0.1, n_iter=100),
        "ncrf-fast": NCRFEstimator(mu=0.01, n_iter=10),
    }

def _init_demo_experiment() -> AppleSeed:
    e = AppleSeed(DATA_ROOT)
    e._register_model(e._coerce_model("acoustic_envelop"))
    subjects = [s for s in (e._field_values.get("subject") or ()) if s != "emptyroom"]
    if not subjects:
        raise RuntimeError(f"No non-emptyroom subjects found under {DATA_ROOT}")
    subject = subjects[0]
    e.set(subject=subject)
    return e


def example_usage():
    e = _init_demo_experiment()
    _log(f"Example run with DATA_ROOT={DATA_ROOT}")
    _log(f"Predictor dir: {e.get('predictor-dir')}")

    # Use named estimator (parameters come from e.estimators['boosting'])
    _log("Loading TRF with estimator='boosting'...")
    boosting_trf = e.load_trf("acoustic_envelop", 0, 0.500, estimator="boosting")

    # Different estimator → different cache file
    _log("Loading TRF with estimator='boosting-l2'...")
    boosting_l2_trf = e.load_trf("acoustic_envelop", 0, 0.500, estimator="boosting-l2")

    # NCRF estimators are demonstrated by run_ncrf_demo()
    # ncrf_trf = e.load_trf("acoustic_envelop", 0, 0.500, estimator="ncrf")
    # ncrf_fast_trf = e.load_trf("acoustic_envelop", 0, 0.500, estimator="ncrf-fast")
    return boosting_trf, boosting_l2_trf

def _run_demo(estimator: str, **load_trf_kwargs):
    e = _init_demo_experiment()
    _log(f"Initialized experiment with DATA_ROOT={DATA_ROOT}")
    _log(f"Subject={e.get('subject')}, Epoch={e.get('epoch')}, Estimator={estimator!r}, Options={load_trf_kwargs!r}")
    _ensure_demo_predictor(e)
    trf_path = e.load_trf(
        "acoustic_envelop",
        tstart=0,
        tstop=0.500,
        estimator=estimator,
        path_only=True,
        **load_trf_kwargs,
    )
    _log(f"TRF cache path: {trf_path}")
    _log(f"Running TRF with estimator={estimator!r} (make=True)...")
    trf = e.load_trf(
        "acoustic_envelop",
        tstart=0,
        tstop=0.500,
        estimator=estimator,
        make=True,
        **load_trf_kwargs,
    )
    _log(f"Done: {trf}")
    return trf


def run_boosting_demo():
    """Recommended boosting demo using the estimator registry."""
    return _run_demo("boosting")


def run_ncrf_demo():
    """Recommended NCRF demo: let the estimator choose its own valid semantics."""
    return _run_demo("ncrf", data="meg")


def run_ncrf_compat_demo():
    """Backward-compatibility demo for legacy NCRF call sites."""
    return _run_demo("ncrf", data="source", inv="ncrf", mask="aparc")


if __name__ == "__main__":
    run_ncrf_demo()
