from pathlib import Path
import numpy as np
from eelbrain import save, NDVar, UTS
from eelbrain.pipeline import RawFilter, PrimaryEpoch, LabelVar

from trftools.pipeline import TRFExperiment, FilePredictor
from trftools.pipeline.estimator import BoostingEstimator, NCRFEstimator


def _log(msg: str):
    print(f"[demo] {msg}")


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
    for stimulus in ["Appleseed"]:
        path = pred_dir / f"{stimulus}~acoustic_envelop.pickle"
        save.pickle(x, path)
    _log(f"Wrote synthetic envelopes to {pred_dir} (bypass only).")

def _ensure_unsplit_meg(root: str, subject: str):
    """Create symlinks for split FIF files so BIDS lookup can find unsplit names."""
    meg_dir = Path(root) / f"sub-{subject}" / "meg"
    if not meg_dir.exists():
        return
    for split in meg_dir.glob(f"sub-{subject}_task-*_run-*_split-01_meg.fif"):
        unsplit = Path(str(split).replace("_split-01_meg.fif", "_meg.fif"))
        if unsplit.exists():
            continue
        try:
            unsplit.symlink_to(split.name)
            _log(f"Linked {unsplit.name} -> {split.name}")
        except OSError:
            # If symlink fails (e.g., permissions), fall back silently
            pass


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

    # Predictor "acoustic_envelop" must exist as derivatives/predictors/{stimulus}~acoustic_envelop.pickle
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

# example usage of estimators
def example_usage():
    e = AppleSeed(DATA_ROOT)
    e._register_model(e._coerce_model("acoustic_envelop"))
    _log(f"Example run with DATA_ROOT={DATA_ROOT}")
    _log(f"Predictor dir: {e.get('predictor-dir')}")

    # Use named estimator (parameters come from e.estimators['boosting'])
    _log("Loading TRF with estimator='boosting'...")
    boosting_trf = e.load_trf("acoustic_envelop", 0, 0.500, estimator="boosting")

    # Different estimator → different cache file
    _log("Loading TRF with estimator='boosting-l2'...")
    boosting_l2_trf = e.load_trf("acoustic_envelop", 0, 0.500, estimator="boosting-l2")

    # NCRF estimators (when using source-space / inv)
    # ncrf_trf = e.load_trf("acoustic_envelop", 0, 0.500, estimator="ncrf")
    # ncrf_fast_trf = e.load_trf("acoustic_envelop", 0, 0.500, estimator="ncrf-fast")
    return boosting_trf, boosting_l2_trf

def estimator_pipeline_usage(estimator: str = "boosting"):
    """Run TRF with estimator config from AppleSeed.estimators.

    Note: NCRF requires source-space data and a valid inverse (inv).
    """
    e = AppleSeed(DATA_ROOT)
    e._register_model(e._coerce_model("acoustic_envelop"))
    subjects = [s for s in (e._field_values.get("subject") or ()) if s != "emptyroom"]
    if not subjects:
        raise RuntimeError(f"No non-emptyroom subjects found under {DATA_ROOT}")
    subject = subjects[0]
    e.set(subject=subject)
    _log(f"Initialized experiment with DATA_ROOT={DATA_ROOT}")
    # Bypass for split FIFs: ensure unsplit filenames exist for BIDS lookup
    # _ensure_unsplit_meg(DATA_ROOT, subject)
    data = "meg"
    inv = None
    mask = None
    if estimator.startswith("ncrf"):
        data = "source"
        inv = "ncrf"
        parcs = sorted(getattr(e, "_parcs", {}).keys())
        if parcs:
            mask = "aparc" if "aparc" in parcs else parcs[0]
        else:
            _log("No parcellations available; NCRF requires a source-space mask (parc).")
            _log("Please create a parcellation or set one in the experiment, then retry.")
            return None
    _log(f"Subject={e.get('subject')}, Epoch={e.get('epoch')}, Data={data!r}, inv={inv!r}, mask={mask!r}")
    _ensure_synthetic_envelope(e)
    trf_path = e.load_trf(
        "acoustic_envelop",
        tstart=0,
        tstop=0.500,
        estimator=estimator,
        data=data,
        inv=inv,
        mask=mask,
        path_only=True,
    )
    _log(f"TRF cache path: {trf_path}")
    _log(f"Running TRF for sub-01 with estimator={estimator!r} (make=True, data={data!r}, inv={inv!r})...")
    trf = e.load_trf(
        "acoustic_envelop",
        tstart=0,
        tstop=0.500,
        estimator=estimator,
        make=True,
        data=data,
        inv=inv,
        mask=mask,
    )
    _log(f"Done: {trf}")
    return trf


def run_sub01_only():
    """Fast path using named estimator from the experiment."""
    # NCRF requires source-space data and a valid inverse (inv).
    return estimator_pipeline_usage("ncrf")


if __name__ == "__main__":
    run_sub01_only()
