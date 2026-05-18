from pathlib import Path
from eelbrain import load, combine
from eelbrain.pipeline import RawFilter, PrimaryEpoch, LabelVar

from trftools.pipeline import TRFExperiment, FilePredictor
from trftools.pipeline.estimator import BoostingEstimator, NCRFEstimator


def _log(msg: str):
    print(f"[demo] {msg}")


DIR = Path(__file__).parent
STIMULI_LENGTHS = load.tsv(DIR / "appleseed_stimuli.txt", types="fv")
STIMULI_PILOTS = STIMULI_LENGTHS.sub("stimulus != '11'").repeat(2)
STIMULI_REAL = STIMULI_LENGTHS.sub("stimulus != '11b'").repeat(2)


def _stimuli_for_subject(subject: str):
    return STIMULI_PILOTS if subject in ("R2650", "R2652") else STIMULI_REAL


def _ensure_demo_predictor(e):
    """Check that per-stimulus demo predictors exist."""
    pred_dir = Path(e.get("predictor-dir"))
    stimuli = _stimuli_for_subject(e.get("subject"))
    missing = []
    for stimulus in stimuli["stimulus"].cells:
        target = pred_dir / f"{stimulus}~acoustic_envelop.pickle"
        if not target.exists():
            missing.append(target)
    if missing:
        missing_str = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Required predictor files not found:\n{missing_str}")
    _log(f"Using {len(stimuli['stimulus'].cells)} per-stimulus acoustic_envelop predictors from {pred_dir}.")


# Path for BIDS (Brain Imaging Data Structure) data
# example:
# sub-R2349/ (subject)
#   ├── meg/ (meg raw data)
#   ├── anat/ (mri data)
# derivatives/
#   ├── predictors/ (predictor files)
#   ├── ica/ (noise removal results)
#   ├── freesurfer/ (map MEG signals from sensors → brain surfaces)
#   ├── trans/ (coordinate alignment)
#   ├── eelbrain/ (Store intermediate results used by the TRF pipeline)

DATA_ROOT = "/Users/yanyuwoo/Data/Appleseed_BIDS_20251216"


class AppleSeed(TRFExperiment):
    data_dir = "meg"
    subject_re = r"sub-[A-Za-z0-9]+"  # BIDS subjects, e.g. sub-01 or sub-R2677
    sessions = ["Appleseed"]  # must match BIDS task name in filenames

    raw = {"0.5-20": RawFilter("raw", 0.5, 20, cache=False)}

    # At least one epoch required for load_trf. Task name must match BIDS (e.g. task-Appleseed in filenames).
    epochs = {
        "Appleseed": PrimaryEpoch("Appleseed", "event == 'onset'", tmin=0, tmax="length", samplingrate=100),
        # Minimal covariance epoch for source-space/NCRF demos.
        "cov": PrimaryEpoch("Appleseed", None, tmin=-0.100, tmax=0.0, samplingrate=100),
    }
    defaults = {"epoch": "Appleseed"}

    # Predictor files must exist as derivatives/predictors/{stimulus}~acoustic_envelop.pickle
    predictors = {
        "acoustic_envelop": FilePredictor(),
    }

    models = {
        "acoustic_envelop": "acoustic_envelop",
    }

    # Eelbrain derives events from the raw stimulus channel here. The final
    # segment-wise stimulus labels are injected in label_events() so FilePredictor
    # loads one predictor file per segment instead of slicing a long concatenation.
    variables = {
        "event": LabelVar("trigger", {162: "onset", 167: "offset"}),
    }
    tests = {}

    # Estimators registry
    estimators = {
        # partitions required when n_cases not in 3..10 (e.g. 22 trials)
        "boosting": BoostingEstimator(basis=0.050, partitions=5),
        "boosting-l2": BoostingEstimator(error="l2", partitions=5),
        "ncrf": NCRFEstimator(mu=0.001, n_iter=100),
        "ncrf-fast": NCRFEstimator(mu=0.01, n_iter=10),
    }

    def fix_events(self, ds):
        if ds.info.get("subject") == "R2676":
            return combine([ds[:10], ds[11:]])
        return ds

    def label_events(self, ds):
        subject = ds.info.get("subject")
        stimuli = _stimuli_for_subject(subject)

        if stimuli.n_cases != ds.n_cases:
            raise RuntimeError(
                f"Expected {stimuli.n_cases} stimulus rows for {subject}, "
                f"got {ds.n_cases} events"
            )

        _log(f"label_events() called for subject={subject}")
        _log(f"Event count={ds.n_cases}")
        _log(f"Stimulus count={stimuli.n_cases}")
        _log(f"Assigned stimulus sequence={list(stimuli['stimulus'])}")

        ds.update(stimuli)

        _log("All event-stimulus mappings:")
        for i in range(ds.n_cases):
            stim = ds["stimulus"][i]
            length = ds["length"][i]
            _log(f"{i:02d}: stimulus={stim}, length={length}")

        return ds
    # Initialize the demo experiment
def _init_demo_experiment() -> AppleSeed:
    e = AppleSeed(DATA_ROOT)
    e._register_model(e._coerce_model("acoustic_envelop"))
    subjects = [s for s in (e._field_values.get("subject") or ()) if s != "emptyroom"]
    if not subjects:
        raise RuntimeError(f"No non-emptyroom subjects found under {DATA_ROOT}")
    subject = subjects[0]
    e.set(subject=subject)
    return e


def _demo_subjects(e: AppleSeed, n_subjects: int = 3) -> list[str]:
    subjects = [s for s in (e._field_values.get("subject") or ()) if s != "emptyroom"]
    if len(subjects) < n_subjects:
        raise RuntimeError(f"Need at least {n_subjects} non-emptyroom subjects under {DATA_ROOT}, found {len(subjects)}")
    return subjects[:n_subjects]


# Run the demo
def _run_demo(estimator: str, subject: str = None, **load_trf_kwargs):
    e = _init_demo_experiment()
    if subject is not None:
        e.set(subject=subject)
    _log(f"Initialized experiment with DATA_ROOT={DATA_ROOT}")
    _log(f"Subject={e.get('subject')}, Epoch={e.get('epoch')}, Estimator={estimator!r}, Options={load_trf_kwargs!r}")
    _ensure_demo_predictor(e)
    if load_trf_kwargs.get("data") == "meg":
        _log("Using sensor-space MEG data.")
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
    """Recommended sensor-space boosting demo using the estimator registry."""
    return _run_demo("boosting", data="meg")


def run_boosting_demo_multi():
    """Run the sensor-space boosting demo for three subjects to compare subject-specific behavior."""
    e = _init_demo_experiment()
    subjects = _demo_subjects(e, 3)
    _log(f"Running boosting demo for subjects: {', '.join(subjects)}")
    return [_run_demo("boosting", subject=subject, data="meg") for subject in subjects]


def run_ncrf_demo():
    """Recommended NCRF demo with estimator-managed sensor-space semantics."""
    return _run_demo("ncrf")


if __name__ == "__main__":
    run_boosting_demo_multi()
    # run_boosting_demo()
    # run_ncrf_demo()
