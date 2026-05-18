from pathlib import Path

from eelbrain import combine, load
from eelbrain.pipeline import LabelVar, PrimaryEpoch, RawFilter

from trftools.pipeline import FilePredictor, TRFExperiment


def _log(msg: str):
    print(f"[direct-boosting-demo] {msg}")


DIR = Path(__file__).parent
STIMULI_LENGTHS = load.tsv(DIR / "appleseed_stimuli.txt", types="fv")
STIMULI_PILOTS = STIMULI_LENGTHS.sub("stimulus != '11'").repeat(2)
STIMULI_REAL = STIMULI_LENGTHS.sub("stimulus != '11b'").repeat(2)

DATA_ROOT = "/Users/yanyuwoo/Data/Appleseed_BIDS_20251216"


def _stimuli_for_subject(subject: str):
    return STIMULI_PILOTS if subject in ("R2650", "R2652") else STIMULI_REAL


class AppleSeedDirectBoosting(TRFExperiment):
    data_dir = "meg"
    subject_re = r"sub-[A-Za-z0-9]+"
    sessions = ["Appleseed"]

    raw = {"0.5-20": RawFilter("raw", 0.5, 20, cache=False)}
    defaults = {"epoch": "Appleseed"}

    epochs = {
        "Appleseed": PrimaryEpoch("Appleseed", "event == 'onset'", tmin=0, tmax="length", samplingrate=100),
    }

    predictors = {
        "acoustic_envelop": FilePredictor(),
    }

    models = {
        "acoustic_envelop": "acoustic_envelop",
    }

    variables = {
        "event": LabelVar("trigger", {162: "onset", 167: "offset"}),
    }
    tests = {}

    def fix_events(self, ds):
        if ds.info.get("subject") == "R2676":
            return combine([ds[:10], ds[11:]])
        return ds

    def label_events(self, ds):
        stimuli = _stimuli_for_subject(ds.info.get("subject"))
        if stimuli.n_cases != ds.n_cases:
            raise RuntimeError(
                f"Expected {stimuli.n_cases} stimulus rows for {ds.info.get('subject')}, got {ds.n_cases} events"
            )
        ds.update(stimuli)
        return ds


def _init_demo_experiment() -> AppleSeedDirectBoosting:
    e = AppleSeedDirectBoosting(DATA_ROOT)
    e._register_model(e._coerce_model("acoustic_envelop"))
    subjects = [s for s in (e._field_values.get("subject") or ()) if s != "emptyroom"]
    if not subjects:
        raise RuntimeError(f"No non-emptyroom subjects found under {DATA_ROOT}")
    e.set(subject=subjects[0])
    return e


def _demo_subjects(e: AppleSeedDirectBoosting, n_subjects: int = 3) -> list[str]:
    subjects = [s for s in (e._field_values.get("subject") or ()) if s != "emptyroom"]
    if len(subjects) < n_subjects:
        raise RuntimeError(f"Need at least {n_subjects} non-emptyroom subjects under {DATA_ROOT}, found {len(subjects)}")
    return subjects[:n_subjects]


def _ensure_demo_predictor(e):
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


def _run_demo(subject: str = None):
    e = _init_demo_experiment()
    if subject is not None:
        e.set(subject=subject)
    _log(f"Initialized experiment with DATA_ROOT={DATA_ROOT}")
    _log(f"Subject={e.get('subject')}, Epoch={e.get('epoch')}")
    _ensure_demo_predictor(e)
    _log("Running direct boosting TRF (no estimator registry, make=True)...")
    trf = e.load_trf(
        "acoustic_envelop",
        tstart=0,
        tstop=0.500,
        basis=0.050,
        error="l1",
        partitions=5,
        data="meg",
        make=True,
    )
    _log(f"Done: {trf}")
    return trf


def run_boosting_demo():
    return _run_demo()


def run_boosting_demo_multi():
    e = _init_demo_experiment()
    subjects = _demo_subjects(e, 3)
    _log(f"Running direct boosting demo for subjects: {', '.join(subjects)}")
    return [_run_demo(subject=subject) for subject in subjects]


if __name__ == "__main__":
    run_boosting_demo_multi()
