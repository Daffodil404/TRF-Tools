import re
from pathlib import Path

import eelbrain
from eelbrain import save


DATA_ROOT = Path("/Users/yanyuwoo/Data/Appleseed_BIDS_20251216")
WAV_DIR = Path("/Users/yanyuwoo/Data/stimuli")
PREDICTOR_DIR = DATA_ROOT / "derivatives" / "predictors"

SAMPLINGRATE = 100
LOW_FREQUENCY = 0.5
HIGH_FREQUENCY = 20


def _wav_sort_key(path: Path) -> tuple[float, str]:
    match = re.search(r"(\d+)", path.stem)
    number = int(match.group(1)) if match else float("inf")
    return number, path.name


def _stimulus_id(path: Path) -> str:
    match = re.fullmatch(r"(\d+)([A-Za-z]*)", path.stem)
    if match is None:
        raise ValueError(f"Unexpected stimulus filename: {path.name}")
    return "".join(part for part in match.groups() if part)


def main():
    if not WAV_DIR.exists():
        raise FileNotFoundError(f"Stimulus directory not found: {WAV_DIR}")

    wav_paths = sorted((path for path in WAV_DIR.iterdir() if path.suffix.lower() == ".wav"), key=_wav_sort_key)
    if not wav_paths:
        raise FileNotFoundError(f"No wav files found in: {WAV_DIR}")

    PREDICTOR_DIR.mkdir(parents=True, exist_ok=True)

    wrote = []
    for path in wav_paths:
        stimulus = _stimulus_id(path)
        wav = eelbrain.load.wav(path)
        envelope = wav.envelope()
        envelope = eelbrain.filter_data(envelope, LOW_FREQUENCY, HIGH_FREQUENCY, pad="reflect")
        envelope = eelbrain.resample(envelope, SAMPLINGRATE)
        envelope.name = "acoustic_envelop"
        out_path = PREDICTOR_DIR / f"{stimulus}~acoustic_envelop.pickle"
        save.pickle(envelope, out_path)
        wrote.append((stimulus, out_path, envelope.time.nsamples))

    print(f"Wrote {len(wrote)} predictor files to: {PREDICTOR_DIR}")
    for stimulus, out_path, nsamples in wrote:
        print(f"  {stimulus}: {out_path.name} ({nsamples} samples)")


if __name__ == "__main__":
    main()
