from pathlib import Path

import eelbrain
from eelbrain import save


DATA_ROOT = Path("/Users/yanyuwoo/Data/Appleseed_BIDS_20251216")
WAV_PATH = Path("/Users/yanyuwoo/Data/1.wav")
PREDICTOR_DIR = DATA_ROOT / "derivatives" / "predictors"
OUTPUT_PATH = PREDICTOR_DIR / "Appleseed~acoustic_envelop.pickle"

SAMPLINGRATE = 100
LOW_FREQUENCY = 0.5
HIGH_FREQUENCY = 20


def main():
    if not WAV_PATH.exists():
        raise FileNotFoundError(f"Wav file not found: {WAV_PATH}")

    PREDICTOR_DIR.mkdir(parents=True, exist_ok=True)

    wav = eelbrain.load.wav(WAV_PATH)
    envelope = wav.envelope()
    envelope = eelbrain.filter_data(envelope, LOW_FREQUENCY, HIGH_FREQUENCY, pad="reflect")
    envelope = eelbrain.resample(envelope, SAMPLINGRATE)
    envelope.name = "acoustic_envelop"

    save.pickle(envelope, OUTPUT_PATH)
    print(f"Wrote predictor: {OUTPUT_PATH}")
    print(f"Predictor tstep: {envelope.time.tstep}")
    print(f"Predictor nsamples: {envelope.time.nsamples}")


if __name__ == "__main__":
    main()
