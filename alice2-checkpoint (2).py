from eelbrain.pipeline import *
from trftools.pipeline import *

DATA_ROOT = "~/Data/Alice"

SEGMENT_DURATION = {
    '1': 57.541,
    '2': 60.845,
    '3': 63.259,
    '4': 69.989,
    '5': 66.273,
    '6': 63.778,
    '7': 62.897,
    '8': 57.311,
    '9': 57.226,
    '10': 61.27,
    '11': 56.17,
    '12': 46.983,
}


class Alice(TRFExperiment):
    # Pre-BIDS attributes
    data_dir = 'eeg'
    subject_re = r'S\d\d'
    sessions = ['alice']

    raw = {
        '0.5-20': RawFilter('raw', 0.5, 20, cache=False),
    }

    variables = {
        'duration': LabelVar('event', {k: v + 1 for k, v in SEGMENT_DURATION.items()}),
    }

    tests = {
        'mytest': TTestRelated('dsa', 'ds'),
    }

    estimators = {
        'ncrf': NCRFEstimator(mu=0.1, n_iter=100),
        'boosting': BoostingEstimator(basis=0.050),
        'ncrf-fast': NCRFEstimator(mu=01, n_iter=10),
    }



e = Alice(DATA_ROOT)

e.load_test(..., test='mytest')

ncrf = e.load_trf('acoustic_envelop', 0, 0.500, estimator='ncrf')
ncrf_2 = e.load_trf('acoustic_envelop', 0, 0.500, estimator='ncrf-fast')

boosting_trf = e.load_trf('acoustic_envelop', 0, 0.500, estimator='boosting')
