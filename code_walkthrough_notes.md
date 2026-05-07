# Code Walkthrough Notes

## 1. Opening

Today I will give a short walkthrough of the code changes for integrating NCRF into the TRF pipeline.

The main goal of this work is to let the pipeline support an estimator-based entry for NCRF, instead of relying only on the older calling style.

At a high level, the pipeline takes:

- neural data
- a stimulus-derived predictor
- an estimator choice

and then fits a temporal response model.

For this walkthrough, I will focus on:

1. where the execution starts
2. how the predictor is prepared
3. how the estimator is resolved
4. how the NCRF path is selected

## 2. Big Picture

The simplest way to understand the pipeline is:

`audio file -> predictor pickle -> pipeline -> estimator -> fitted result`

In my case:

- the audio input is `1.wav`
- the predictor is stored as `Appleseed~acoustic_envelop.pickle`
- the pipeline is implemented in `TRFExperiment`
- the estimator is `ncrf`

## 3. Start From The Demo

Open:

- `trftools/pipeline/estimator/demo.py`

What to say:

- This file is the entry point for the demo run.
- It defines a small experiment setup for the Appleseed dataset.
- It also shows how the new estimator interface is called.

Important places in this file:

- `AppleSeed`
- `_ensure_demo_predictor()`
- `run_ncrf_demo()`

Key message:

- The demo now uses a real predictor file, not a synthetic random time series.
- The recommended NCRF demo explicitly calls `load_trf(..., estimator="ncrf", data="meg")`.

## 4. Predictor Generation

Open:

- `trftools/pipeline/estimator/make_demo_predictor.py`

What to say:

- This script reads the real stimulus audio file `1.wav`.
- It computes an acoustic envelope.
- It filters and resamples the predictor to match the analysis rate.
- It saves the predictor as `Appleseed~acoustic_envelop.pickle`.

Why this matters:

- Previously, the demo used a synthetic predictor only for smoke testing.
- Now the predictor is generated from a real stimulus file.

## 5. Estimator Abstraction

Open:

- `trftools/pipeline/estimator/estimator.py`
- `trftools/pipeline/estimator/ncrf.py`

What to say:

- `Estimator` is the base abstraction for model fitting methods.
- `NCRFEstimator` stores NCRF-specific parameters such as `mu` and `n_iter`.
- `NCRFEstimator` also validates which arguments are allowed.

Important message:

- NCRF should only use sensor-space data in this interface.
- It should not accept source-style arguments such as `inv` or `mask`.

This is one of the main changes based on supervisor feedback.

## 6. Main Pipeline Entry

Open:

- `trftools/pipeline/_experiment.py`

Start with:

- `load_trf()`

What to say:

- `load_trf()` is the main orchestration function.
- It receives the model name, timing window, analysis options, and estimator.
- It resolves the estimator and applies estimator-specific settings before dispatching the actual fit job.

Important functions to point at:

- `_resolve_estimator()`
- `_apply_estimator_params()`
- `_trf_job()`

Key message:

- This is where the new estimator-based interface is integrated into the existing pipeline.

## 7. NCRF Execution Path

Still in:

- `trftools/pipeline/_experiment.py`

Point to:

- `_trf_job()`
- `_trf_job_ncrf_estimator()`

What to say:

- `_trf_job()` is responsible for choosing which fitting path to run.
- When the estimator is `ncrf`, the pipeline now branches into a dedicated NCRF-specific path.
- This is cleaner than relying only on the older logic.

Important clarification:

- The code still keeps the older calling style for compatibility.
- At the same time, it adds a new estimator-specific entrance for NCRF.

This means the pipeline supports:

- legacy calling patterns
- the new estimator-driven interface

## 8. Predictor Loading Support

Open:

- `trftools/pipeline/_predictor.py`

What to say:

- This file handles predictor loading and preprocessing.
- I also fixed a predictor resampling bug here during debugging.

Simple explanation:

- If the predictor sampling rate does not match the analysis rate, the pipeline needs to either resample it or raise a clear error.

## 9. Overall Changes

You can say:

1. I introduced an estimator abstraction and connected it to the pipeline entry in `load_trf()`.
2. I added an NCRF estimator and made its argument semantics stricter and clearer.
3. I added a separate estimator-specific entrance for the NCRF runtime path.
4. I updated the demo to use the latest Appleseed dataset setup.
5. I replaced the synthetic predictor path with a real predictor generated from the stimulus audio.
6. I also improved the surrounding integration logic, including predictor loading and dataset compatibility handling.

## 10. Design Pattern

If someone asks about the design idea, you can say:

- The estimator layer is close to a Strategy pattern.
- The pipeline keeps one common entry point, `load_trf()`.
- Different estimators provide different behavior through estimator objects.
- For example, `NCRFEstimator` supplies its own parameter validation and fitting configuration.

You do not need to overstate it. A simple way to explain it is:

- There is one common pipeline interface.
- The estimator object decides method-specific behavior.
- This makes it easier to extend the pipeline with additional fitting methods later.

## 11. Current Limitations

Be clear about what is not finished yet:

1. The NCRF runtime path is not fully decoupled from all legacy logic yet.
2. The cache naming still looks like the old boosting-style naming.
3. The current demo predictor is based on one available stimulus file, so the full stimulus-to-predictor mapping is still incomplete.
4. The dependency combination for `ncrf` and `eelbrain` still needs to be treated carefully.

## 12. Closing

Suggested closing:

This week, the main progress was moving NCRF toward an estimator-based integration, while also making the pipeline more consistent with the required parameter semantics. The remaining work is mostly about cleaning up the runtime path, cache naming, and scaling the predictor generation from a demo case to the full dataset.

## 13. Very Short Version

If time is short, you can say:

I added an estimator-based entry for NCRF, tightened the NCRF parameter rules, updated the demo to work with the latest dataset, and replaced the synthetic predictor with a real predictor generated from the stimulus audio. I also debugged several data and integration issues. The main remaining work is to further clean up the NCRF execution path and complete the full predictor setup.
