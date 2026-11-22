# Contribute to ProbeWatch

## Run the tests

Set up Python and install the dependencies using the [README](../README.md#run-the-offline-demo). Run the test suite from the repository root before submitting changes.

```bash
python -m pytest -q
```

Tests run offline and use deterministic inputs. Keep provider calls and model downloads out of tests.

## Change scoring or evaluation

- Keep the detector independent of simulator modules. Scoring consumes the [turn log schema](ARCHITECTURE.md#log-schema-v10-the-contract) and excludes ground-truth labels.
- Keep existing signal column order. Append new signals, test them on small hand-built sessions, and add a memoryless equivalent to the feature baseline.
- Split by session ID and calibrate on training data only. Compare scorers at the same target false-positive budget and report each scorer's realized held-out false-positive rate.
- Report recall at 1% and 0.1% false-positive budgets when the experiment supports them, along with projected precision at attack prevalence of 1%, 0.1%, and 0.01%. Identify unavailable baselines and inconclusive operating points.
- Run the offline experiment after a feature change and record the setup, results, and regressions in [RESULTS.md](RESULTS.md).

Use the [roadmap](ROADMAP.md) to find planned work. Include a synthetic or redacted example with bug reports. Keep credentials, real transcripts, and generated run artifacts out of contributions.
