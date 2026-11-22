# ProbeWatch

ProbeWatch scores chatbot sessions from exported logs and flags patterns associated with adaptive prompt attacks. Use it to review conversations after they happen, alongside your existing per-prompt guardrail.

The detector runs offline. It scores signals such as repeated prompts, refusals, changes in strategy, and repeated references to protected information. Alerts identify the first flagged turn and include a signal breakdown for investigation.

## Run the offline demo

Use Python 3.10 or newer. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_mvp.py experiment
python run_mvp.py score examples/sample_app_logs.jsonl
```

The experiment creates synthetic logs, writes a fitted detector to `runs/detector.json`, and saves metrics and charts under `runs/results/`. The scoring command loads that detector and prints one JSON object per flagged session, followed by a summary line. The demo needs no API key or model downloads after you install the dependencies.

## Evidence and limitations

The [MVP results](docs/RESULTS.md) cover a synthetic experiment against a feature-based per-prompt baseline. They do not establish performance on production logs, against stronger baselines, or at production false-positive rates. Treat the demo detector and its threshold as starting points for evaluation. ProbeWatch reviews individual sessions offline and cannot connect activity across sessions or accounts.

## Documentation

- [Score your own logs](docs/SCORING.md) — export format, detector selection, and alert fields
- [Architecture](docs/ARCHITECTURE.md) — modules, log schema, and detector internals
- [Evaluation results](docs/RESULTS.md) — experiment setup, findings, and limitations
- [Research](docs/RESEARCH.md) — threat model, coverage, and prior work
- [Roadmap](docs/ROADMAP.md) — planned experiments and development priorities
- [Contributing](docs/CONTRIBUTING.md) — tests and development conventions

## License

ProbeWatch is released under the [MIT License](LICENSE).
