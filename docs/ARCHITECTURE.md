# ProbeWatch architecture

A **lab** generates labelled sessions; a **detector** scores sessions from logs alone. They share exactly
one contract: the turn log schema. The detector never imports lab code.

```
Lab (lab.py / live.py)                                   Detector side (logs only)
actor -> guardrail -> target bot  ==>  turns.jsonl  ==>  normalise -> signals -> score -> alert (+ ATLAS IDs)
  (attackers, benign users)           (schema v1.0)       detector.py            evaluate.py: metrics, charts
```

## Modules

| Module | Responsibility | May import |
|---|---|---|
| `probewatch/__init__.py` | `SCHEMA_VERSION`, `ATLAS_IDS` | nothing |
| `probewatch/lab.py` | `KeywordGuardrail`, `MockTarget`, attackers, benign personas, `simulate()` | stdlib |
| `probewatch/live.py` | LLM-played roles via LiteLLM, `LLM` wrapper with call cap, `simulate()` | lab (`_turn`, `_gap`), litellm (lazy) |
| `probewatch/detector.py` | `load_sessions`, `load_labels` (eval only), `raw_signals`, `Detector` | numpy, sklearn. **Never lab/live** |
| `probewatch/evaluate.py` | `split`, `evaluate_run`, `metrics`, `ablation`, plots | detector, numpy, matplotlib |
| `probewatch/baselines.py` | Turn scorer protocol and detector adapter | detector, numpy |
| `run_mvp.py` | CLI | everything |

## Log schema v1.0 (the contract)

One JSON object per line, one line per turn.

| Field | Type | Required by detector |
|---|---|---|
| `schema_version` | str | part of the export contract |
| `session_id` | str | yes |
| `turn_index` | int, from 0 | yes |
| `ts` | ISO-8601 string | yes |
| `prompt` | str | yes |
| `response` | str or null | part of the export contract |
| `blocked` | bool or null | optional (a true value always counts as a refusal) |
| `label.actor` | `benign` or `attacker_*` | **eval only, stripped on load** |
| `label.trick_type` | str or null | eval only |
| `label.leaked` | bool | eval only |
| `meta` | dict (seed, persona, models) | no |

## Detector today (v2)

- `raw_signals(turns, window=10, mode="session"|"prompt") -> ndarray (n_turns, 5)`
  - `near_dup`: mean max character-n-gram similarity of recent prompts to their 5 predecessors
  - `block_density`: blocked or refused turns in the window
  - `strategy_switch`: distinct trick-cue families in the window (regex)
  - `topic_fixation`: mean similarity of window prompts to `ASSET_TEXT`
  - `velocity`: turns per minute across the window
  - `mode="prompt"` = per-prompt baseline with memoryless equivalents
- Embeddings: sklearn `HashingVectorizer(char_wb, 3–5)`, L2-normalised. No downloads, deterministic.
- `Detector.fit(benign, max_fpr, attack_sessions=None)`: per-signal calibration to 0..1 using benign
  p50–p99. Then either fixed weights (v1) or logistic regression with interaction terms (v2,
  `learned=True`). Threshold = the upper training quantile of benign session peak scores, plus a small offset.
- `first_alert(turns) -> dict | None`: session_id, alert_turn, score, threshold, signal breakdown,
  evidence turns, ATLAS IDs, reason string (template, no LLM).

## Baseline interfaces

```python
class TurnScorer(Protocol):          # probewatch/baselines.py: any detector or baseline
    name: str
    def score_turns(self, turns: list[dict]) -> np.ndarray: ...   # [0,1] per turn

```

The detector adapter supports the feature-based per-prompt scorer and session scorers.
Prompt Guard 2 and the task-aware LLM judges are planned. The current CLI runs the feature
baseline and the session detectors.

## Evaluation

- `split()` by session, stratified by actor, 70/30, seeded.
- Fit on the train split (benign for calibration; attackers too for v2). Score held-out sessions.
- Metrics per attacker type: guardrail breached %, caught-before-leak %, alerted %, median lead turns,
  benign false-positive %.
- The baseline metrics helper also computes pre-leak recall, any-time recall, Wilson intervals,
  and projected precision at specified attack base rates. The current CLI still writes the
  original MVP metrics.
- Always compare: per-prompt baseline vs session detectors; plus "never saw this attacker type in
  training" for v2.

## Design decisions

| Decision | Choice | Why |
|---|---|---|
| Where it runs | Offline on logs | Testable, works on any app's export; an inline proxy can wrap it later |
| Combiner | v2 logistic regression with interactions | MVP showed single signals mislead (rephrasers repeat, shoppers fixate); combinations carry the signal |
| Embeddings | Char n-gram hashing | Free, offline, deterministic. Weakness: not semantic, so swap in a local sentence model later |
| Storage | JSONL | Diffable, simple, readable by AI tools |
| LLM judge | None in v1 | Cheap, deterministic, immune to monitor-poisoning text; optional escalation-only judge later |
