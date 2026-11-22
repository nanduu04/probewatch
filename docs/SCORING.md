# Score chat logs offline

## Export turns as JSON Lines

Write one JSON object per turn. Use a unique session ID for each conversation, a turn index starting at zero, and an ISO 8601 timestamp with a timezone. ProbeWatch groups records by session ID and sorts each session by turn index.

```json
{"schema_version":"1.0","session_id":"chat-001","turn_index":0,"ts":"2026-01-01T12:00:00Z","prompt":"How do I reset my password?","response":"Open account settings and select Reset password.","blocked":false}
```

Use `null` for the response when the guardrail blocks a turn before the app generates a reply. Set `blocked` to `true` when your app records a block. The detector also treats a null response or a response matching its refusal heuristic as a refusal, even when `blocked` is `false`. Keep original turn boundaries and order when exporting. The detector discards evaluation labels if present. See the [log schema](ARCHITECTURE.md#log-schema-v10-the-contract) and [sample export](../examples/sample_app_logs.jsonl).

## Select a fitted detector

Follow the [offline demo](../README.md#run-the-offline-demo) to create `runs/detector.json`. From the repository root, run:

```bash
python run_mvp.py score path/to/chat-logs.jsonl --detector runs/detector.json
```

You can omit `--detector` to use that same path. The demo detector learns from simulated traffic. Before using scores to prioritize reviews, calibrate on representative benign sessions and evaluate separately on held-out sessions from your application.

## Read alert output

The command emits the first threshold crossing per flagged session. Each JSON object includes the following fields.

| Field | Meaning |
|---|---|
| `session_id` and `alert_turn` | Identify the conversation and first flagged turn. |
| `score` and `threshold` | Show the detector score and the threshold used for that run. |
| `signals` and `reason` | Show the normalized signals and summarize the highest values. |
| `evidence_turns` | Lists up to five blocked or refused turns in the alert window. This list can be empty. |
| `atlas` | Lists the MITRE ATLAS technique IDs associated with the detector. |

The final line starts with `#` and reports the number of flagged sessions. Exclude it when parsing alerts as JSON Lines. Keep real transcripts and alert exports local and use synthetic examples in public reports.
