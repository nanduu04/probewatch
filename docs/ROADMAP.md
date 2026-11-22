# Roadmap

ProbeWatch focuses on offline review of adaptive probing within a chatbot session. It complements per-prompt guardrails and operates on one application's exported logs.

## Validate the current detector

- Compare the feature-based baseline and session detector with stronger per-prompt scorers on the same held-out sessions. The scorer adapter and baseline metrics helper are available. The CLI currently reports the original MVP metrics.
- Measure recall at 1% and 0.1% benign false-positive budgets. Report observed false-positive rates, sample sizes, uncertainty, and projected precision at realistic attack rates.
- Evaluate on appropriately licensed public datasets and representative benign traffic. Use synthetic or redacted examples in public results.
- Measure sensitivity to application-specific refusal wording and protected-asset vocabulary. Sessions are scored independently, so the current detector cannot connect activity across conversations.

## Improve triage where evaluation supports it

- Rank alerts for review and measure recall at fixed review-queue sizes.
- Let users describe the information their application protects.
- Evaluate additional signals against the strongest available baseline before adopting them.

See the [evaluation log](RESULTS.md) for measured results and [architecture](ARCHITECTURE.md) for current behavior.
