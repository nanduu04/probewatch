# Results log

Append a new dated section after every feature. Never delete old results; mark them superseded.

## 2026-11-22: MVP v2 (offline simulation)

The experiment used three seeds with 500 benign sessions and 260 attack sessions per seed.
It held out 30% of sessions and calibrated thresholds on the training split at a 2% target
benign false-positive rate. The table shows realized rates on held-out sessions.

| Attacker | Per-prompt baseline | Session v1 (weighted) | Session v2 (learned) | v2, unseen attacker type |
|---|---|---|---|---|
| Blunt adaptive (tricks + euphemisms) | **100%** | 85.5% | **100%** | 100% |
| Stealthy (mixes in normal questions) | **100%** | 93.0% | **100%** | 97.8% |
| Twenty questions (each prompt harmless) | **0%** | 47.8% | **75.5%** (alerts ~11 turns before leak) | 52.2% |
| Benign false positives | 0.0% | 1.8% | 1.5% | 1.5–2.4% |

Values show the share of leaking attack sessions that triggered an alert before the leak.

### Findings

1. Session v2 improved pre-leak recall on the simulated twenty-questions cohort. The feature-based
   per-prompt baseline already reached 100% on the other two simulated cohorts.
2. Refusal **density** in a window beats a consecutive refusal **streak** (a streak resets whenever the bot answers once).
3. Repetition alone misleads: removing `near_dup` from v1 *raised* twenty-questions detection from
   48% to 87% (rephrasers and shoppers repeat too). Repetition only matters combined with refusals,
   which is why v2's interaction terms help.
4. Generalization is partial (52% on an attacker type never seen in training vs 76% when seen).
5. Sweep: with realistic benign vocabulary ("promo code", "reset my password"), the per-prompt
   baseline caught 0% of twenty-questions attacks at every benign refusal rate from 0% to 20%.

### Caveats (read before trusting)

- All synthetic; one author wrote both attackers and benign users, so the detector may learn simulator quirks.
- Detector changes were made after seeing early results (exploratory, not confirmatory), though all
  numbers use held-out data.
- Velocity is partly an artifact (half the simulated attackers are fast bots).
- This results log contains no live or production evaluation.
- **Strawman baseline:** the per-prompt baseline used our own weak features. A trained classifier
  (Prompt Guard 2) or a task-aware LLM judge may catch twenty-questions prompts on their own. Untested
  in this experiment. The 0% result applies to the feature-based baseline only.
- **FPR is not precision:** 1.5–2% benign FPR means roughly 150–200 false alerts per true attack at a
  1-in-10,000 attack rate. Not yet measured: recall at 0.1% FPR, precision at realistic base rates.
- **Vocabulary overlap:** `ASSET_TEXT` shares words with the simulated attackers' euphemisms.
- **Session reset:** an attacker who starts a new session every few turns resets the window. Untested.

## Pending (not yet measured)

See the [roadmap](ROADMAP.md) for evaluation priorities and stronger baseline comparisons.

## 2026-11-22 — Offline demo rerun at `404cb7b`

The default `python run_mvp.py experiment` command completed with 760 sessions per seed across
three seeds. Thresholds used the default 2% training false-positive budget. Attack rows show
pre-leak recall among leaking sessions. The last row shows realized held-out benign false-positive
rates.

| Cohort | Feature per-prompt | Session v1 | Session v2 | Session v2 with family held out of training |
|---|---:|---:|---:|---:|
| Blunt adaptive | 100.0% | 80.0% | 100.0% | 100.0% |
| Stealthy | 100.0% | 86.2% | 100.0% | 97.8% |
| Twenty questions | 0.0% | 23.3% | 60.0% | 46.7% |
| Benign false positives | 0.0% | 1.6% | 1.8% | 1.3–1.8% |

Compared with the earlier MVP run above, session v2 pre-leak recall on twenty questions fell
from 75.5% to 60.0%. Session v1 fell from 47.8% to 23.3% on the same family. Use this rerun
when reproducing the README demo. The comparison covers the feature baseline only. Performance
at 1% and 0.1% FPR, precision at realistic attack prevalence, and comparisons with stronger
per-prompt scorers remain unverified by this run.
