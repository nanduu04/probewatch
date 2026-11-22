# Research: attacks ProbeWatch must handle

This page describes the research behind session-level detection. See the [evaluation log](RESULTS.md) for measured ProbeWatch results.

## Taxonomy (and who owns each family)

| Family | Examples | Main defense | ProbeWatch role |
|---|---|---|---|
| **Multi-turn adaptive** | Crescendo, ActorAttack, PAIR/TAP, decomposition, Best-of-N | **Session monitoring** | **Core lane** |
| Disguised single prompts | GCG suffixes, ciphers, low-resource languages, ASCII art, many-shot | Per-prompt guardrail | Count disguise *switching* across a session |
| Indirect injection | Poisoned docs/web, tool outputs, invisible Unicode | Input sanitising, agent design | v3+: log tool/retrieval turns |
| Evading the detector | Many accounts, feedback probing, low-and-slow, monitor poisoning | Identity, cost, rate limits | Partial; can't be fully solved by detection |

## Attack reference

| Attack | Log signature | Key number | Paper |
|---|---|---|---|
| Crescendo | Gradual drift toward target, few refusals, builds on model's replies | +29–61% over prior jailbreaks on GPT-4 (USENIX Sec 2025) | arXiv 2404.01833 |
| ActorAttack | Starts on a related innocent topic, converges | Beat single/multi-turn attacks incl. o1 (ACL 2025) | arXiv 2410.10700 |
| PAIR | Refusal → rephrase loops | Often <20 queries | arXiv 2310.08419 |
| TAP | PAIR + branching | 84% on GPT-4-Turbo, ~22 queries | arXiv 2312.02119 |
| Best-of-N | Many near-duplicates with random caps/char noise | 89% GPT-4o, 78% Claude 3.5 Sonnet at N=10k; power law | arXiv 2412.03556 |
| Decomposition | Benign subtasks that combine into the goal | Sequential monitor stopped 93% (ICLR 2026) | arXiv 2506.10949 |
| Many-shot | One huge prompt with fake dialogue | Power law in number of shots | Anthropic, NeurIPS 2024 |
| GCG suffix | Gibberish suffix; high perplexity | Transfers to black-box models; fluent variants evade perplexity 80% | arXiv 2307.15043; 2404.07921 |
| Ciphers | Non-natural-language text | Some ciphers bypass GPT-4 ~100% in several domains | arXiv 2308.06463 |
| Low-resource languages | Sudden language switch | <1% → 79% on GPT-4 (single-turn, 2023); multi-turn 2026: 60–78% Afrikaans, 42–71% Kiswahili | arXiv 2310.02446; 2605.18239 |
| ASCII art | Block-character layout | Worked on 5 major models | arXiv 2402.11753 |
| Unicode tag smuggling | Invisible code points U+E0000–U+E007F | Trivial to scan for | AWS, Cisco, Trend Micro blogs |
| Indirect injection | Instructions inside retrieved data | Bing Chat, code completion demos | arXiv 2302.12173 |
| Expert humans, multi-turn | Varied, slow, creative | >70% vs single-digit-ASR defenses | arXiv 2408.15221 (MHJ) |

## Limits of the current detector

- A session boundary ends the detector's view of the conversation.
- Refusal patterns and protected-asset vocabulary vary across applications.
- A synthetic benchmark measures performance on its own traffic mix. Evaluate on representative data before using alerts to prioritize reviews.
- Low attack prevalence can yield many false alerts even at low false-positive rates. For the relationship between prevalence and alert precision, see Axelsson's *The Base-Rate Fallacy and the Difficulty of Intrusion Detection*.
- ProbeWatch analyzes existing logs. Enforce access controls in the application and keep credentials outside model context.

## Prior art (session-level defenses)

- Chen et al., lightweight sequential monitors (ICLR 2026): 93% on decomposition; beats Llama-Guard-4 and o3-mini at 90% lower cost.
- DeepContext (Feb 2026): stateful intent-drift model, F1 0.84 on multi-turn jailbreaks.
- Brown et al., Benchmarks for Stateful Defenses (2025–26); Stateful Online Monitoring (May 2026):
  cross-account clustering, 81–87% campaign recall at ~2.4–3.1% FPR with 10× benign traffic; "does
  not solve distributed misuse."
- Also: TurnGate, Paranoid Monitors (ICLR 2026), TwinGate, THRD.

## Test datasets (check licences before use)

MHJ (2,912 prompts / 537 multi-turn conversations), Tensor Trust (563k attacks on a secret password,
ICLR 2024), DecomposedHarm (4,634 tasks), SafeMTData (ActorAttack), BSD generator, AgentDojo (97 tasks,
629 security tests). Attack runners: PyRIT (PAIR, TAP, Crescendo), garak, promptfoo.
