# AI Task Governance Layer

A reference implementation of a pattern I've worked on in production enterprise AI
systems: **classifying incoming work by task type, tracking its true token/compute
cost, routing it to the right execution mode, and publishing the result as a stable
contract** — instead of letting AI usage sprawl across an organization ungoverned.

[![tests](https://github.com/sush0509sunday/Ai_governance_layer/actions/workflows/tests.yml/badge.svg)](https://github.com/sush0509sunday/Ai_governance_layer/actions/workflows/tests.yml)

> **Note on scope:** This is an original, from-scratch implementation built to
> illustrate the architecture pattern for a public portfolio. It does not contain,
> reproduce, or reference any proprietary code, data, or business logic from any
> employer. All sample data, pricing figures, and cost-avoidance estimates are
> synthetic and clearly labeled as such.

## Why this exists

As enterprises scale generative AI, a predictable failure mode shows up: teams adopt
AI quickly, usage grows, and nobody can answer basic questions like *"what is this
actually costing us, broken down by the kind of work it's doing?"* or *"which of this
usage is essential versus wasteful?"* Two well-documented symptoms:

- **Invisible ROI** — AI-assisted output gets absorbed into existing productivity
  metrics and credited to human output, so real gains aren't formally tracked.
- **Token exhaustion** — ungoverned usage (redundant context on every call, unmanaged
  agent loops, no caching or model-tier routing) causes costs to escalate faster than
  per-token prices fall.

This project models a governance layer that addresses both: **classify → route →
track → govern → export**.

### The routing pattern is only half the problem

There is a lot of good writing on *routing* — sending each request to the right model
or agent. Routing decides what a request costs. It does not tell you what you spent,
who spent it, whether it was justified, or when the system's judgment is degrading.

This project implements both halves, and treats the second as the harder one.

## Architecture

```
raw request
     │
     ▼
┌──────────────────┐   rule-based · embedding · LLM · ensemble
│  TaskClassifier  │──▶ task_type + confidence + score breakdown
└──────────────────┘
     │
     ▼
┌──────────────────┐   confidence ≥ escalation threshold → baseline tier
│ AgentOrchestrator│──▶ confidence ≥ manual floor        → escalate one tier
│ (routing policy) │    confidence <  manual floor        → human-in-the-loop
└──────────────────┘    near-tied top-two categories      → flag multi-intent
     │
     ▼
┌──────────────────┐
│   TokenTracker   │──▶ cost attributed by task type AND by tag
└──────────────────┘
     │
     ▼
┌──────────────────┐
│   CostGovernor   │──▶ budget alerts (per task type, per tag) + avoided-cost model
└──────────────────┘
     │
     ▼
┌──────────────────┐
│ DecisionLogger   │──▶ one audit row per decision + routing-health rates
└──────────────────┘
     │
     ▼
┌──────────────────┐
│  ExportBuilder   │──▶ governance_export.json  (versioned contract)
└──────────────────┘
     │
     ▼
 any consumer — dashboard, FinOps tool, chargeback system
```

Each module is deliberately small and swappable:

| Module | Responsibility | What you'd swap in production |
|---|---|---|
| `task_classifier.py` | Classify raw text into a task type | A real embedding API or hosted LLM behind the same `classify()` contract |
| `agent_orchestrator.py` | Execution mode, model tier, escalation, overlap detection | Real model API calls and task-runner infrastructure |
| `token_tracker.py` | Attribute usage and cost to task types and tags | Real usage data from provider billing/usage APIs |
| `cost_governor.py` | Budgets, overage alerts, avoided-cost estimates | Budgets sourced from finance; measured before/after in place of the estimate |
| `observability.py` | Audit trail + routing-health metrics | Ship records to your log pipeline instead of JSONL |
| `governance_export.py` | The versioned contract consumers read | Serve over HTTP instead of writing a file |

## Quickstart

```bash
git clone https://github.com/sush0509sunday/Ai_governance_layer.git
cd Ai_governance_layer
pip install -r requirements.txt

# End-to-end demo: classify, route, track, govern, export
python -m examples.demo

# Enterprise scenario: two initiatives, one governance layer
python -m examples.enterprise_demo

# Test suite
pytest tests/ -v

# Dashboard (after running the demo, which regenerates dashboard/data.json)
cd dashboard && python -m http.server 8000   # then open http://localhost:8000
```

The core library has **zero runtime dependencies**. `pytest` is needed only for tests.

## Four classifier backends, one interface

Every backend satisfies `classify(raw_text) -> ClassificationResult`, so nothing
downstream knows or cares which is running.

| Backend | Method | Strength | Cost |
|---|---|---|---|
| `RuleBasedClassifier` | keyword / regex | deterministic, free, instant | brittle to novel phrasing |
| `EmbeddingClassifier` | cosine similarity | catches paraphrases | needs embedding infra in production |
| `LLMClassifier` | model call | handles genuinely novel input | latency + per-call cost |
| `EnsembleClassifier` | majority vote | disagreement lowers confidence, which drives escalation | runs several backends |

```python
from src import EnsembleClassifier, RuleBasedClassifier, EmbeddingClassifier, AgentOrchestrator

classifier = EnsembleClassifier([RuleBasedClassifier(), EmbeddingClassifier()])
orchestrator = AgentOrchestrator(classifier=classifier)
task = orchestrator.process("T1", "review this PR", tags={"initiative": "agentic_commerce"})
```

`LLMClassifier` takes any callable — no provider is hardcoded — and **rejects labels
outside the `TaskType` enum** rather than routing on a hallucinated category.

## Failure modes this handles

Routing systems break in known ways. These are implemented and tested, not just
described:

- **Router hallucination** — an LLM router returning a category that doesn't exist.
  `LLMClassifier` falls back to `UNCLASSIFIED` instead of trusting it.
- **Overlapping categories** — "review this diff and help me with the error" is two
  intents. When the top two candidates score within a margin, the task is flagged
  `multi_intent` with both candidates recorded, rather than silently resolved.
- **Low-confidence routing** — instead of one hard trust/don't-trust cutoff, there are
  two: below the floor a human takes it; between floor and escalation threshold the
  category is trusted but the request is bumped one model tier.
- **Escalation that costs more than it saves** — see below.

## Calibrating escalation (the knob that decides whether this saves money)

Every escalated request runs on a pricier tier. Set the threshold by intuition and a
governance layer can escalate most of its traffic and cost *more* than no governance
at all.

Classifier confidence clusters rather than spreading evenly, so a threshold landing
inside a cluster makes spend hypersensitive to a tiny change. `escalation_sensitivity()`
makes the cliffs visible — from the bundled demo:

```
threshold   manual    escalated   baseline
------------------------------------------
0.65        10%       0%          90%
0.70        10%       0%          90%
0.75        10%       0%          90%      <- current
0.80        10%       45%         45%
0.85        10%       45%         45%
0.90        10%       50%         40%
```

Escalation jumps from 0% to 45% between 0.75 and 0.80. Pick a threshold sitting in a
*gap* between clusters, calibrated on real traffic — not a round number.

## Tags: cost attribution without an opinion

`GovernedTask.tags` is an open `str -> str` map the engine never interprets. Callers
attach whatever dimensions their organization uses — team, initiative, environment,
cost center — and the layer attributes cost against them.

This is what makes the engine reusable across organizations with different structures,
and it's what lets a downstream consumer do the strategic interpretation without the
engine needing to know what "agentic_commerce" means.

## The enterprise scenario

`examples/enterprise_demo.py` runs the problem an executive actually has: *fund the
strategic initiative heavily, keep routine work cheap, and don't personally review what
thousands of engineers are doing.*

```
=== Portfolio rollup (the exec view) ===

initiative           tasks   spend($)    esc rate   model mix
agentic_commerce     13      0.0519      0%         frontier=4, mid=6, small=3
internal_tools       14      0.0153      0%         mid=4, small=10

=== Budget exceptions (what actually reaches the exec) ===

initiative:internal_tools     $0.0153 / $0.008 cap  (191%)  [OVER CAP]
initiative:agentic_commerce   $0.0519 / $0.08  cap  ( 65%)  [ok]

1 of 2 initiatives breached cap — only these would page a human.
```

Policy is set per initiative and enforced on every request. Only exceptions surface.
Budget caps here are scaled to the demo's synthetic volume so the mechanism is visible
in one run.

## The export contract

The engine is not the end product. Consumers read one versioned document rather than
reaching into internals — see [`docs/EXPORT_SCHEMA.md`](docs/EXPORT_SCHEMA.md).

```python
from src import GovernanceFeed

feed = GovernanceFeed.from_file("dashboard/enterprise_export.json")
feed.alerts(over_cap_only=True)                 # exec view
feed.by_tag("initiative")                       # portfolio rollup
feed.decisions(initiative="agentic_commerce")   # drill-down
```

Three design rules the contract enforces:

1. **Aggregations are computed engine-side, never by the consumer.** If two systems can
   independently compute the same total, they will eventually disagree and neither will
   be trusted.
2. **The payload is versioned.** `GovernanceFeed` rejects an incompatible major version
   instead of silently mis-parsing.
3. **Projections are labeled in the data**, not only the docs — `is_projection: true`
   travels with `projection_methodology` so the caveat can't get separated from the number.

## A note on the "avoided cost" numbers

`CostGovernor.estimate_avoided_cost()` uses an illustrative multiplier to model what
ungoverned usage would have cost. This is a **placeholder assumption, not a measured
result** — the docstring, the exported JSON, and the demo output all say so.

In any real business case, a number like this should be labeled **projected/estimated**
unless it's backed by an actual before/after measurement against historical spend.
Conflating a projection with a realized result is a credibility risk, not just a
modeling nitpick — this project is written to model that discipline, not just the
cost-tracking logic.

## Project structure

```
Ai_governance_layer/
├── src/
│   ├── models.py               # TaskType, GovernedTask, TokenUsage, PriceTable
│   ├── task_classifier.py      # 4 interchangeable classifier backends
│   ├── agent_orchestrator.py   # routing policy, escalation, overlap detection
│   ├── token_tracker.py        # cost attribution by task type and tag
│   ├── cost_governor.py        # budgets, alerts, avoided-cost modeling
│   ├── observability.py        # decision log + routing-health metrics
│   └── governance_export.py    # versioned contract (ExportBuilder / GovernanceFeed)
├── examples/
│   ├── demo.py                 # end-to-end single-tenant run
│   └── enterprise_demo.py      # two initiatives, portfolio rollup, exceptions
├── dashboard/
│   └── index.html              # Chart.js visualization of governance data
├── docs/
│   └── EXPORT_SCHEMA.md        # the integration contract
├── tests/                      # 69 tests across all modules
├── data/                       # synthetic sample requests
└── .github/workflows/tests.yml # CI on Python 3.10 / 3.11 / 3.12
```

## License

MIT — see [LICENSE](LICENSE).
