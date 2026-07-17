# AI Task Governance Layer

A reference implementation of a pattern I've worked on in production enterprise AI systems: **classifying incoming work by task type, tracking its true token/compute cost, and routing it to the right execution mode** — instead of letting AI usage sprawl across an organization ungoverned.

> **Note on scope:** This is an original, from-scratch implementation built to illustrate the architecture pattern for a public portfolio. It does not contain, reproduce, or reference any proprietary code, data, or business logic from any employer. All sample data, pricing figures, and cost-avoidance estimates are synthetic and clearly labeled as such.

## Why this exists

As enterprises scale generative AI, a predictable failure mode shows up: teams adopt AI quickly, usage grows, and nobody can answer basic questions like *"what is this actually costing us, broken down by the kind of work it's doing?"* or *"which of this usage is essential versus wasteful?"* Two well-documented symptoms of this in 2026 industry reporting:

- **Invisible ROI** — AI-assisted output often gets absorbed into existing productivity metrics and credited to human output, so real gains aren't formally tracked as business outcomes.
- **Token exhaustion** — ungoverned usage (redundant context on every call, unmanaged agent loops, no caching or model-tier routing) causes costs to escalate faster than per-token prices fall.

This project models a governance layer that addresses both: **classify → track → govern → route**, so an organization can see where AI spend goes and make deliberate decisions about it, rather than discovering the bill after the fact.

## Architecture

```
                 ┌─────────────────┐
  raw request ──▶│  TaskClassifier  │──▶ task_type + confidence
                 └─────────────────┘
                          │
                          ▼
                 ┌─────────────────────┐
                 │  AgentOrchestrator   │──▶ routes to execution mode:
                 │  (routing policy)    │      - governed_agent (with model tier)
                 └─────────────────────┘      - ungoverned_manual (low-confidence fallback)
                          │
                          ▼
                 ┌─────────────────┐
                 │   TokenTracker   │──▶ aggregates cost by task type
                 └─────────────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │  CostGovernor    │──▶ budget alerts + avoided-cost estimate
                 └─────────────────┘
                          │
                          ▼
                 dashboard/index.html (visualization)
```

Each module is deliberately small and swappable:

| Module | Responsibility | What you'd swap in production |
|---|---|---|
| `task_classifier.py` | Classify raw text into a task type | A trained ML/LLM-based intent classifier — the interface (`classify() -> (task_type, confidence)`) stays identical |
| `agent_orchestrator.py` | Decide execution mode + model tier per task type | Real model API calls and task-runner infrastructure in place of the simulated usage |
| `token_tracker.py` | Attribute token usage and cost to task types | Real usage data from model provider billing/usage APIs |
| `cost_governor.py` | Apply budgets, flag overages, estimate avoided cost | Budgets sourced from finance systems; avoided-cost model replaced with a measured before/after comparison |

## Quickstart

```bash
git clone https://github.com/<your-username>/ai-governance-layer.git
cd ai-governance-layer
pip install -r requirements.txt

# Run the end-to-end demo (classifies sample requests, tracks cost, checks budgets)
python -m examples.demo

# Run the test suite
pytest tests/ -v

# View the dashboard (after running the demo, which regenerates dashboard/data.json)
cd dashboard && python -m http.server 8000
# then open http://localhost:8000
```

## Example output

```
task_id  task_type            conf.  mode               model            cost($)
----------------------------------------------------------------------------------
T000     support_query        0.8    governed_agent      small-fast-model 0.0002
T001     code_review          0.8    governed_agent      frontier-model   0.0092
...

=== Usage summary by task type ===
code_review          tasks=3    cost=$0.0238  est. avoided=$0.0300
workflow_automation  tasks=3    cost=$0.0160  est. avoided=$0.0200
...

Total governed cost: $0.0672
Total estimated avoided cost (illustrative, vs. ungoverned baseline): $0.0900
```

## A note on the "avoided cost" numbers

`CostGovernor.estimate_avoided_cost()` uses an illustrative multiplier to model what ungoverned usage would have cost. This is intentionally transparent about being a **placeholder assumption**, not a measured result — the docstring says so directly. In any real business case (including, if you're reading this because you're evaluating my work for something like an immigration petition or a hiring decision), a number like this should always be clearly labeled as **projected/estimated** unless it's backed by an actual before/after measurement against historical spend. Conflating a projection with a realized result is a credibility risk, not just a modeling nitpick — this project is written to model that discipline, not just the cost-tracking logic.

## Project structure

```
ai-governance-layer/
├── src/
│   ├── models.py              # core data types (TaskType, GovernedTask, TokenUsage, PriceTable)
│   ├── task_classifier.py     # classification interface + reference implementation
│   ├── agent_orchestrator.py  # routing policy + execution simulation
│   ├── token_tracker.py       # usage aggregation by task type
│   └── cost_governor.py       # budgets, alerts, avoided-cost modeling
├── examples/
│   └── demo.py                # end-to-end runnable example
├── dashboard/
│   └── index.html             # Chart.js visualization of governance data
├── tests/                     # pytest test suite (12 tests covering all modules)
├── data/
│   └── sample_requests.json   # synthetic sample data
└── .github/workflows/tests.yml
```

## License

MIT — see [LICENSE](LICENSE).
