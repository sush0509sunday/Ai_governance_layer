# Export contract — `governance_export.json`

**Current version: `1.0`**

This document is the integration boundary between the governance engine
and anything that consumes it. A consumer should depend on this schema
and on `GovernanceFeed` — never on `TokenTracker`, `DecisionLogger`, or
any other internal class.

## Versioning policy

- **Minor bump** (`1.0` → `1.1`) — additive only. New optional fields.
  Existing consumers keep working untouched.
- **Major bump** (`1.x` → `2.0`) — a field was removed, renamed, or
  changed meaning. Consumers must be updated.

`GovernanceFeed` validates the major version on load and raises rather
than silently mis-parsing a document it doesn't understand.

## Document shape

```jsonc
{
  "schema_version": "1.0",
  "generated_at": "2026-07-27T12:00:00+00:00",

  "totals": {
    "total_governed_cost_usd": 0.0912,
    "estimated_avoided_cost_usd": 0.1300,
    "is_projection": true,              // ALWAYS true for avoided cost
    "projection_methodology": "...",    // how the number was derived
    "ungoverned_multiplier": 2.4
  },

  "summaries": {
    "by_task_type": [
      {
        "task_type": "code_review",
        "task_count": 3,
        "total_input_tokens": 620,
        "total_output_tokens": 840,
        "total_cost_usd": 0.0345,
        "avg_cost_per_task_usd": 0.0115,
        "estimated_avoided_cost_usd": 0.05,
        "is_projection": true
      }
    ],
    "by_tag": [
      {
        "tag_key": "initiative",
        "tag_value": "agentic_commerce",
        "task_count": 13,
        "total_cost_usd": 0.0519,
        "avg_cost_per_task_usd": 0.0039,
        "escalated_count": 0,
        "escalation_rate": 0.0,
        "models_used": { "frontier-model": 4, "mid-tier-model": 6 }
      }
    ]
  },

  "alerts": [
    {
      "scope": "tag",                   // "tag" | "task_type"
      "subject": "initiative:internal_tools",
      "tag_key": "initiative",
      "tag_value": "internal_tools",
      "spend_usd": 0.0153,
      "cap_usd": 0.008,
      "pct_of_cap": 1.91,
      "status": "over_cap"              // "over_cap" | "ok"
    }
  ],

  "routing_health": {
    "decisions": 31,
    "escalation_rate": 0.0,
    "manual_fallback_rate": 0.12,
    "multi_intent_rate": 0.03,
    "model_mix": { "small-fast-model": 13, "mid-tier-model": 10 }
  },

  "decisions": [
    {
      "task_id": "E000",
      "timestamp": "2026-07-27T12:00:00+00:00",
      "task_type": "code_review",
      "classification_confidence": 0.9,
      "classification_method": "ensemble",
      "execution_mode": "governed_agent",
      "routed_model": "frontier-model",
      "baseline_model": null,           // set only when escalated
      "escalated": false,
      "multi_intent": false,
      "reason": null,                   // set on manual fallback
      "input_tokens": 96,
      "output_tokens": 302,
      "cost_usd": 0.012848,
      "tags": { "initiative": "agentic_commerce" }
    }
  ]
}
```

`routing_health` and `decisions` are present only when the engine was
run with a `DecisionLogger`. `decisions` can be omitted for large
exports via `ExportBuilder.build(include_decisions=False)`.

## Three grains, three audiences

| Key | Grain | Answers |
|---|---|---|
| `alerts` | exceptions only | "What needs my attention?" |
| `summaries` | rolled up | "Where is the money going?" |
| `decisions` | one row per request | "Why did *this* cost that?" |

An exec-facing view should read `alerts` first and stop there. A
drill-down view reads `decisions`. Neither should recompute anything
found in `summaries`.

## Rules for consumers

1. **Never recompute an aggregate the engine already published.** If a
   consumer derives its own total from `decisions[]`, the two numbers
   will eventually diverge and neither will be trusted. The engine is
   the single source of truth.
2. **Never present a projected figure as measured.** Any field carrying
   `is_projection: true` must be labeled as an estimate wherever it is
   displayed. `projection_methodology` travels with the data precisely
   so the caveat cannot get separated from the number.
3. **Check `schema_version` before parsing.** `GovernanceFeed` does this
   for you.
4. **Treat `tags` as opaque.** The engine attributes cost against tags
   without interpreting them. Grouping, naming and strategic meaning are
   the consumer's job.

## Reading the contract

```python
from src import GovernanceFeed

feed = GovernanceFeed.from_file("dashboard/enterprise_export.json")

feed.alerts(over_cap_only=True)                # exec view
feed.by_tag("initiative")                      # portfolio rollup
feed.decisions(initiative="agentic_commerce")  # drill-down
feed.routing_health()                          # classifier health
```

## Transport

The same document is valid over any of these; consumers parse one schema
regardless of delivery:

- **File** — `ExportBuilder.write(path)`, read with `GovernanceFeed.from_file`.
- **In-process** — `ExportBuilder.build()` returns the dict directly.
- **HTTP** — serve `build()` as a JSON response body; the consumer wraps
  it with `GovernanceFeed(response.json())`.
