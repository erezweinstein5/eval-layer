# Judge Robustness

LLM judges fail in predictable ways. Your harness must **never drop a
result** silently — tag every outcome so downstream aggregation can see it.

## Drop-in helper: `parse_judge_response`

Copy this into your harness:

```python
import json
import re
from typing import Any


def parse_judge_response(text: str) -> dict[str, Any]:
    """Extract the judge's JSON object. Tries progressively looser patterns.

    Returns `{"error": "parse_failed", ...}` rather than raising — the caller
    is responsible for recording the failure.
    """
    # 1. Direct JSON
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass

    # 2. Fenced ```json block
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        try:
            return json.loads(fence.group(1))
        except json.JSONDecodeError:
            pass

    # 3. Outermost {...}
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass

    return {"error": "parse_failed", "raw": text[:500]}
```

## Retry-once on parse failure

Don't retry silently on *every* call — that hides judge bugs. Retry
*once* with a clarifying reminder:

```python
def judge_with_retry(render_prompt, max_retries: int = 1) -> dict:
    prompt = render_prompt()
    for attempt in range(max_retries + 1):
        raw = call_judge_llm(prompt)
        parsed = parse_judge_response(raw)
        if "error" not in parsed:
            return parsed
        # Retry with clarifying reminder
        prompt = render_prompt() + (
            "\n\nYour previous response was not valid JSON. "
            "Respond with exactly one JSON object — no prose before or after."
        )
    return parsed  # final failure, tagged with {"error": "parse_failed"}
```

## Never drop a result

Every row in your raw JSONL must have a `judge` field, even when the judge
failed. Downstream code must handle `judge=None` and `judge={"error": ...}`
without crashing.

```python
# In your per-case loop:
try:
    judge = judge_with_retry(lambda: render_judge_prompt(case, agent_output))
except Exception as e:
    judge = {"error": f"judge_exception: {type(e).__name__}: {e}"}

row = {
    "case_id": case["id"],
    "agent_output": agent_output,
    "judge": judge,                 # <- always present, even on failure
    "error": agent_error,           # <- agent-side error, independent of judge
    ...
}
raw_file.write(json.dumps(row) + "\n")
```

## Defensive score aggregation

When summing scores, assume every field may be missing:

```python
def compute_scores(results: list[dict], rubric: dict) -> dict:
    dims = [d["name"] for d in rubric["dimensions"]]
    weights = {d["name"]: d["weight"] for d in rubric["dimensions"]}
    scales = {d["name"]: d["scale"] for d in rubric["dimensions"]}

    per_dim = {d: [] for d in dims}
    weighted_per_case: list[float] = []

    for r in results:
        # judge may be None, {"error": ...}, or {"scores": {...}}
        scores = (r.get("judge") or {}).get("scores") or {}
        weighted_sum = 0.0
        total_weight = 0.0
        for d in dims:
            v = scores.get(d)
            if v is None:
                continue                          # don't count missing
            per_dim[d].append(v)
            weighted_sum += (v / scales[d]) * weights[d]
            total_weight += weights[d]
        if total_weight > 0:                      # don't divide by zero
            weighted_per_case.append(weighted_sum / total_weight)

    return {
        "per_dimension_avg": {
            d: round(sum(vs) / len(vs), 2) if vs else None
            for d, vs in per_dim.items()
        },
        "weighted_overall": (
            round(sum(weighted_per_case) / len(weighted_per_case), 3)
            if weighted_per_case else None
        ),
        "n_scored": len(weighted_per_case),
        "n_total": len(results),
    }
```

**Rule of thumb**: every dict access inside score aggregation must use
`.get()` with a default. Use `None` to mean "missing" and check for it
explicitly — don't coerce to 0 (which would depress averages silently).

## Failure categories to report

Your final report should distinguish four outcomes:

| Outcome | `agent_error` | `recommendation` | `judge` |
|---|---|---|---|
| Success | `None` | populated | `{"scores": {...}}` |
| Agent failure | set | `None` | `None` |
| Parse failure | `None` | `None` | (not called) |
| Judge failure | `None` | populated | `{"error": "..."}` |

The leaderboard should show `n_successful / n_total`, not just an average
score. A framework that produces brilliant output 10% of the time and crashes
90% is **worse** than one that produces mediocre output 100% of the time —
hide that distinction and the report lies.
