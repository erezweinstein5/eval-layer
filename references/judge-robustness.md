# Judge Robustness

Every judge outcome must be recorded. Validate complete responses before
aggregating, whether the backend is a prompted LLM or Jev.

## Import the canonical helpers

The executable implementation lives in
[scripts/judge_results.py](../scripts/judge_results.py). Copy that file to
`evals/judge_results.py` when generating a harness, then import it:

```python
from judge_results import (
    compute_scores,
    parse_judge_response,
    validate_judge,
    validate_rubric,
)
```

Inside this skill repository, use the `scripts.judge_results` module instead.
Do not maintain separate parser, validator, or aggregation implementations in
the harness, replay tool, or report generator.

| Helper | Contract |
|---|---|
| `validate_rubric(rubric)` | Returns `None` on success; raises `ValueError` for invalid configuration |
| `parse_judge_response(text)` | Returns a JSON object or a tagged `parse_failed` dict |
| `validate_judge(parsed, rubric)` | Returns a canonical judge or a tagged failure; invalid rubric raises `ValueError` |
| `compute_scores(results, rubric)` | Validates each judge and aggregates only complete successes; invalid rubric raises `ValueError` |

Validate the rubric before running any subject or judge. The shared checks
require a nonempty dimension list, unique nonblank names, integer scales of at
least 2, finite positive weights summing to 1 (absolute tolerance `1e-9`), and a
finite `pass_threshold` in `[0, 1]`. Booleans are not numeric values.
Level descriptors and Jev-specific level requirements are checked by the
backend adapter separately; they are not part of this shared validation.

## Parse, then validate

`parse_judge_response` tries direct JSON, a JSON code fence, then an object
surrounded by prose. Successfully decoded arrays, strings, numbers, booleans,
and null are failures. It does not search inside such values for a nested
judge. Invalid inputs return `{"error": "parse_failed", "raw": ...}` with an
excerpt of at most 500 characters.
Nonfinite constants such as `NaN` and `Infinity`, including overflowing numeric
literals, are rejected. Failure diagnostics remain serializable with
`json.dumps(..., allow_nan=False)`: unsafe raw objects become printable strings.

Parsing proves only that an object was extracted. `validate_judge` requires
`scores` and `details` to contain exactly the rubric dimension names. Missing
or extra dimensions fail validation. Every score must be a finite, nonboolean
number within `1..scale`; fractional scores are accepted for both backends.

An incomplete judge is a **failed evaluation**. Do not use its surviving
dimensions, turn absent values into zero, or renormalize the weights.

## Successful judge contract

These fields describe the judge, separately from the agent's metadata:

| Field | Meaning |
|---|---|
| `backend` | `"llm"` or `"jev"`; absent on legacy input means `"llm"` |
| `scores` | Exact dimension names mapped to valid numeric scores |
| `details` | Exact dimension names mapped to backend-specific detail objects |
| `overall_reasoning` | String or `None`; defaults to `None` when unavailable |
| `model_id` | Optional string naming the requested judge model |
| `resolved_model_id` | Optional string identifying the resolved judge model |
| `usage` | Dict or `None`; missing usage stays unknown |
| `latency_ms` | Finite nonnegative number or `None`; missing latency stays unknown |
| `raw_response` | Optional original backend response for inspection or replay |

Omit unknown optional model IDs. Do not invent a resolved ID or substitute
zero for missing usage. Validation preserves transport metadata and other
top-level fields; it does not mutate the input.
When `usage.input_tokens` or `usage.output_tokens` is present, it must be a
nonnegative integer, excluding booleans. Omit unknown counts or use `usage: None`.
The whole result, including provider-specific metadata, must serialize as
finite JSON. Unsafe values cause a tagged failure with a safe raw diagnostic.

For `llm`, each detail contains string `reasoning`, list-of-strings `evidence`,
string `suggestion`, and `confidence` equal to `high`, `medium`, or `low`.
The normal existing LLM format needs no new transport fields; validation adds
the backend and unknown metadata defaults.

For `jev`, the adapter must explicitly set `backend: "jev"`. Each detail has
`reasoning`, `evidence`, and `suggestion` set to `None`. Keep numeric confidence
in `[0, 1]` as reported. This contract does **not** interpret it as a calibrated
probability that the answer is correct or translate it into LLM confidence
labels. Confidence is required: missing, null, and malformed values fail
validation. Do not conceal a backend failure by substituting unknown confidence.

Jev scores are stored as floats without rounding to rubric integers.
Do not synthesize explanations for a backend that supplied none. Retain the
raw response when available to distinguish backend output from normalization.

## Retry once on LLM parse failure

The following wrapper accepts the caller's LLM function and rendered prompt.
It retries only parse failures, once. Schema failures are recorded directly.
Jev transport retries belong in its adapter.

```python
from judge_results import parse_judge_response, validate_judge, validate_rubric


def judge_with_retry(call_judge_llm, prompt: str, rubric: dict) -> dict:
    validate_rubric(rubric)
    current_prompt = prompt
    for attempt in range(2):
        try:
            parsed = parse_judge_response(call_judge_llm(current_prompt))
        except Exception as exc:
            return {
                "backend": "llm",
                "error": "judge_exception",
                "reason": f"{type(exc).__name__}: {exc}",
            }
        if parsed.get("error") != "parse_failed" or attempt == 1:
            return validate_judge(parsed, rubric)
        current_prompt = prompt + (
            "\n\nYour previous response was not a JSON object. "
            "Respond with exactly one JSON object, without surrounding prose."
        )
```

Invoke this wrapper only when the user has enabled judging. During offline
checks, mock the callable. `--no-judge` means no judge call or invented scores.

## Never drop a result

Every raw JSONL row has a `judge` field: a canonical response, a tagged failure,
or `None` for skipped judging. Keep the **7-field agent contract** unchanged
inside `agent_result`, alongside case identity and external gates.

```python
import json


def write_row(raw_file, case, agent_result, judge, gates, *, subject, trial,
              evidence=None):
    expected = {
        "recommendation", "latency_ms", "tool_calls",
        "input_tokens", "output_tokens", "model_id", "error",
    }
    if set(agent_result) != expected:
        raise ValueError("agent_result must retain the 7-field metadata contract")
    row = {
        "case_id": case["id"],
        "subject": subject,
        "trial": trial,
        "input": case["input"],
        "context": case.get("context"),
        "expected_output": case.get("expected_output"),
        "evidence": evidence,
        "agent_output": agent_result["recommendation"],
        "agent_result": agent_result,
        "gates": gates,
        "judge": judge,
    }
    raw_file.write(json.dumps(row, allow_nan=False) + "\n")
    return row
```

Catch provider exceptions in the adapter and return a tagged judge failure.
The agent's `error` remains independent of a judge-side `error`.
Persist every case/trial, including agent failures and skipped judges.
Save the original input and inline evidence alongside the output so
[rejudge.py](../scripts/rejudge.py) can evaluate it without rerunning the agent.
Copy reference scores and provenance metadata only when they belong to that
exact saved output and rubric; do not attach the current case sketch's grades
as if they were grades for a new candidate.

## Defensive aggregation and external gates

Import `compute_scores` in both live evaluation and replay/report code. It
retains the established report fields:

| Field | Meaning |
|---|---|
| `n_total` | All input rows, including skipped, invalid, and failed judges |
| `n_scored` | Complete judges that passed backend-aware validation |
| `per_dimension_avg` | Raw dimension means, rounded to 2 decimals; `None` if unscored |
| `weighted_overall` | Mean of `sum(weight * score / scale)`, rounded to 3 decimals |
| `error_counts` | Counts of `judge_missing`, `schema_failed`, and existing error tags |

No scored rows means `weighted_overall: None`. Missing and explicitly null
judges count as `judge_missing`; malformed rows or judges count as
`schema_failed`. Any tagged error excludes the entire judge, even if it also
contains plausible scores. Never drop those rows from `n_total`.

⚠️ These are quality statistics, not a pass rate. A valid judge can score an
output whose required external checks failed. Keep that quality score visible
alongside the failed gates; a high score cannot make that trial pass.
`compute_scores` does not set verdicts or modify rows. Reports must derive
trial pass/fail from external gates plus the **unrounded** per-trial score.

This example accepts a boolean already computed by the harness from all
applicable external gates:

```python
import math

from judge_results import validate_judge


def judged_trial_passes(row: dict, rubric: dict, external_gates_passed: bool) -> bool:
    judge = validate_judge(row.get("judge"), rubric)
    if external_gates_passed is not True or "error" in judge:
        return False
    weighted = math.fsum(
        dim["weight"] * (judge["scores"][dim["name"]] / dim["scale"])
        for dim in rubric["dimensions"]
    )
    return weighted >= rubric["pass_threshold"]
```

When judging is skipped, report deterministic acceptance independently;
do not use this optional rubric gate to replace the harness's external checks.

## Failure categories to report

| Outcome | `agent_result.error` | `agent_result.recommendation` | `judge` |
|---|---|---|---|
| Scored output | `None` | populated | Canonical validated judge |
| Agent failure | set | `None` | `None` |
| Agent output parse failure | `None` or adapter parse error | `None` | `None` |
| Judge failure | `None` | populated | `{"error": "..."}` |
| Judge skipped | `None` | populated | `None` |

Also show external gate failures separately for coding cases; see
[references/codex.md](codex.md). Display `n_scored / n_total` with quality
averages and an independently computed `n_passed / n_total` for acceptance.
Judge failure counts alone do not identify agent failures or failed gates.
