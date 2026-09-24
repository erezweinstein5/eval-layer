# Jev judge backend

Use this reference when the user selects Jev as the evaluator. Jev is a judge
backend, independent of the agent runtime: Codex, Claude Code, and every existing
framework adapter can produce the outputs it scores. The existing `llm` judge
path remains the default. No new SDK dependency is needed.

## API and configuration

The supplied [jev_judge.py](../scripts/jev_judge.py) calls the direct TypeSafe
endpoint `POST https://api.typesafe.ai/v1/systemone` using `TYPESAFE_API_KEY`.
Set that environment variable through your normal credential mechanism; never
put it into cases, rubrics, reports, or command-line arguments.

Select the model explicitly with `--judge-model`. A versioned model is preferable
for comparisons; aliases such as `jev-latest` can change. The adapter records
both the requested `model_id` and returned `resolved_model_id`. It never falls
back to another model or provider. Official model availability is in source [2].

Copy these modules beside the generated harness:

- [jev_judge.py](../scripts/jev_judge.py): request construction, HTTP, and normalization.
- [judge_results.py](../scripts/judge_results.py): shared parsing, validation, and aggregation.

The request contains `model`, `state`, and `questions`. Build one `score` question
per rubric dimension; include the dimension's meaning in `instructions`, because
question IDs are routing keys and are not sent to inference. Each question's
`criteria` is the dimension's ordered array of level descriptions [1].

`state` should include the original input, supplied context, agent output,
reference-answer sketch, and relevant evidence (diff and independent checks for
coding tasks). Do not include the candidate's existing judge scores, human
reference grades, or another subject's result. Keep rubric calibration examples
separate from held-out reference grades. Never silently truncate evidence to fit
an API limit; explicitly design a consistent evidence selection policy instead.

## Score mapping

A three-level rubric has keys `1`, `2`, `3`. The API receives the descriptions in
that order and returns levels `0`, `1`, `2`. Each answer includes its score,
legend, probability distribution, and numeric confidence [1, 3].

```text
Jev score = 1.75 on levels 0..2
Canonical score = 1.75 + 1 = 2.75 on rubric levels 1..3
Normalized contribution = weight × (2.75 / 3)
```

Do not round to integers or divide the raw Jev score by `scale - 1`; that would
change the existing harness normalization. Aggregation keeps `score / scale`
for both backends. Pass thresholds use the same normalized units (e.g. `0.7`).
A minimum score therefore contributes `1 / scale`, as it did before this adapter.

The adapter accepts complete textual descriptors for 2–10 levels, matching the
API limit [1]; this skill normally designs three- or five-level dimensions.
It validates all dimensions, finite values, bounds, legend correspondence,
probability keys and sums, and agreement between score and distribution. Any
invalid dimension fails the whole judge result; partial results are not averaged.
Live Jev 1.13 responses observed on 2026-09-22 round scores and probabilities to
two decimal places. Validation allows the corresponding bounded rounding error:
`0.005 * level_count` for probability mass and
`0.005 * (1 + sum(level_indices))` for the score. Original scores and distributions
remain unchanged; larger discrepancies still fail validation.

## Result contract

Keep the seven **agent** fields unchanged. Put judge information in `row["judge"]`:

```json
{
  "backend": "jev",
  "model_id": "jev-latest",
  "resolved_model_id": "jev-1.13.0",
  "scores": {"correctness": 2.75},
  "details": {
    "correctness": {
      "reasoning": null,
      "evidence": null,
      "suggestion": null,
      "confidence": 0.4,
      "raw_score": 1.75,
      "legend": {"0": "Wrong", "1": "Partly correct", "2": "Correct"},
      "probabilities": {"0": 0.0, "1": 0.25, "2": 0.75}
    }
  },
  "overall_reasoning": null,
  "usage": {"input_tokens": 80, "output_tokens": 12},
  "latency_ms": 123,
  "attempts": 1,
  "review_threshold": 0.5,
  "review_dimensions": ["correctness"]
}
```

Illustrative values only. The actual result also retains the full `raw_response`.
On failure it has `error` and `reason`, no usable scores, and available request
metadata/raw response. Reports must distinguish configuration, HTTP/transport,
and schema failures. An error cannot become a passing evaluation.

Jev does not generate the LLM judge's explanations. Render these as **unavailable
for this backend**; do not invent prose, mislabel a probability as evidence, or
call another model to fill the missing fields. The actual diff/check artifacts
remain evidence supplied to the evaluator.

Numeric confidence describes the returned distribution, not a probability that
the answer is correct [4]. The adapter defaults to flagging dimensions below
`review_threshold=0.5`; this is an adjustable starting point, not a calibrated
quality guarantee. Flags retain scores and route attention to review, not to an
automatic fallback. Keep LLM `high/medium/low` labels separate from Jev confidence.

## Harness routing

Generated harnesses expose `--judge-backend llm|jev` (default `llm`) and
`--judge-model`. Model selection belongs to the judge and must not overwrite the
agent's model. Keep `--no-judge` as the first branch so it needs no credentials:

```python
from jev_judge import evaluate
from judge_results import parse_judge_response, validate_judge


def score_case(case, agent_output, rubric, args, llm_call, evidence=None):
    # llm_call is the existing project judge: (state, rubric, model_id) -> text.
    # Invoke this only after the harness has recorded any agent failure.
    if args.no_judge:
        return None
    state = {
        "input": case["input"],
        "context": case.get("context"),
        "agent_output": agent_output,
        "expected_output": case.get("expected_output"),
        "evidence": evidence,
    }
    if args.judge_backend == "jev":
        return validate_judge(evaluate(state, rubric, model_id=args.judge_model), rubric)
    raw = llm_call(state, rubric, args.judge_model)
    return validate_judge(parse_judge_response(raw), rubric)
```

The surrounding harness catches LLM/provider errors, records `judge: null` and
`judge_status: skipped` when disabled, and always writes a row. Independent
acceptance checks still determine functional success. A high score or confidence
cannot override a failed required check. Compare judge backends on saved outputs;
do not accidentally vary agent behavior while evaluating judge differences.

## Replay saved outputs without rerunning the agent

[rejudge.py](../scripts/rejudge.py) is an executable replay tool. It loads saved
JSONL, copies each agent result, evaluates those same outputs, and writes a new
result set plus JSON/Markdown reports into a new directory. It never calls an
agent or edits a repository fixture.

```sh
# Offline wiring check: no judge, key, or model needed.
python scripts/rejudge.py --input saved.jsonl --rubric main.yaml \
  --output /tmp/jev-skipped --no-judge

# Live Jev evaluation: invokes TypeSafe and uses the configured API key.
python scripts/rejudge.py --input saved.jsonl --rubric main.yaml \
  --output /tmp/jev-results --judge-backend jev --judge-model jev-latest

# Revalidate/report the existing saved LLM judge; no new LLM calls.
python scripts/rejudge.py --input saved.jsonl --rubric main.yaml \
  --output /tmp/llm-baseline --judge-backend llm
```

Use a fresh output directory each time. JSON rubrics need only stdlib; YAML
rubrics need the project's existing `pyyaml` dependency. Rows need `case_id`,
`input`, and `agent_output`; use `subject` and `trial` to distinguish repeated
cases. Include `context`, `expected_output`, and inline `evidence` when available.
`gates` is a mapping of applicable gate names to booleans. Do not put file paths
in place of evidence content: the replay tool does not implicitly read them.

The comparison preserves `baseline_judge` and matches pairs by the same saved
subject/case/trial. Report paired coverage with score disagreement, not just an
average over survivors. Keep skipped/failed rows in totals. Judge latency and
usage are separate from agent usage; missing token counts or retry usage stay
unknown, never zero. No cost estimate is inferred from a hardcoded pricing table.

Human-reference agreement requires grades for that exact output and rubric.
A reference score attached only to the test-case sketch is not a human grade of
the candidate. Replay requires `reference_metadata.graded_by: human` and a
`graded_output_sha256` matching its recorded canonical `agent_output` hash, plus
`graded_rubric_sha256` matching the current rubric.
Use `agent_output_sha256` and `rubric_sha256` from `scripts/rejudge.py` at grading
time; replay records both hashes. Unmatched references are labeled unavailable
instead of silently calculating misleading agreement. Legacy saved LLM baselines
can still be paired within their original row, but missing rubric provenance is
reported as unverified. Explicit mismatched rubric hashes exclude a pair.

## Retry and validation behavior

The adapter uses a 30-second per-attempt timeout by default, at most three
attempts, and exponential backoff. It retries transport failures and HTTP
408/429/500/502/503/504/529. It does not retry authentication, request validation,
redirects, or malformed successful responses. `Retry-After` is honored up to a
30-second wait; a longer requested delay stops the run rather than retrying early.
Retries may consume additional billable work even when usage is unavailable.
Returned usage belongs to the final successful response, not an invented total
for failed attempts. All retries contribute to measured judge latency.

Run `python3 -m unittest discover -s tests -v`. Tests inject fake HTTP responses,
exercise mixed scales/fractional scores, and replay recorded rows without paid
calls. A separate live smoke test is needed to establish service compatibility;
these tests establish request/response handling and workflow behavior only.

## Official sources

API contract checked 2026-09-22:

1. [TypeSafe API](https://docs.typesafe.ai/api): endpoint, Score schema and errors.
2. [Models](https://docs.typesafe.ai/models): model selection and alias behavior.
3. [Score](https://docs.typesafe.ai/primitives/score): ordered levels and fractional scores.
4. [Confidence](https://docs.typesafe.ai/confidence): distribution-based confidence.
