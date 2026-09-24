"""Offline parsing, validation, and aggregation shared by judge backends.

``validate_rubric`` returns None or raises ValueError for configuration errors.
Judge failures are data: parsing and judge validation return tagged error dicts.
Missing backend means legacy ``llm``; ``jev`` must be explicitly identified.
No helper calls a model, changes agent metadata, or decides external gate status.
"""

import json
import math
import re
from typing import Any


def _json_safe(value: Any) -> bool:
    """Check the whole payload, including provider-specific metadata."""
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError, OverflowError, RecursionError):
        return False
    return True


def _diagnostic(value: Any) -> Any:
    """Keep JSON-safe raw data; stringify unsafe data without dropping a row."""
    if _json_safe(value):
        return value
    try:
        return repr(value)
    except Exception:
        return f"<unserializable {type(value).__name__}>"


def _reject_constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON constant: {value}")


def _finite_number(value: Any) -> bool:
    """Accept JSON numbers, excluding bool, NaN, infinity, and overflow."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def parse_judge_response(text: str) -> dict:
    """Extract an object from JSON, a code fence, or surrounding prose.

    Successfully decoded non-objects are failures, never containers to search
    for a nested judge. Failures return ``error: parse_failed`` and a raw excerpt.
    This function does not validate the judge schema.
    """
    failure = {"error": "parse_failed", "raw": text[:500] if isinstance(text, str) else None}
    if not isinstance(text, str):
        return failure

    def decode(candidate: str) -> dict:
        value = json.loads(candidate, parse_constant=_reject_constant)
        return value if isinstance(value, dict) and _json_safe(value) else failure

    try:
        return decode(text)
    except (ValueError, RecursionError):
        pass

    fence = re.search(r"```(?:json)?\s*(.*)\s*```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        try:
            return decode(fence.group(1))
        except (ValueError, RecursionError):
            # Do not salvage an object nested in malformed fenced JSON.
            return failure
    if "```" in text:
        return failure

    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        try:
            return decode(text[start:end + 1])
        except (ValueError, RecursionError):
            pass
    return failure


def validate_rubric(rubric: dict) -> None:
    """Raise ValueError for an invalid shared rubric; otherwise return None.

    Require nonempty, uniquely named dimensions; integer scales >= 2; positive
    finite weights summing to one (absolute tolerance 1e-9); and a finite
    pass_threshold in [0, 1]. Booleans are not numbers. Level descriptors and
    any backend-specific requirements are the caller's responsibility.
    """
    if not isinstance(rubric, dict):
        raise ValueError("rubric must be an object")
    dimensions = rubric.get("dimensions")
    if not isinstance(dimensions, list) or not dimensions:
        raise ValueError("dimensions must be a nonempty list")
    names, weights = set(), []
    for dim in dimensions:
        if not isinstance(dim, dict):
            raise ValueError("each dimension must be an object")
        name = dim.get("name")
        if not isinstance(name, str) or not name.strip() or name in names:
            raise ValueError("dimension names must be nonempty unique strings")
        names.add(name)
        scale = dim.get("scale")
        if (isinstance(scale, bool) or not isinstance(scale, int)
                or scale < 2 or not _finite_number(scale)):
            raise ValueError(f"{name}: scale must be a finite integer >= 2")
        weight = dim.get("weight")
        if not _finite_number(weight) or not 0 < weight <= 1:
            raise ValueError(f"{name}: weight must be finite and positive, at most 1")
        weights.append(weight)
    if not math.isclose(math.fsum(weights), 1.0, rel_tol=0, abs_tol=1e-9):
        raise ValueError("dimension weights must sum to 1")
    threshold = rubric.get("pass_threshold")
    if not _finite_number(threshold) or not 0 <= threshold <= 1:
        raise ValueError("pass_threshold must be a finite number in [0, 1]")


def validate_judge(parsed: dict, rubric: dict) -> dict:
    """Return a canonical judge or ``schema_failed``; never score partial data.

    Invalid rubric configuration raises ValueError. Existing tagged failures
    remain failures. Canonicalization does not mutate the input. LLM scores keep
    their numeric type; Jev scores become floats without rounding. Jev details
    require null reasoning/evidence/suggestion and numeric confidence in [0, 1].
    Confidence is required; missing or null confidence is a schema failure.

    Legacy LLM responses need no transport metadata: missing overall_reasoning,
    usage, and latency_ms become None. Optional model IDs must be strings when
    present. Unknown top-level fields (including raw_response) are preserved.
    """
    validate_rubric(rubric)
    return _validate_judge(parsed, rubric)


def _validate_judge(parsed: Any, rubric: dict) -> dict:
    """Validate a response against a rubric already checked by the caller."""
    backend = parsed.get("backend", "llm") if isinstance(parsed, dict) else "llm"

    def fail(reason: str, error: str = "schema_failed") -> dict:
        result = {"error": error, "reason": reason, "raw": _diagnostic(parsed)}
        if backend in ("llm", "jev"):
            result["backend"] = backend
        return result

    if not isinstance(parsed, dict):
        return fail("judge must be an object")
    if "error" in parsed:
        if isinstance(parsed["error"], str) and parsed["error"]:
            if not _json_safe(parsed):
                return fail("failed judge contains non-JSON or nonfinite values", parsed["error"])
            return dict(parsed)
        return fail("error must be a nonempty string on a failed judge")
    if backend not in ("llm", "jev"):
        return fail("backend must be llm or jev")

    dims = {dim["name"]: dim for dim in rubric["dimensions"]}
    scores, details = parsed.get("scores"), parsed.get("details")
    if not isinstance(scores, dict) or set(scores) != set(dims):
        return fail("scores must contain exactly the rubric dimension names")
    if not isinstance(details, dict) or set(details) != set(dims):
        return fail("details must contain exactly the rubric dimension names")
    for name, dim in dims.items():
        score = scores[name]
        if not _finite_number(score) or not 1 <= score <= dim["scale"]:
            return fail(f"{name}: score must be finite and within 1..{dim['scale']}")
        detail = details[name]
        if not isinstance(detail, dict):
            return fail(f"{name}: details must be an object")
        required = {"reasoning", "evidence", "suggestion", "confidence"}
        if not required.issubset(detail):
            return fail(f"{name}: missing explanation or confidence fields")
        confidence = detail["confidence"]
        if backend == "llm":
            if not isinstance(detail["reasoning"], str) or not isinstance(detail["suggestion"], str):
                return fail(f"{name}: reasoning and suggestion must be strings")
            evidence = detail["evidence"]
            if not isinstance(evidence, list) or not all(isinstance(item, str) for item in evidence):
                return fail(f"{name}: evidence must be a list of strings")
            if not isinstance(confidence, str) or confidence not in ("high", "medium", "low"):
                return fail(f"{name}: llm confidence must be high, medium, or low")
        else:
            if any(detail[key] is not None for key in ("reasoning", "evidence", "suggestion")):
                return fail(f"{name}: jev explanation fields must be null")
            if not _finite_number(confidence) or not 0 <= confidence <= 1:
                return fail(f"{name}: jev confidence must be numeric in [0, 1]")

    overall = parsed.get("overall_reasoning")
    if overall is not None and not isinstance(overall, str):
        return fail("overall_reasoning must be a string or null")
    usage = parsed.get("usage")
    if usage is not None and not isinstance(usage, dict):
        return fail("usage must be an object or null")
    for key in ("input_tokens", "output_tokens"):
        if usage is not None and key in usage:
            count = usage[key]
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                return fail(f"usage.{key} must be a nonnegative integer when present")
    latency = parsed.get("latency_ms")
    if latency is not None and (not _finite_number(latency) or latency < 0):
        return fail("latency_ms must be finite and nonnegative or null")
    for key in ("model_id", "resolved_model_id"):
        if key in parsed and not isinstance(parsed[key], str):
            return fail(f"{key} must be a string when present")
    if not _json_safe(parsed):
        return fail("judge contains non-JSON or nonfinite values")

    return {
        **parsed,
        "backend": backend,
        "scores": {name: float(score) if backend == "jev" else score for name, score in scores.items()},
        "details": {name: dict(detail) for name, detail in details.items()},
        "overall_reasoning": overall,
        "usage": usage,
        "latency_ms": latency,
    }


def compute_scores(results: list[dict], rubric: dict) -> dict:
    """Aggregate only complete, valid judges, retaining every row in n_total.

    Raw per-dimension means round to 2 decimals; the weighted mean rounds to 3,
    using sum(weight * score / scale). Missing dimensions are never reweighted.
    error_counts includes judge_missing, schema_failed, and existing error tags.
    These are quality statistics, not pass/fail decisions. External gates and
    agent failures remain authoritative in the harness; input rows are untouched.
    """
    validate_rubric(rubric)
    dimensions = rubric["dimensions"]
    per_dim = {dim["name"]: [] for dim in dimensions}
    weighted, errors = [], {}
    for row in results:
        if not isinstance(row, dict):
            judge = {"error": "schema_failed"}
        elif row.get("judge") is None:
            judge = {"error": "judge_missing"}
        else:
            judge = _validate_judge(row["judge"], rubric)
        if "error" in judge:
            tag = judge["error"]
            errors[tag] = errors.get(tag, 0) + 1
            continue
        scores = judge["scores"]
        for name in per_dim:
            per_dim[name].append(scores[name])
        weighted.append(math.fsum(
            dim["weight"] * (scores[dim["name"]] / dim["scale"])
            for dim in dimensions
        ))
    return {
        "per_dimension_avg": {
            name: round(math.fsum(value / len(values) for value in values), 2) if values else None
            for name, values in per_dim.items()
        },
        "weighted_overall": round(math.fsum(value / len(weighted) for value in weighted), 3) if weighted else None,
        "n_scored": len(weighted),
        "n_total": len(results),
        "error_counts": errors,
    }
