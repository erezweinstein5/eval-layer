"""Replay saved outputs without invoking an agent.

Examples (the first two commands are offline):
    python scripts/rejudge.py --input saved.jsonl --rubric rubric.json --output replay
    python scripts/rejudge.py --input saved.jsonl --rubric rubric.yaml --output audit --no-judge
    python scripts/rejudge.py --input saved.jsonl --rubric rubric.yaml --output jev-run \
        --judge-backend jev --judge-model MODEL_ID

The output directory must not exist. It receives results.jsonl, summary.json,
and report.md. LLM mode validates the saved judge; it has no live LLM adapter.
Jev mode calls only the judge, using TYPESAFE_API_KEY through jev_judge.

Reference provenance hashes the complete saved agent_output value (including
metadata when wrapped), as UTF-8 JSON with sorted keys, compact separators,
ensure_ascii=False, and allow_nan=False. Human references require that hash in
reference_metadata.graded_output_sha256, graded_rubric_sha256, and graded_by:
human. The rubric hash covers the complete rubric after JSON key normalization
(YAML integer level keys become strings), using the same canonical encoding.
A baseline is paired within its saved row; any explicit identity/hash must match.
Normalized differences use score / scale, with rubric weights across dimensions.
"""

import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import html
import json
import math
from pathlib import Path
import sys
import time

if __package__:
    from .judge_results import compute_scores, validate_judge, validate_rubric
else:
    from judge_results import compute_scores, validate_judge, validate_rubric


def _json(value, **kwargs):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, **kwargs)


def agent_output_sha256(agent_output):
    """Hash the exact saved JSON value, not a reference sketch or judge state."""
    canonical = _json(agent_output, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def rubric_sha256(rubric):
    """Hash the full rubric, including descriptors, weights and pass threshold.

    JSON round-tripping normalizes YAML integer keys before sorting. Duplicate
    keys created by normalization are rejected rather than hashed ambiguously.
    """
    validate_rubric(rubric)
    return agent_output_sha256(_load_json(_json(rubric)))


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"nonfinite JSON number: {value}")


def _load_json(text):
    value = json.loads(text, object_pairs_hook=_unique_object,
                       parse_constant=_reject_constant)
    _json(value)  # Also catches exponent overflow, e.g. 1e999.
    return value


def load_rubric(path):
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as exc:
            raise ValueError("YAML rubrics require PyYAML; JSON needs only stdlib") from exc
        try:
            rubric = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise ValueError(f"invalid YAML rubric: {exc}") from exc
    else:
        rubric = _load_json(text)
    validate_rubric(rubric)
    _json(rubric)
    return rubric


@dataclass(frozen=True)
class _UnreadableRow:
    raw_line: str
    reason: str


def load_rows(path):
    """Keep malformed physical JSONL lines as failures, including blank lines."""
    rows = []
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            text = line.rstrip("\r\n")
            try:
                rows.append(_load_json(text))
            except (ValueError, RecursionError) as exc:
                rows.append(_UnreadableRow(text, f"invalid JSONL: {exc}"))
    return rows


def _identity(row):
    """Missing optional subject/trial stay missing; never invent identifiers."""
    if not isinstance(row, dict):
        raise ValueError("saved row must be an object")
    case_id = row.get("case_id")
    if type(case_id) not in (str, int) or case_id == "":
        raise ValueError("case_id must be a nonempty string or integer")
    subject, trial = row.get("subject"), row.get("trial")
    if subject is not None and (not isinstance(subject, str) or not subject.strip()):
        raise ValueError("subject must be a nonempty string when present")
    if trial is not None and (type(trial) not in (str, int) or trial == ""):
        raise ValueError("trial must be a nonempty string or integer when present")
    return _json([subject, case_id, trial])


def _check_duplicates(rows):
    seen = {}
    for line, row in enumerate(rows, 1):
        try:
            identity = _identity(row)
        except ValueError:
            continue
        if identity in seen:
            raise ValueError(f"duplicate subject/case_id/trial at lines {seen[identity]} and {line}")
        seen[identity] = line


def gate_status(row):
    """Only an explicit mapping of booleans is accepted; absence adds no gate."""
    if "gates" not in row:
        return "not_applicable"
    gates = row["gates"]
    if (not isinstance(gates, dict)
            or any(not isinstance(k, str) or not k.strip() or type(v) is not bool
                   for k, v in gates.items())):
        return "malformed"
    if not gates:
        return "not_applicable"
    return "passed" if all(gates.values()) else "failed"


def _agent_failed(row):
    containers = [row]
    for field in ("agent_result", "agent_metadata"):
        if isinstance(row.get(field), dict):
            containers.append(row[field])
    output = row.get("agent_output")
    if isinstance(output, dict) and "recommendation" in output:
        containers.append(output)
    return any(container.get(key) is not None and container.get(key) != ""
               for container in containers for key in ("error", "agent_error"))


def build_state(row):
    """Allowlist saved evidence; do not open paths, stringify fields, or add labels."""
    if "input" not in row or row["input"] is None:
        raise ValueError("missing saved input")
    if "agent_output" not in row or row["agent_output"] is None:
        raise ValueError("missing saved agent_output")
    output = row["agent_output"]
    if isinstance(output, dict) and "recommendation" in output:
        output = output["recommendation"]
        if output is None:
            raise ValueError("missing saved agent_output.recommendation")
    state = {key: deepcopy(row[key]) for key in
             ("input", "context", "expected_output", "evidence") if key in row}
    state["agent_output"] = deepcopy(output)
    _json(state)
    return state


def _evaluate_jev(state, rubric, model_id, review_threshold):
    # Keep provider modules completely outside the offline/no-judge import path.
    if __package__:
        from .jev_judge import evaluate
    else:
        from jev_judge import evaluate
    return evaluate(state, rubric, model_id=model_id, review_threshold=review_threshold)


def _checked_judge(judge, rubric, backend, expected_model=None):
    if isinstance(judge, dict) and judge.get("backend", "llm") != backend:
        checked = {"backend": backend, "error": "backend_mismatch",
                   "reason": f"expected {backend} judge, found {judge.get('backend')!r}"}
    elif judge is None:
        checked = {"backend": backend, "error": "judge_missing",
                   "reason": "no saved LLM judge to reuse" if backend == "llm" else "judge response missing"}
    else:
        try:
            _json(judge)
            checked = validate_judge(judge, rubric)
        except (ValueError, TypeError, OverflowError, RecursionError):
            checked = {"backend": backend, "error": "schema_failed",
                       "reason": "judge is not finite JSON"}
    if "error" in checked:
        checked["backend"] = backend
        if isinstance(judge, dict):
            for key in ("model_id", "resolved_model_id", "latency_ms", "usage"):
                value = judge.get(key)
                valid = (isinstance(value, str) if key.endswith("model_id")
                         else type(value) in (int, float) and 0 <= value < math.inf
                         if key == "latency_ms" else isinstance(value, dict))
                if valid and key not in checked:
                    try:
                        _json(value)
                    except (ValueError, TypeError, OverflowError, RecursionError):
                        continue
                    checked[key] = deepcopy(value)
        if expected_model is not None:
            checked.setdefault("model_id", expected_model)
    return checked


def _weighted(scores, rubric):
    return math.fsum(dim["weight"] * (scores[dim["name"]] / dim["scale"])
                     for dim in rubric["dimensions"])


def judge_identity(judge):
    judge = judge if isinstance(judge, dict) else {}
    return {
        "backend": judge.get("backend", "llm") if judge else None,
        "requested_model_id": judge.get("model_id"),
        "resolved_model_id": judge.get("resolved_model_id"),
    }


def _difference(scores, reference, rubric):
    deltas = {dim["name"]: (scores[dim["name"]] - reference[dim["name"]]) / dim["scale"]
              for dim in rubric["dimensions"]}
    return {
        "normalized_mae": math.fsum(dim["weight"] * abs(deltas[dim["name"]])
                                    for dim in rubric["dimensions"]),
        "signed_normalized_delta": math.fsum(dim["weight"] * deltas[dim["name"]]
                                             for dim in rubric["dimensions"]),
        "per_dimension": deltas,
    }


def _baseline_comparison(row, judge, rubric, output_hash, rubric_hash):
    unavailable = {"status": "unavailable", "reason": "current_judge_unavailable"}
    if judge is None or "error" in judge:
        return unavailable
    baseline = row.get("baseline_judge")
    if baseline is None:
        return {**unavailable, "reason": "baseline_judge_missing"}
    checked = validate_judge(baseline, rubric)
    if "error" in checked:
        return {**unavailable, "reason": "baseline_judge_invalid"}
    metadata = row.get("baseline_judge_metadata", {})
    if not isinstance(metadata, dict):
        return {"status": "unmatched", "reason": "baseline_metadata_invalid"}
    # A row-local saved baseline is paired by construction. Respect any stronger
    # provenance assertion instead of silently comparing another output/trial.
    rubric_verified = False
    for source in (baseline, metadata):
        for key in ("subject", "case_id", "trial"):
            if key in source and source[key] != row.get(key):
                return {"status": "unmatched", "reason": "baseline_identity_mismatch"}
        for key in ("graded_output_sha256", "agent_output_sha256"):
            if key in source and source[key] != output_hash:
                return {"status": "unmatched", "reason": "baseline_output_mismatch"}
        for key in ("graded_rubric_sha256", "rubric_sha256"):
            if key in source:
                if source[key] != rubric_hash:
                    return {"status": "unmatched", "reason": "baseline_rubric_mismatch",
                            "rubric_provenance": "mismatch"}
                rubric_verified = True
    return {
        "status": "matched",
        "rubric_provenance": "verified" if rubric_verified else "unverified",
        "baseline": judge_identity(checked),
        "current": judge_identity(judge),
        **_difference(judge["scores"], checked["scores"], rubric),
    }


def _human_comparison(row, judge, rubric, output_hash, rubric_hash):
    metadata = row.get("reference_metadata")
    grader = metadata.get("graded_by") if isinstance(metadata, dict) else None
    result = {"status": "unavailable", "graded_by": grader,
              "reason": "human_reference_missing"}
    if row.get("reference_scores") is None:
        return result
    if not isinstance(grader, str) or not grader.strip():
        return {**result, "reason": "grader_unavailable"}
    if grader != "human":
        return {**result, "reason": "reference_not_human"}
    if not output_hash or metadata.get("graded_output_sha256") != output_hash:
        return {**result, "status": "unmatched", "reason": "reference_output_unmatched"}
    if metadata.get("graded_rubric_sha256") != rubric_hash:
        return {**result, "status": "unmatched", "reason": "reference_rubric_unmatched"}
    reference = row["reference_scores"]
    dimensions = rubric["dimensions"]
    if not isinstance(reference, dict) or set(reference) != {d["name"] for d in dimensions}:
        return {**result, "reason": "reference_scores_invalid"}
    for dim in dimensions:
        value = reference[dim["name"]]
        if type(value) not in (int, float) or not 1 <= value <= dim["scale"]:
            return {**result, "reason": "reference_scores_invalid"}
    if judge is None or "error" in judge:
        return {**result, "reason": "current_judge_unavailable"}
    return {"status": "matched", "graded_by": grader, "rubric_provenance": "verified",
            **_difference(judge["scores"], reference, rubric)}


def replay_rows(rows, rubric, *, judge_backend="llm", judge_model=None,
                no_judge=False, review_threshold=0.5):
    """Return independent replay rows; malformed records remain visible failures."""
    validate_rubric(rubric)
    rubric_hash = rubric_sha256(rubric)
    if judge_backend not in ("llm", "jev"):
        raise ValueError("judge_backend must be llm or jev")
    if not no_judge and judge_backend == "jev" and (
            not isinstance(judge_model, str) or not judge_model.strip()):
        raise ValueError("--judge-model is required for live Jev replay")
    if type(review_threshold) not in (int, float) or not 0 <= review_threshold <= 1:
        raise ValueError("--review-threshold must be finite and between 0 and 1")
    rows = list(rows)
    _check_duplicates(rows)  # Entire input is checked before any paid request.
    results = []
    for line, original in enumerate(rows, 1):
        if isinstance(original, _UnreadableRow):
            row = {"source_raw_line": original.raw_line}
            invalid = original.reason
        elif not isinstance(original, dict):
            row = {"source_row": deepcopy(original)}
            invalid = "saved row must be an object"
        else:
            row = deepcopy(original)
            try:
                _identity(row)
                _json(row)
                invalid = None
            except (ValueError, TypeError, RecursionError) as exc:
                invalid = str(exc)
        row["baseline_judge"] = deepcopy(row.get("baseline_judge", row.get("judge")))
        row["judge"] = None
        gate = gate_status(row)
        info = {"source_line": line, "backend": judge_backend, "gate_status": gate,
                "requested_model_id": judge_model if judge_backend == "jev" else None,
                "review_threshold": review_threshold if judge_backend == "jev" else None,
                "rubric_sha256": rubric_hash,
                "agent_output_sha256": None, "weighted_score": None,
                "status": "invalid_record", "reason": invalid, "passed": False}
        if not invalid and _agent_failed(row):
            info.update(status="agent_failed", reason="saved agent error; judge skipped")
        elif not invalid:
            try:
                state = build_state(row)
                info["agent_output_sha256"] = agent_output_sha256(row["agent_output"])
            except (ValueError, TypeError, RecursionError) as exc:
                info.update(reason=str(exc))
            else:
                if no_judge:
                    info.update(status="judge_skipped", reason="--no-judge",
                                passed=False if gate in ("failed", "malformed") else None)
                else:
                    started = time.perf_counter()
                    try:
                        candidate = (original.get("judge") if judge_backend == "llm"
                                     else _evaluate_jev(state, deepcopy(rubric), judge_model,
                                                        review_threshold))
                        judge = _checked_judge(candidate, rubric, judge_backend,
                                               expected_model=judge_model if judge_backend == "jev" else None)
                    except Exception as exc:
                        judge = {"backend": judge_backend, "error": "judge_exception",
                                 "reason": type(exc).__name__,
                                 "latency_ms": (time.perf_counter() - started) * 1000}
                        if judge_backend == "jev":
                            judge["model_id"] = judge_model
                    if judge_backend == "jev":
                        judge.setdefault("model_id", judge_model)
                        judge["rubric_sha256"] = rubric_hash
                    row["judge"] = judge
                    if "error" in judge:
                        info.update(status="judge_failed", reason=judge["error"])
                    else:
                        weighted = _weighted(judge["scores"], rubric)
                        info.update(status="judged", reason=None, weighted_score=weighted,
                                    passed=gate not in ("failed", "malformed")
                                    and weighted >= rubric["pass_threshold"])
                        info["requested_model_id"] = judge.get("model_id")
        info["baseline_comparison"] = _baseline_comparison(
            row, row["judge"], rubric, info["agent_output_sha256"], rubric_hash)
        info["human_reference"] = _human_comparison(
            row, row["judge"], rubric, info["agent_output_sha256"], rubric_hash)
        row["rejudge"] = info
        row["judge_status"] = info["status"]
        row["passed"] = info["passed"]
        results.append(row)
    return results


def _mean(values):
    return math.fsum(value / len(values) for value in values) if values else None


def _pair_summary(comparisons, rubric):
    matched = [value for value in comparisons if value["status"] == "matched"]
    return {
        "paired_n": len(matched),
        "unmatched_n": len(comparisons) - len(matched),
        "normalized_mae": _mean([value["normalized_mae"] for value in matched]),
        "signed_normalized_delta": _mean([value["signed_normalized_delta"] for value in matched]),
        "per_dimension": {
            dim["name"]: {
                "normalized_mae": _mean([abs(value["per_dimension"][dim["name"]]) for value in matched]),
                "signed_normalized_delta": _mean([value["per_dimension"][dim["name"]] for value in matched]),
            } for dim in rubric["dimensions"]
        },
        "unavailable_reasons": dict(Counter(
            value["reason"] for value in comparisons if value["status"] != "matched")),
        "paired_rubric_provenance": dict(Counter(
            value["rubric_provenance"] for value in matched)),
    }


def _aggregate(rows, rubric, no_judge):
    scores = compute_scores(rows, rubric)
    n_passed = sum(row["passed"] is True for row in rows)
    baseline = [row["rejudge"]["baseline_comparison"] for row in rows]
    human = [row["rejudge"]["human_reference"] for row in rows]
    pairs = {}
    for comparison in baseline:
        if comparison["status"] == "matched":
            key = _json([comparison["baseline"], comparison["current"]], sort_keys=True)
            pairs.setdefault(key, []).append(comparison)
    baseline_summary = _pair_summary(baseline, rubric)
    baseline_summary["by_model_pair"] = [
        {"baseline": values[0]["baseline"], "current": values[0]["current"],
         **_pair_summary(values, rubric)} for values in pairs.values()
    ]
    human_summary = _pair_summary(human, rubric)
    human_summary.update(
        label="MAE and leniency vs human-graded references for the exact saved output and rubric",
        leniency=human_summary["signed_normalized_delta"],
        status="available" if human_summary["paired_n"] else "unavailable",
    )
    return {
        **scores,
        "score_population": "valid judges only; pass_rate uses all saved rows",
        "n_passed": n_passed,
        "n_failed": sum(row["passed"] is False for row in rows),
        "n_unassessed": sum(row["passed"] is None for row in rows),
        "pass_rate": n_passed / len(rows) if rows and not no_judge else None,
        "status_counts": dict(Counter(row["rejudge"]["status"] for row in rows)),
        "gate_counts": dict(Counter(row["rejudge"]["gate_status"] for row in rows)),
        "baseline_comparison": baseline_summary,
        "human_reference": human_summary,
    }


def _record_report(row):
    info, judge = row["rejudge"], row["judge"]
    judge = judge if isinstance(judge, dict) else {}
    baseline = row["baseline_judge"] if isinstance(row["baseline_judge"], dict) else {}
    details = judge.get("details", {}) if "error" not in judge else {}
    return {
        **{key: row.get(key) for key in ("subject", "case_id", "trial")},
        **deepcopy(info),
        "judge_identity": judge_identity(judge),
        "baseline_identity": judge_identity(row["baseline_judge"]),
        "baseline_latency_ms": baseline.get("latency_ms"),
        "baseline_usage": deepcopy(baseline.get("usage")),
        "latency_ms": judge.get("latency_ms"),
        "usage": judge.get("usage"),
        "confidence": {name: detail["confidence"] for name, detail in details.items()},
        "review_dimensions": judge.get("review_dimensions"),
        "review_threshold": judge.get("review_threshold", info["review_threshold"]),
        "explanations": {
            name: {key: detail.get(key) for key in ("reasoning", "evidence", "suggestion")}
            for name, detail in details.items()
        },
    }


def summarize(rows, rubric, *, judge_backend="llm", judge_model=None, no_judge=False,
              review_threshold=0.5):
    subjects = {}
    for row in rows:
        # Malformed subjects may be objects; JSON keys preserve them in reports.
        subjects.setdefault(_json(row.get("subject"), sort_keys=True), []).append(row)
    return {
        "judge_backend": judge_backend,
        "requested_judge_model": judge_model if judge_backend == "jev" else None,
        "review_threshold": review_threshold if judge_backend == "jev" else None,
        "mode": "no-judge" if no_judge else ("live-judge" if judge_backend == "jev" else "saved-judge"),
        "normalization": "score / scale; rubric-weighted per-dimension differences",
        "rubric_sha256": rubric_sha256(rubric),
        "hash_method": {
            "algorithm": "sha256", "encoding": "utf-8", "sort_keys": True,
            "separators": [",", ":"], "ensure_ascii": False, "allow_nan": False,
            "output_value": "complete saved agent_output",
            "rubric_value": "complete current rubric after JSON key normalization",
        },
        **_aggregate(rows, rubric, no_judge),
        "subjects": [
            {"subject": values[0].get("subject"), **_aggregate(values, rubric, no_judge)}
            for values in subjects.values()
        ],
        "records": [_record_report(row) for row in rows],
    }


def _cell(value):
    if value is None:
        return "unavailable"
    text = value if isinstance(value, str) else _json(value)
    return html.escape(text).replace("|", "&#124;").replace("\n", "&#10;").replace("\r", "&#13;")


def render_markdown(summary):
    lines = [
        "# Saved-output judge replay", "",
        f"Mode: {_cell(summary['mode'])}; backend: {_cell(summary['judge_backend'])}; "
        f"requested model: {_cell(summary['requested_judge_model'])}.",
        f"Review threshold: {_cell(summary['review_threshold'])}.",
        "",
        f"Passed: {summary['n_passed']}/{summary['n_total']} saved rows. "
        f"Scored: {summary['n_scored']}; failed: {summary['n_failed']}; "
        f"unassessed: {summary['n_unassessed']}. Pass rate: {_cell(summary['pass_rate'])}.",
        "Score averages cover valid judges only. Pass-rate denominators retain every saved row.",
        f"Status counts: {_cell(summary['status_counts'])}. Gates: {_cell(summary['gate_counts'])}.",
        "",
        "| Subject | Rows | Scored | Passed | Pass rate | Weighted score (scored only) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for subject in summary["subjects"]:
        lines.append("| " + " | ".join(_cell(subject[key]) for key in
                     ("subject", "n_total", "n_scored", "n_passed", "pass_rate", "weighted_overall")) + " |")
    lines.extend(["", f"Normalization: {summary['normalization']}.",
                  f"Current rubric SHA-256: {summary['rubric_sha256']}.",
                  f"Hash method: {_cell(summary['hash_method'])}.",
                  "Legacy baseline rubric provenance is unverified unless an explicit hash matches; "
                  "its paired disagreement does not establish human agreement.", ""])
    for label, value in (("Baseline disagreement", summary["baseline_comparison"]),
                         (summary["human_reference"]["label"], summary["human_reference"])):
        lines.extend([
            f"## {label}", "",
            f"Paired n: {value['paired_n']}; unavailable/unmatched: {value['unmatched_n']}. "
            f"Normalized MAE: {_cell(value['normalized_mae'])}; "
            f"{'leniency' if 'leniency' in value else 'signed normalized delta'}: "
            f"{_cell(value['signed_normalized_delta'])}.",
            f"Unavailable reasons: {_cell(value['unavailable_reasons'])}.",
            f"Paired rubric provenance: {_cell(value['paired_rubric_provenance'])}.",
            f"Per dimension: {_cell(value['per_dimension'])}.", "",
        ])
    for pair in summary["baseline_comparison"]["by_model_pair"]:
        lines.append(f"- Baseline {_cell(pair['baseline'])}; current {_cell(pair['current'])}; "
                     f"paired n: {pair['paired_n']}; normalized MAE: {_cell(pair['normalized_mae'])}.")
    lines.extend(["", "## Saved rows", ""])
    for record in summary["records"]:
        identity = {key: record[key] for key in ("subject", "case_id", "trial")}
        lines.extend([
            f"### Row {record['source_line']}: {_cell(identity)}", "",
            f"Status: {_cell(record['status'])}; reason: {_cell(record['reason'])}; "
            f"gate: {_cell(record['gate_status'])}; passed: {_cell(record['passed'])}.",
            f"Judge: {_cell(record['judge_identity'])}; baseline: {_cell(record['baseline_identity'])}.",
            f"Latency ms: {_cell(record['latency_ms'])}; usage: {_cell(record['usage'])}.",
            f"Baseline latency ms: {_cell(record['baseline_latency_ms'])}; "
            f"baseline usage: {_cell(record['baseline_usage'])}.",
            f"Confidence: {_cell(record['confidence'])}; review threshold: {_cell(record['review_threshold'])}; "
            f"review dimensions: {_cell(record['review_dimensions'])}.",
            f"Baseline pairing: {_cell(record['baseline_comparison'])}.",
            f"Human reference: {_cell(record['human_reference'])}.", "",
            "| Dimension | Reasoning | Evidence | Suggestion |",
            "| --- | --- | --- | --- |",
        ])
        if not record["explanations"]:
            lines.append("| unavailable | unavailable | unavailable | unavailable |")
        for name, detail in record["explanations"].items():
            lines.append("| " + " | ".join(_cell(value) for value in
                         (name, detail["reasoning"], detail["evidence"], detail["suggestion"])) + " |")
        lines.append("")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, type=Path, help="saved JSONL; never reruns the agent")
    parser.add_argument("--rubric", required=True, type=Path, help="JSON or YAML rubric")
    parser.add_argument("--output", required=True, type=Path, help="NEW directory for replay reports")
    parser.add_argument("--judge-backend", choices=("llm", "jev"), default="llm",
                        help="llm reuses saved LLM judges; jev calls the Jev judge")
    parser.add_argument("--judge-model", help="required for live Jev; LLM identity comes from saved judges")
    parser.add_argument("--no-judge", action="store_true", help="offline audit; skip all judge calls")
    parser.add_argument("--review-threshold", type=float, default=0.5,
                        help="flag Jev dimensions below this confidence (default: 0.5)")
    args = parser.parse_args(argv)
    if args.judge_backend == "jev" and not args.no_judge and (
            not args.judge_model or not args.judge_model.strip()):
        parser.error("--judge-model is required for live Jev replay")
    if not 0 <= args.review_threshold <= 1:
        parser.error("--review-threshold must be finite and between 0 and 1")
    try:
        rubric = load_rubric(args.rubric)
        rows = load_rows(args.input)
        _check_duplicates(rows)
        # Exclusive creation happens before the first judge call.
        args.output.mkdir(parents=True, exist_ok=False)
        results = replay_rows(rows, rubric, judge_backend=args.judge_backend,
                              judge_model=args.judge_model, no_judge=args.no_judge,
                              review_threshold=args.review_threshold)
        summary = summarize(results, rubric, judge_backend=args.judge_backend,
                            judge_model=args.judge_model, no_judge=args.no_judge,
                            review_threshold=args.review_threshold)
        with (args.output / "results.jsonl").open("x", encoding="utf-8") as stream:
            for row in results:
                stream.write(_json(row) + "\n")
        (args.output / "summary.json").write_text(_json(summary, indent=2) + "\n", encoding="utf-8")
        (args.output / "report.md").write_text(render_markdown(summary), encoding="utf-8")
    except (OSError, ValueError, TypeError, RecursionError) as exc:
        print(f"rejudge: {exc}", file=sys.stderr)
        return 2
    print(f"Saved {len(results)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
