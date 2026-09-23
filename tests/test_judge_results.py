"""Offline contract tests; no credentials, provider SDKs, or network calls."""

from copy import deepcopy
import json
import math
import unittest

from scripts.judge_results import (
    compute_scores,
    parse_judge_response,
    validate_judge,
    validate_rubric,
)


def rubric():
    return {
        "dimensions": [
            {"name": "quality", "scale": 5, "weight": 0.75},
            {"name": "format", "scale": 3, "weight": 0.25},
        ],
        "pass_threshold": 0.7,
    }


def llm():
    return {
        "scores": {"quality": 4, "format": 3},
        "details": {
            name: {
                "reasoning": "Matches the descriptor.",
                "evidence": ["The requested field is present."],
                "suggestion": "Include an example.",
                "confidence": "high",
            }
            for name in ("quality", "format")
        },
        "overall_reasoning": "Meets the main requirements.",
    }


def jev():
    return {
        "backend": "jev",
        "scores": {"quality": 4.25, "format": 2},
        "details": {
            name: {
                "reasoning": None, "evidence": None, "suggestion": None,
                "confidence": 0.8,
            }
            for name in ("quality", "format")
        },
        "overall_reasoning": None,
        "model_id": "requested-model",
        "resolved_model_id": "resolved-model",
        "usage": {"input_tokens": 10},
        "latency_ms": 20.5,
        "raw_response": {"provider_score": 4.25},
    }


class ParseJudgeTests(unittest.TestCase):
    def test_direct_fenced_and_prose_objects(self):
        response = llm()
        response["details"]["quality"]["reasoning"] = "Literal {braces} and ``` in text."
        raw = json.dumps(response)
        # Code fences inside JSON strings are safe for direct JSON.
        self.assertEqual(parse_judge_response(raw), response)
        for wrapped in (raw, f"```json\n{raw}\n```", f"```\n{raw}\n```",
                        f"```JSON\n{raw}\n```"):
            with self.subTest(wrapped=wrapped[:20]):
                self.assertEqual(parse_judge_response(wrapped), response)
        self.assertEqual(parse_judge_response(f"Here is the grade:\n{json.dumps(llm())}\nDone."), llm())

    def test_all_json_nonobjects_fail_even_when_containing_an_object(self):
        for value in (None, True, False, 0, 1.5, "", "text", [], [llm()]):
            for wrapped in (json.dumps(value), f"```json\n{json.dumps(value)}\n```"):
                with self.subTest(wrapped=wrapped[:40]):
                    self.assertEqual(parse_judge_response(wrapped)["error"], "parse_failed")
        self.assertEqual(parse_judge_response(json.dumps(json.dumps(llm())))["error"], "parse_failed")

    def test_malformed_and_nontext_inputs_are_tagged(self):
        for value in (None, True, 17, [], {}, b"{}", "", "no JSON", "{", '{"a": 1,}',
                      '```json\n[{"scores": {}}', '```json\n[{"scores": {}} broken]\n```'):
            with self.subTest(value=value):
                self.assertEqual(parse_judge_response(value)["error"], "parse_failed")

    def test_parse_failure_excerpt_is_bounded(self):
        self.assertEqual(len(parse_judge_response("x" * 1000)["raw"]), 500)

    def test_empty_object_parses_but_does_not_validate(self):
        self.assertEqual(parse_judge_response("{}"), {})
        self.assertEqual(validate_judge({}, rubric())["error"], "schema_failed")

    def test_nonfinite_json_is_a_serializable_parse_failure(self):
        for token in ("NaN", "Infinity", "-Infinity", "1e999"):
            raw = json.dumps(llm()).replace('"quality": 4', f'"quality": {token}', 1)
            for text in (raw, f"```json\n{raw}\n```", f"Grade:\n{raw}"):
                with self.subTest(token=token, prefix=text[:10]):
                    failure = parse_judge_response(text)
                    self.assertEqual(failure["error"], "parse_failed")
                    json.dumps(failure, allow_nan=False)
        raw = json.dumps(llm())[:-1] + ', "extra": {"value": NaN}}'
        self.assertEqual(parse_judge_response(raw)["error"], "parse_failed")


class RubricTests(unittest.TestCase):
    def test_valid_rubric_returns_none_without_mutation(self):
        value = rubric()
        before = deepcopy(value)
        self.assertIsNone(validate_rubric(value))
        self.assertEqual(value, before)

    def test_shape_and_name_validation(self):
        for value in (None, [], {}, {"dimensions": []}, {"dimensions": {}},
                      {"dimensions": [None]}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_rubric(value)
        for name in (None, "", " ", True, 1, [], "quality"):
            value = rubric()
            value["dimensions"][1]["name"] = name
            with self.subTest(name=name), self.assertRaises(ValueError):
                validate_rubric(value)

    def test_scales_are_integers_at_least_two(self):
        for scale in (None, True, False, "5", 5.0, 1, 0, -1, math.nan, math.inf, 10 ** 1000):
            value = rubric()
            value["dimensions"][0]["scale"] = scale
            with self.subTest(scale=scale), self.assertRaises(ValueError):
                validate_rubric(value)
        value = rubric()
        value["dimensions"][0]["scale"] = 2
        self.assertIsNone(validate_rubric(value))

    def test_positive_finite_weights_and_sum(self):
        for weight in (None, True, "0.75", 0, -0.1, math.nan, math.inf, 10 ** 1000, 0.5):
            value = rubric()
            value["dimensions"][0]["weight"] = weight
            with self.subTest(weight=weight), self.assertRaises(ValueError):
                validate_rubric(value)
        value = rubric()
        value["dimensions"][0]["weight"] = 0.75000000001
        self.assertIsNone(validate_rubric(value))

    def test_threshold_required_finite_and_inclusive(self):
        value = rubric()
        del value["pass_threshold"]
        with self.assertRaises(ValueError):
            validate_rubric(value)
        for threshold in (None, True, "0.7", -0.01, 1.01, math.nan, math.inf, 10 ** 1000):
            value = rubric()
            value["pass_threshold"] = threshold
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                validate_rubric(value)
        for threshold in (0, 1):
            value = rubric()
            value["pass_threshold"] = threshold
            self.assertIsNone(validate_rubric(value))

    def test_levels_are_left_to_backend_validation(self):
        value = rubric()
        value["dimensions"][0]["levels"] = {"caller": "validates this"}
        self.assertIsNone(validate_rubric(value))


class ValidateJudgeTests(unittest.TestCase):
    def assert_invalid(self, value):
        result = validate_judge(value, rubric())
        self.assertEqual(result["error"], "schema_failed")
        self.assertIn("reason", result)
        json.dumps(result, allow_nan=False)

    def test_legacy_llm_is_canonicalized_without_losing_fields(self):
        value = llm()
        before = deepcopy(value)
        result = validate_judge(value, rubric())
        self.assertEqual(result["backend"], "llm")
        self.assertIsNone(result["usage"])
        self.assertIsNone(result["latency_ms"])
        for key, content in before.items():
            self.assertEqual(result[key], content)
        self.assertEqual(value, before)
        self.assertEqual(result, validate_judge(result, rubric()))

    def test_jev_floats_null_explanations_and_metadata(self):
        value = jev()
        before = deepcopy(value)
        result = validate_judge(value, rubric())
        self.assertNotIn("error", result)
        self.assertEqual(result, before)
        self.assertTrue(all(type(score) is float for score in result["scores"].values()))
        self.assertEqual(result["scores"]["quality"], 4.25)
        self.assertEqual(value, before)
        self.assertEqual(result, validate_judge(result, rubric()))

    def test_nonobject_payloads_and_unknown_backends(self):
        for value in (None, [], [llm()], "", True, 1):
            with self.subTest(value=value):
                self.assert_invalid(value)
        for backend in (None, "", "other", True, [], {}):
            value = llm()
            value["backend"] = backend
            with self.subTest(backend=backend):
                self.assert_invalid(value)

    def test_failure_tags_are_preserved_and_bad_tags_rejected(self):
        value = {"backend": "jev", "error": "http_failed", "scores": llm()["scores"]}
        self.assertEqual(validate_judge(value, rubric()), value)
        for error in (None, "", [], {}, True):
            value = llm()
            value["error"] = error
            self.assert_invalid(value)

    def test_scores_and_details_require_exact_dimension_keys(self):
        for make in (llm, jev):
            for field in ("scores", "details"):
                for replacement in (None, [], {}, {"quality": 4}, {"quality": 4, "other": 3}):
                    value = make()
                    value[field] = replacement
                    with self.subTest(backend=make.__name__, field=field, value=replacement):
                        self.assert_invalid(value)
                value = make()
                value[field]["extra"] = value[field]["quality"]
                self.assert_invalid(value)

    def test_scores_reject_bool_nonfinite_and_out_of_range(self):
        for make in (llm, jev):
            for score in (None, True, False, "4", [], {}, math.nan, math.inf, -math.inf,
                          10 ** 1000, 0, -1, 5.01):
                value = make()
                value["scores"]["quality"] = score
                with self.subTest(backend=make.__name__, score=score):
                    self.assert_invalid(value)
            value = make()
            value["scores"]["format"] = 3.01
            self.assert_invalid(value)

    def test_fractional_scores_and_scale_endpoints_are_valid_for_both_backends(self):
        for make in (llm, jev):
            for quality, format_score in ((1, 1), (5, 3), (3.625, 2.25)):
                value = make()
                value["scores"] = {"quality": quality, "format": format_score}
                result = validate_judge(value, rubric())
                self.assertNotIn("error", result)
                self.assertEqual(result["scores"], value["scores"])

    def test_detail_objects_and_required_fields(self):
        for make in (llm, jev):
            for detail in (None, [], {}, True):
                value = make()
                value["details"]["quality"] = detail
                self.assert_invalid(value)
            for key in ("reasoning", "evidence", "suggestion", "confidence"):
                value = make()
                del value["details"]["quality"][key]
                self.assert_invalid(value)

    def test_llm_explanation_and_confidence_types(self):
        for field, invalid in (
            ("reasoning", (None, [], 1)),
            ("suggestion", (None, {}, 1)),
            ("evidence", (None, "quote", {}, [1], [None])),
            ("confidence", (None, [], {}, "HIGH", "unknown", 0.8, True)),
        ):
            for content in invalid:
                value = llm()
                value["details"]["quality"][field] = content
                self.assert_invalid(value)
        for confidence in ("high", "medium", "low"):
            value = llm()
            value["details"]["quality"]["confidence"] = confidence
            self.assertNotIn("error", validate_judge(value, rubric()))

    def test_jev_requires_explicit_backend_and_null_explanations(self):
        value = jev()
        del value["backend"]
        self.assert_invalid(value)
        for key in ("reasoning", "evidence", "suggestion"):
            for content in ("invented explanation", [], 0):
                value = jev()
                value["details"]["quality"][key] = content
                self.assert_invalid(value)

    def test_jev_confidence_range_types_and_required_value(self):
        for confidence in (0, 1, 0.456):
            value = jev()
            value["details"]["quality"]["confidence"] = confidence
            self.assertNotIn("error", validate_judge(value, rubric()))
        for confidence in (None, "high", "0.8", True, False, [], {}, -0.001, 1.001,
                           math.nan, math.inf, -math.inf, 10 ** 1000):
            value = jev()
            value["details"]["quality"]["confidence"] = confidence
            self.assert_invalid(value)

    def test_metadata_types_and_unknown_values(self):
        for field, invalid in (
            ("overall_reasoning", ([], {}, 7)),
            ("usage", ([], 0, "unknown")),
            ("latency_ms", (True, "5", -1, math.nan, math.inf)),
            ("model_id", (None, [], 1)),
            ("resolved_model_id", (None, {}, 1)),
        ):
            for content in invalid:
                value = jev()
                value[field] = content
                self.assert_invalid(value)
        value = llm()
        del value["overall_reasoning"]
        value.update(backend="llm", usage=None, latency_ms=0)
        result = validate_judge(value, rubric())
        self.assertIsNone(result["overall_reasoning"])
        self.assertEqual(result["latency_ms"], 0)

    def test_usage_token_counts_are_nonnegative_integers_when_present(self):
        for make in (llm, jev):
            for key in ("input_tokens", "output_tokens"):
                for count in (None, True, False, -1, 1.5, 2.0, "2", [], math.nan, math.inf):
                    value = make()
                    value["usage"] = {key: count}
                    with self.subTest(backend=make.__name__, key=key, count=count):
                        self.assert_invalid(value)
                for count in (0, 1):
                    value = make()
                    value["usage"] = {key: count}
                    self.assertEqual(validate_judge(value, rubric())["usage"], {key: count})
            for usage in (None, {}, {"provider_note": "No token counts reported"}):
                value = make()
                value["usage"] = usage
                self.assertNotIn("error", validate_judge(value, rubric()))

    def test_full_payload_rejects_unsafe_extras_with_serializable_diagnostics(self):
        for bad in (math.nan, math.inf, -math.inf, object(), {1, 2}):
            for field in ("raw_response", "provider_metadata"):
                value = jev()
                value[field] = {"nested": [bad]}
                result = validate_judge(value, rubric())
                self.assertEqual(result["error"], "schema_failed")
                self.assertIsInstance(result["raw"], str)
                json.dumps(result, allow_nan=False)
            value = llm()
            value["details"]["quality"]["extra"] = bad
            self.assert_invalid(value)
            value = llm()
            value["usage"] = {"extra": bad}
            self.assert_invalid(value)

    def test_unsafe_tagged_failures_retain_tag_and_diagnostic(self):
        for bad in (math.nan, math.inf, object()):
            failure = {"backend": "jev", "error": "http_failed", "raw": {"value": bad}}
            result = validate_judge(failure, rubric())
            self.assertEqual(result["error"], "http_failed")
            self.assertEqual(result["backend"], "jev")
            self.assertIsInstance(result["raw"], str)
            json.dumps(result, allow_nan=False)

    def test_cycles_and_unprintable_values_do_not_drop_failure_rows(self):
        class Unprintable:
            def __repr__(self):
                raise ValueError("cannot format")

        value = llm()
        value["extra"] = value
        self.assert_invalid(value)
        failure = validate_judge(Unprintable(), rubric())
        self.assertEqual(failure["error"], "schema_failed")
        self.assertIn("unserializable", failure["raw"])
        json.dumps(failure, allow_nan=False)

    def test_json_safe_backend_extras_are_retained(self):
        value = jev()
        value["details"]["quality"].update(
            raw_score=3.25, probabilities={"3": 0.75, "4": 0.25},
            legend={"3": "Good", "4": "Excellent"},
        )
        result = validate_judge(value, rubric())
        self.assertEqual(result["details"], value["details"])
        json.dumps(result, allow_nan=False)

    def test_invalid_rubric_raises_instead_of_blame_on_judge(self):
        with self.assertRaises(ValueError):
            validate_judge({"error": "parse_failed"}, {})
        with self.assertRaises(ValueError):
            compute_scores([], {})


class ComputeScoresTests(unittest.TestCase):
    def test_unsafe_metadata_excludes_an_otherwise_valid_judge(self):
        bad = llm()
        bad["provider_metadata"] = {"extra": math.nan}
        summary = compute_scores([{"judge": llm()}, {"judge": bad}], rubric())
        self.assertEqual(summary["n_scored"], 1)
        self.assertEqual(summary["n_total"], 2)
        self.assertEqual(summary["error_counts"], {"schema_failed": 1})
        json.dumps(summary, allow_nan=False)

    def test_mixed_backends_use_raw_means_and_score_over_scale(self):
        rows = [{"judge": llm()}, {"judge": jev()}]
        before = deepcopy(rows)
        result = compute_scores(rows, rubric())
        self.assertEqual(result, {
            "per_dimension_avg": {"quality": 4.12, "format": 2.5},
            "weighted_overall": 0.827,
            "n_scored": 2, "n_total": 2, "error_counts": {},
        })
        self.assertEqual(rows, before)

    def test_invalid_and_partial_judges_never_contribute_any_dimension(self):
        partial = llm()
        del partial["scores"]["format"]
        bad_details = llm()
        bad_details["details"]["format"]["confidence"] = 0.5
        extra = llm()
        extra["scores"]["extra"] = 5
        rows = [
            {"judge": llm()}, {"judge": partial}, {"judge": bad_details},
            {"judge": extra}, {"judge": {"error": "timeout", **llm()}},
            {"judge": {"error": "parse_failed"}}, {"judge": None}, {},
            {"judge": []}, {"judge": False}, {"judge": {"error": []}}, None,
        ]
        result = compute_scores(rows, rubric())
        self.assertEqual(result["n_total"], 12)
        self.assertEqual(result["n_scored"], 1)
        self.assertEqual(result["weighted_overall"], 0.85)
        self.assertEqual(result["per_dimension_avg"], {"quality": 4.0, "format": 3.0})
        self.assertEqual(result["error_counts"], {
            "schema_failed": 7, "timeout": 1, "parse_failed": 1, "judge_missing": 2,
        })

    def test_empty_and_unscored_results_have_null_scores(self):
        for rows in ([], [{"judge": None}], [{"judge": {"scores": {"quality": 5}}}]):
            result = compute_scores(rows, rubric())
            self.assertEqual(result["n_total"], len(rows))
            self.assertEqual(result["n_scored"], 0)
            self.assertIsNone(result["weighted_overall"])
            self.assertEqual(result["per_dimension_avg"], {"quality": None, "format": None})

    def test_quality_statistic_does_not_override_failed_external_gate(self):
        value = llm()
        value["scores"] = {"quality": 5, "format": 3}
        row = {
            "gates": {"required_checks": False}, "passed": False,
            "agent_result": {
                "recommendation": {"answer": "done"}, "latency_ms": 100,
                "tool_calls": 0, "input_tokens": None, "output_tokens": None,
                "model_id": "agent-model", "error": None,
            },
            "judge": value,
        }
        before = deepcopy(row)
        summary = compute_scores([row], rubric())
        self.assertEqual(summary["weighted_overall"], 1.0)
        self.assertEqual(row, before)
        self.assertFalse(row["passed"])
        self.assertNotIn("passed", summary)
        self.assertNotIn("pass_rate", summary)
        # The harness combines an actual external gate with any rubric threshold.
        passed = all(row["gates"].values()) and summary["weighted_overall"] >= rubric()["pass_threshold"]
        self.assertFalse(passed)


if __name__ == "__main__":
    unittest.main()
