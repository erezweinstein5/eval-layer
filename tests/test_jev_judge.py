"""Wire-contract tests use injected transport: no credentials or paid API calls."""
import copy
import json
import math
import unittest
from unittest.mock import Mock, patch
from urllib.error import URLError

from scripts.jev_judge import build_request, evaluate


def rubric():
    return {"dimensions": [
        {"name": "correctness", "scale": 3, "weight": 0.4,
         "levels": {1: "Wrong", 2: "Partly correct", 3: "Correct"}},
        {"name": "completeness", "scale": 5, "weight": 0.6,
         "levels": {str(i): f"Completeness level {i}" for i in range(1, 6)}},
    ], "pass_threshold": 0.7}


def response(payload):
    answers = {}
    for name, question in payload["questions"].items():
        levels = question["criteria"]
        p = {str(i): 0.0 for i in range(len(levels))}
        p[str(len(levels) - 2)] = 0.25
        p[str(len(levels) - 1)] = 0.75
        answers[name] = {"type": "score", "score": len(levels) - 1.25,
                         "legend": dict(enumerate(levels)),
                         "probabilities": p, "confidence": 0.4}
        answers[name]["legend"] = {str(i): v for i, v in enumerate(levels)}
    return {"model": "jev-test-pinned", "answers": answers,
            "usage": {"input_tokens": 80, "output_tokens": 12}}


class JevTests(unittest.TestCase):
    def invoke(self, raw=None, **kwargs):
        def transport(payload, key, timeout):
            self.payload = payload
            self.assertEqual(key, "test-key-only")
            return 200, {}, json.dumps(response(payload) if raw is None else raw).encode()
        return evaluate({"input": "Task", "agent_output": "Answer"}, rubric(),
                        model_id="jev-test", api_key="test-key-only",
                        transport=transport, **kwargs)

    def test_wire_request_fractional_mapping_and_metadata(self):
        result = self.invoke()
        self.assertNotIn("error", result)
        self.assertEqual(set(self.payload), {"state", "questions", "model"})
        self.assertEqual(self.payload["questions"]["correctness"]["criteria"],
                         ["Wrong", "Partly correct", "Correct"])
        self.assertIn("correctness", self.payload["questions"]["correctness"]["instructions"])
        self.assertEqual(result["scores"], {"correctness": 2.75, "completeness": 4.75})
        self.assertEqual(result["resolved_model_id"], "jev-test-pinned")
        self.assertEqual(result["model_id"], "jev-test")
        self.assertEqual(result["usage"]["input_tokens"], 80)
        self.assertEqual(result["review_dimensions"], ["correctness", "completeness"])
        detail = result["details"]["correctness"]
        self.assertEqual(detail["raw_score"], 1.75)
        for field in ("reasoning", "evidence", "suggestion"):
            self.assertIsNone(detail[field])
        self.assertEqual(result["attempts"], 1)
        self.assertGreaterEqual(result["latency_ms"], 0)

    def test_boundary_scores_and_zero_confidence(self):
        payload = build_request("state", rubric(), "jev-test")
        for endpoint in (0, -1):
            raw = response(payload)
            for answer in raw["answers"].values():
                keys = list(answer["legend"])
                key = keys[endpoint]
                answer.update(score=int(key), confidence=0,
                              probabilities={k: int(k == key) for k in keys})
            result = self.invoke(raw, review_threshold=0)
            self.assertNotIn("error", result)
            self.assertEqual(result["review_dimensions"], [])
            self.assertEqual(result["scores"]["correctness"], 1 if endpoint == 0 else 3)

    def test_live_rounded_score_and_probabilities_are_preserved(self):
        raw = response(build_request("state", rubric(), "jev-test"))
        answer = raw["answers"]["correctness"]
        # Observed from Jev 1.13: the visible probabilities imply 1.67 but the
        # separately rounded score is 1.68. Do not reject or replace the score.
        answer.update(score=1.68, confidence=0.52,
                      probabilities={"0": 0.11, "1": 0.11, "2": 0.78})
        result = self.invoke(raw)
        self.assertNotIn("error", result)
        self.assertAlmostEqual(result["scores"]["correctness"], 2.68)
        self.assertEqual(result["details"]["correctness"]["probabilities"],
                         answer["probabilities"])
        # Rounded probabilities need not sum to exactly one.
        answer.update(score=1.0, probabilities={"0": 0.33, "1": 0.33, "2": 0.33})
        self.assertNotIn("error", self.invoke(raw))
        answer["score"] = 1.2
        self.assertEqual(self.invoke(raw)["error"], "schema_failed")

    def test_schema_failures_never_expose_partial_scores(self):
        good = response(build_request("state", rubric(), "jev-test"))
        bad = []
        for value in ([], None, True, "x", 7):
            bad.append(value)
        for field in ("answers", "model", "usage"):
            row = copy.deepcopy(good); del row[field]; bad.append(row)
        for field, value in (("score", True), ("score", -1), ("score", 3),
                             ("score", 10 ** 400),
                             ("score", 1.1), ("confidence", 2), ("confidence", None),
                             ("type", "choice"), ("legend", {}),
                             ("probabilities", {"0": .2, "1": .2, "2": .2})):
            row = copy.deepcopy(good)
            row["answers"]["correctness"][field] = value; bad.append(row)
        row = copy.deepcopy(good); del row["answers"]["completeness"]; bad.append(row)
        row = copy.deepcopy(good); row["answers"]["extra"] = {}; bad.append(row)
        row = copy.deepcopy(good); row["usage"]["input_tokens"] = True; bad.append(row)
        for raw in bad:
            with self.subTest(raw=raw):
                # None is a real null here, not invoke's default sentinel.
                result = evaluate("state", rubric(), model_id="jev-test", api_key="test",
                                  transport=lambda *args: (200, {}, json.dumps(raw).encode()))
                self.assertEqual(result["error"], "schema_failed")
                self.assertNotIn("scores", result)
                self.assertIn("raw_response", result)

    def test_invalid_json_is_failure_and_retained(self):
        for body in (b'not JSON', b'{"score": NaN}', b'\xff'):
            result = evaluate("state", rubric(), model_id="jev-test", api_key="secret",
                              transport=lambda *args: (200, {}, body))
            self.assertEqual(result["error"], "schema_failed")
            self.assertIsInstance(result["raw_response"], str)

    def test_deep_non_object_response_is_tagged_failure(self):
        # Decoder recursion limits vary across supported Python versions.
        result = evaluate("state", rubric(), model_id="jev-test", api_key="secret",
                          transport=lambda *args: (200, {}, b'[' * 1100 + b']' * 1100))
        self.assertEqual(result["error"], "schema_failed")
        self.assertNotIn("scores", result)

    def test_exponent_overflow_in_extra_field_keeps_failure_serializable(self):
        good = response(build_request("state", rubric(), "jev-test"))
        body = json.dumps(good)[:-1] + ', "extra": 1e999}'
        result = evaluate("state", rubric(), model_id="jev-test", api_key="secret",
                          transport=lambda *args: (200, {}, body.encode()))
        self.assertEqual(result["error"], "schema_failed")
        self.assertIsInstance(result["raw_response"], str)
        json.dumps(result, allow_nan=False)

    def test_invalid_model_and_key_are_safe_configuration_failures(self):
        transport = Mock()
        result = evaluate("state", rubric(), model_id=math.nan, api_key="key",
                          transport=transport)
        self.assertEqual(result["error"], "configuration_error")
        json.dumps(result, allow_nan=False)
        result = evaluate("state", rubric(), model_id="jev-test", api_key="secret\nvalue",
                          transport=transport)
        self.assertEqual(result["error"], "configuration_error")
        self.assertNotIn("secret", json.dumps(result))
        transport.assert_not_called()

    def test_retry_transient_errors_and_report_last_usage_only(self):
        good = response(build_request("state", rubric(), "jev-test"))
        transport = Mock(side_effect=[(429, {"Retry-After": "2"}, b'limited'),
                                     (529, {}, b'busy'),
                                     (200, {}, json.dumps(good).encode())])
        sleep = Mock()
        result = evaluate("state", rubric(), model_id="jev-test", api_key="key",
                          transport=transport, sleep=sleep)
        self.assertNotIn("error", result)
        self.assertEqual(result["attempts"], 3)
        self.assertEqual(result["usage"], good["usage"])
        self.assertEqual([c.args[0] for c in sleep.call_args_list], [2.0, 1.0])

    def test_nonretryable_statuses(self):
        for status in (301, 401, 403, 422):
            transport = Mock(return_value=(status, {}, b'bad'))
            result = evaluate("state", rubric(), model_id="jev-test", api_key="key",
                              transport=transport)
            self.assertEqual(result["http_status"], status)
            self.assertEqual(transport.call_count, 1)

    def test_retry_exhaustion_and_long_retry_after(self):
        for headers, attempts in (({}, 3), ({"retry-after": "120"}, 1)):
            transport = Mock(return_value=(503, headers, b'busy'))
            result = evaluate("state", rubric(), model_id="jev-test", api_key="key",
                              transport=transport, sleep=Mock())
            self.assertEqual(result["error"], "http_error")
            self.assertEqual(transport.call_count, attempts)

    def test_transport_timeout_and_no_credential_leak(self):
        transport = Mock(side_effect=URLError("test-secret-key"))
        result = evaluate("state", rubric(), model_id="jev-test", api_key="test-secret-key",
                          transport=transport, sleep=Mock())
        self.assertEqual(result["error"], "transport_error")
        self.assertEqual(transport.call_count, 3)
        self.assertNotIn("test-secret-key", json.dumps(result))
        result = evaluate("state", rubric(), model_id="jev-test", api_key="test-secret-key",
                          transport=lambda *args: (401, {}, b'test-secret-key'))
        self.assertNotIn("test-secret-key", json.dumps(result))

    def test_configuration_and_missing_key_do_not_call_transport(self):
        transport = Mock()
        with patch.dict("os.environ", {}, clear=True):
            result = evaluate("state", rubric(), model_id="jev-test", transport=transport)
        self.assertEqual(result["error"], "configuration_error")
        self.assertEqual(result["attempts"], 0)
        for kwargs in ({"review_threshold": math.nan}, {"timeout_s": 0},
                       {"max_attempts": True}, {"max_attempts": 6}):
            result = evaluate("state", rubric(), model_id="jev-test", api_key="key",
                              transport=transport, **kwargs)
            self.assertEqual(result["error"], "configuration_error")
        transport.assert_not_called()

    def test_ambiguous_or_incomplete_levels_rejected_before_request(self):
        for levels in ({1: "one", "2": "two", 3: "three"}, {1: "one"},
                       {1: "one", 2: "", 3: "three"}, {True: "one", 2: "two", 3: "three"}):
            r = rubric(); r["dimensions"][0]["levels"] = levels
            with self.assertRaises(ValueError):
                build_request("state", r, "jev-test")

    def test_http_request_uses_official_endpoint_and_bearer_header(self):
        from scripts import jev_judge
        payload = build_request("state", rubric(), "jev-test")
        response_mock = Mock(status=200, headers={})
        response_mock.read.return_value = json.dumps(response(payload)).encode()
        context = Mock()
        context.__enter__ = Mock(return_value=response_mock)
        context.__exit__ = Mock(return_value=False)
        with patch.object(jev_judge, "build_opener") as build:
            build.return_value.open.return_value = context
            result = evaluate("state", rubric(), model_id="jev-test", api_key="key")
        self.assertNotIn("error", result)
        request = build.return_value.open.call_args.args[0]
        self.assertEqual(request.full_url, "https://api.typesafe.ai/v1/systemone")
        self.assertEqual(request.get_header("Authorization"), "Bearer key")
        self.assertEqual(json.loads(request.data), payload)
        self.assertEqual(build.return_value.open.call_args.kwargs["timeout"], 30)


if __name__ == "__main__":
    unittest.main()
