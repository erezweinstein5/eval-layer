"""Offline adapter tests: fake credentials and injected/mocked HTTP only."""

import copy
import io
import json
import math
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError, URLError

from scripts import llm_judge
from scripts.judge_results import validate_judge
from scripts.llm_judge import build_messages, complete, estimate_cost, evaluate


def rubric():
    return {"dimensions": [
        {"name": "accuracy", "scale": 3, "weight": 0.4,
         "levels": {1: "Wrong", 2: "Partially right", 3: "Right"}},
        {"name": "coverage", "scale": 5, "weight": 0.6,
         "levels": {str(i): f"Coverage level {i}" for i in range(1, 6)}},
    ], "pass_threshold": 0.7}


def grade():
    return {
        "scores": {"accuracy": 2.75, "coverage": 4},
        "details": {name: {
            "reasoning": "Matches the descriptor.",
            "evidence": ["Recorded observation."],
            "suggestion": "Add supporting evidence.", "confidence": "medium",
        } for name in ("accuracy", "coverage")},
        "overall_reasoning": "Satisfies most requirements.",
    }


def response(text="Answer"):
    return {
        "model": "resolved-test-model",
        "choices": [{"message": {"role": "assistant", "content": text},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 40,
                  "prompt_tokens_details": {"cached_tokens": 25},
                  "completion_tokens_details": {"reasoning_tokens": 30}},
    }


def wire(raw):
    return Mock(return_value=(200, {}, json.dumps(raw).encode()))


def tools():
    return [{"type": "function", "function": {
        "name": "lookup_policy", "description": "Read a policy entry.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}},
                       "required": ["query"], "additionalProperties": False},
    }}]


def tool_response(content=None):
    raw = response(content)
    raw["choices"][0]["finish_reason"] = "tool_calls"
    raw["choices"][0]["message"]["tool_calls"] = [{
        "id": "call_1", "type": "function",
        "function": {"name": "lookup_policy", "arguments": '{ "query": "refund" }'},
    }]
    return raw


class OfflineTest(unittest.TestCase):
    def setUp(self):
        # Never consult this machine's real API key or provider environment.
        self.env = patch.dict("os.environ", {}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)
        self.network = patch.object(llm_judge, "build_opener",
                                    side_effect=AssertionError("unexpected network"))
        self.network.start()
        self.addCleanup(self.network.stop)

    def invoke(self, raw=None, **kwargs):
        kwargs.setdefault("model_id", "requested-test-model")
        kwargs.setdefault("api_key", "fake-secret")
        kwargs.setdefault("transport", wire(response() if raw is None else raw))
        return complete([{"role": "user", "content": "Question"}], **kwargs)


class CompletionTests(OfflineTest):
    def test_request_body_defaults_and_metrics(self):
        transport = wire(response())
        with patch.object(llm_judge.time, "perf_counter", side_effect=[1, 1.125]):
            result = self.invoke(transport=transport)
        self.assertNotIn("error", result)
        url, payload, key, timeout = transport.call_args.args
        self.assertEqual(url, "https://api.openai.com/v1/chat/completions")
        self.assertEqual(payload, {
            "model": "requested-test-model",
            "messages": [{"role": "user", "content": "Question"}],
            "max_tokens": 4096, "temperature": 0,
        })
        self.assertEqual((key, timeout), ("fake-secret", 120))
        self.assertEqual(result["text"], "Answer")
        self.assertEqual(result["model_id"], "requested-test-model")
        self.assertEqual(result["resolved_model_id"], "resolved-test-model")
        self.assertEqual(result["latency_ms"], 125)
        self.assertEqual(result["usage"]["input_tokens"], 100)
        self.assertEqual(result["usage"]["output_tokens"], 40)
        self.assertEqual(result["usage"]["cached_input_tokens"], 25)
        self.assertEqual(result["usage"]["completion_tokens_details"], {"reasoning_tokens": 30})
        self.assertEqual(result["raw_response"], response())
        self.assertEqual(result["tool_calls"], [])
        self.assertEqual(result["assistant_message"], {"role": "assistant", "content": "Answer"})
        transport.assert_called_once()

    def test_env_and_explicit_precedence_origin_or_versioned_base(self):
        for base in ("https://bedrock-mantle.us-east-2.api.aws",
                     "https://bedrock-mantle.us-east-2.api.aws/v1/"):
            with patch.dict("os.environ", OPENAI_BASE_URL=base, OPENAI_API_KEY="fake-env"):
                transport = wire(response())
                result = self.invoke(api_key=None, transport=transport)
                self.assertNotIn("error", result)
                self.assertEqual(transport.call_args.args[0],
                                 "https://bedrock-mantle.us-east-2.api.aws/v1/chat/completions")
                self.assertEqual(transport.call_args.args[2], "fake-env")
                self.invoke(base_url=llm_judge.DEFAULT_BASE_URL, api_key="explicit",
                            transport=transport)
                self.assertEqual(transport.call_args.args[0],
                                 "https://api.openai.com/v1/chat/completions")
                self.assertEqual(transport.call_args.args[2], "explicit")

    def test_json_mode_and_explicit_sampling(self):
        transport = wire(response())
        self.invoke(transport=transport, json_mode=True, temperature=0.5,
                    max_tokens=512, timeout_s=10)
        payload = transport.call_args.args[1]
        self.assertEqual(payload["response_format"], {"type": "json_object"})
        self.assertEqual(payload["max_tokens"], 512)
        self.assertEqual(payload["temperature"], 0.5)
        self.assertEqual(transport.call_args.args[3], 10)

    def test_usage_holes_remain_unknown_and_not_inferred_from_total(self):
        for usage in (None, {}, {"total_tokens": 10},
                      {"prompt_tokens": None, "completion_tokens": None}):
            raw = response()
            raw["usage"] = usage
            result = self.invoke(raw)
            self.assertNotIn("error", result)
            self.assertEqual(result["usage"], {"input_tokens": None, "output_tokens": None})
        raw = response()
        del raw["usage"]
        del raw["model"]
        result = self.invoke(raw)
        self.assertIsNone(result["resolved_model_id"])
        self.assertIsNone(result["usage"]["input_tokens"])
        raw["usage"] = {"prompt_tokens": 0}
        result = self.invoke(raw)
        self.assertEqual(result["usage"], {"input_tokens": 0, "output_tokens": None})

    def test_invalid_usage_is_schema_failure(self):
        for field in ("prompt_tokens", "completion_tokens"):
            for bad in (True, -1, 1.5, "1", math.nan, math.inf, 10 ** 400):
                raw = response()
                raw["usage"][field] = bad
                with self.subTest(field=field, bad=bad):
                    result = self.invoke(raw)
                    self.assertEqual(result["error"], "schema_failed")
                    json.dumps(result, allow_nan=False)
        for usage in ([], "bad", {"prompt_tokens_details": []},
                      {"completion_tokens_details": {"reasoning_tokens": True}},
                      {"prompt_tokens": 10, "prompt_tokens_details": {"cached_tokens": 11}}):
            raw = response()
            raw["usage"] = usage
            self.assertEqual(self.invoke(raw)["error"], "schema_failed")

    def test_http_errors_and_redirects_are_not_retried(self):
        for status in (301, 302, 307, 308, 400, 401, 429, 500, 503):
            transport = Mock(return_value=(status, {"Retry-After": "1",
                                                   "Location": "https://example.org"},
                                           b'{"error":"fake-secret"}'))
            result = self.invoke(transport=transport)
            self.assertEqual(result["error"], "http_error")
            self.assertEqual(result["http_status"], status)
            self.assertNotIn("fake-secret", json.dumps(result))
            transport.assert_called_once()

    def test_transport_exceptions_never_leak_credentials_or_retry(self):
        for exc in (TimeoutError("fake-secret"), URLError("fake-secret"),
                    OSError("fake-secret"), RuntimeError("fake-secret")):
            transport = Mock(side_effect=exc)
            result = self.invoke(transport=transport)
            self.assertEqual(result["error"], "transport_error")
            self.assertNotIn("fake-secret", json.dumps(result))
            self.assertIsNone(result["usage"]["output_tokens"])
            transport.assert_called_once()

    def test_injected_http_exception_preserves_status(self):
        error = HTTPError("https://example.org", 401, "fake-secret", {},
                          io.BytesIO(b'{"error":"fake-secret"}'))
        result = self.invoke(transport=Mock(side_effect=error))
        self.assertEqual(result["error"], "http_error")
        self.assertEqual(result["http_status"], 401)
        self.assertNotIn("fake-secret", json.dumps(result))
        class BrokenBody(io.BytesIO):
            def read(self, *args):
                raise OSError("fake-secret")

        error = HTTPError("https://example.org", 401, "fake-secret", {}, BrokenBody())
        result = self.invoke(transport=Mock(side_effect=error))
        self.assertEqual(result["error"], "transport_error")
        self.assertNotIn("fake-secret", json.dumps(result))

    def test_bad_json_nonobjects_and_exponent_overflow_are_safe(self):
        for body in (b"not JSON", b"\xff", b'{"extra":"\xff"}', b"[]", b"null", b"true",
                     b'{"extra":NaN}', b'{"extra":Infinity}', b'{"extra":1e999}',
                     b"[" * 1100 + b"]" * 1100):
            with self.subTest(body=body[:30]):
                result = self.invoke(transport=Mock(return_value=(200, {}, body)))
                self.assertEqual(result["error"], "schema_failed")
                json.dumps(result, allow_nan=False)
        body = json.dumps(response())[:-1] + ', "extra":1e999}'
        result = self.invoke(transport=Mock(return_value=(200, {}, body)))
        self.assertEqual(result["error"], "schema_failed")
        self.assertIsInstance(result["raw_response"], str)
        json.dumps(result, allow_nan=False)

    def test_schema_failures_keep_available_billed_usage(self):
        for choices in (None, [], [{}], [{"message": {"content": None}}],
                        [{"message": {"content": []}}],
                        [{"message": {"content": "Partial"}, "finish_reason": "length"}],
                        [{"message": {"content": ""}, "finish_reason": "content_filter"}]):
            raw = response()
            raw["choices"] = choices
            result = self.invoke(raw)
            self.assertEqual(result["error"], "schema_failed")
            self.assertEqual(result["usage"]["output_tokens"], 40)

    def test_escaped_echoed_key_is_redacted_after_json_decoding(self):
        body = json.dumps(response("fake-secret")).replace("fake-secret", r"fake-\u0073ecret")
        result = self.invoke(transport=Mock(return_value=(200, {}, body)))
        self.assertEqual(result["text"], "[REDACTED]")
        self.assertNotIn("fake-secret", json.dumps(result))

    def test_unsafe_endpoints_rejected_before_transport(self):
        for base in ("http://api.openai.com/v1", "https://evil.example/v1",
                     "https://api.openai.com.evil.example/v1",
                     "https://api.openai.com@evil.example/v1",
                     "https://fake-secret@api.openai.com/v1",
                     "https://api.openai.com/v1?key=fake-secret",
                     "https://api.openai.com/v1#fake-secret",
                     "https://api.openai.com:123/v1", "https://api.openai.com:bad",
                     "https://api.openai.com/\nv1", "https://api.openai.com/other"):
            transport = Mock()
            result = self.invoke(base_url=base, transport=transport)
            self.assertEqual(result["error"], "configuration_error", base)
            self.assertNotIn("fake-secret", json.dumps(result))
            transport.assert_not_called()

    def test_localhost_requires_explicit_url_and_key(self):
        for base in ("http://localhost:8080", "http://127.0.0.1:8080/v1",
                     "http://[::1]:8080/v1"):
            self.assertNotIn("error", self.invoke(base_url=base))
        with patch.dict("os.environ", OPENAI_BASE_URL="http://localhost:8080",
                        OPENAI_API_KEY="fake-env"):
            transport = Mock()
            self.assertEqual(self.invoke(transport=transport)["error"], "configuration_error")
            self.assertEqual(self.invoke(base_url="http://localhost:8080", api_key=None,
                                         transport=transport)["error"], "configuration_error")
            transport.assert_not_called()

    def test_bad_config_never_calls_transport_and_is_json_safe(self):
        for kwargs in ({"api_key": None}, {"api_key": ""}, {"api_key": "fake-secret\n"},
                       {"api_key": "secret\x7f"}, {"model_id": math.nan},
                       {"model_id": ""}, {"max_tokens": True}, {"max_tokens": 0},
                       {"temperature": True}, {"temperature": math.inf},
                       {"timeout_s": 0}, {"timeout_s": math.nan}, {"json_mode": 1}):
            transport = Mock()
            result = self.invoke(transport=transport, **kwargs)
            self.assertEqual(result["error"], "configuration_error")
            json.dumps(result, allow_nan=False)
            transport.assert_not_called()
        for content in (math.inf, object()):
            transport = Mock()
            result = complete([{"role": "user", "content": content}], model_id="test",
                              api_key="fake", transport=transport)
            self.assertEqual(result["error"], "configuration_error")
            transport.assert_not_called()

    def test_real_http_builder_is_mocked_and_redirects_disabled(self):
        context = Mock()
        context.__enter__ = Mock(return_value=Mock(status=200, headers={},
                                                  read=Mock(return_value=json.dumps(response()).encode())))
        context.__exit__ = Mock(return_value=False)
        with patch.object(llm_judge, "build_opener") as build:
            build.return_value.open.return_value = context
            result = self.invoke(transport=None)
        self.assertNotIn("error", result)
        request = build.return_value.open.call_args.args[0]
        self.assertEqual(request.get_header("Authorization"), "Bearer fake-secret")
        self.assertEqual(request.method, "POST")
        self.assertEqual(request.full_url, "https://api.openai.com/v1/chat/completions")
        handler = build.call_args.args[0]
        self.assertIsNone(handler.redirect_request(request, None, 302, "", {},
                                                  "https://example.org"))
        self.assertEqual(build.return_value.open.call_args.kwargs["timeout"], 120)
        with patch.object(llm_judge, "build_opener") as build:
            build.return_value.open.side_effect = HTTPError(
                request.full_url, 307, "redirect", {"Location": "https://example.org"},
                io.BytesIO(b"redirect"))
            result = self.invoke(transport=None)
        self.assertEqual(result["http_status"], 307)
        build.return_value.open.assert_called_once()


class ToolCallTests(OfflineTest):
    def test_optional_tools_and_choice_are_only_sent_when_supplied(self):
        transport = wire(response())
        self.invoke(transport=transport)
        self.assertNotIn("tools", transport.call_args.args[1])
        self.assertNotIn("tool_choice", transport.call_args.args[1])
        definitions = tools()
        before = copy.deepcopy(definitions)
        self.invoke(tools=definitions, transport=transport)
        self.assertEqual(transport.call_args.args[1]["tools"], definitions)
        self.assertEqual(transport.call_args.args[1]["tool_choice"], "auto")
        self.assertEqual(definitions, before)
        for choice in ("none", "required", {"type": "function",
                                           "function": {"name": "lookup_policy"}}):
            self.invoke(tools=definitions, tool_choice=choice, transport=transport)
            self.assertEqual(transport.call_args.args[1]["tool_choice"], choice)
        self.invoke(tools=definitions, tool_choice=None, transport=transport)
        self.assertNotIn("tool_choice", transport.call_args.args[1])

    def test_null_or_text_tool_turn_preserves_calls_and_billed_usage(self):
        for content in (None, "", "I will check the policy."):
            raw = tool_response(content)
            raw["choices"][0]["message"]["tool_calls"].append({
                "id": "call_2", "type": "function",
                "function": {"name": "lookup_policy", "arguments": '{"query":"cancel"}'},
            })
            result = self.invoke(raw, tools=tools())
            self.assertNotIn("error", result)
            self.assertEqual(result["text"], content)
            self.assertEqual(result["tool_calls"], raw["choices"][0]["message"]["tool_calls"])
            self.assertEqual(result["assistant_message"], raw["choices"][0]["message"])
            self.assertEqual(result["usage"]["output_tokens"], 40)
            self.assertEqual(result["usage"]["completion_tokens_details"]["reasoning_tokens"], 30)
            self.assertEqual(result["resolved_model_id"], raw["model"])
            json.dumps(result, allow_nan=False)

    def test_assistant_message_can_be_replayed_with_a_tool_reply(self):
        transport = Mock(side_effect=[
            (200, {}, json.dumps(tool_response()).encode()),
            (200, {}, json.dumps(response("Refund policy found.")).encode()),
        ])
        messages = [{"role": "user", "content": "What is the refund policy?"}]
        first = complete(messages, model_id="test", api_key="fake-secret",
                         tools=tools(), transport=transport)
        self.assertNotIn("error", first)
        call = first["tool_calls"][0]
        self.assertEqual(json.loads(call["function"]["arguments"]), {"query": "refund"})
        messages = [*messages, first["assistant_message"],
                    {"role": "tool", "tool_call_id": call["id"], "content": "14 days"}]
        second = complete(messages, model_id="test", api_key="fake-secret",
                          tools=tools(), transport=transport)
        self.assertNotIn("error", second)
        self.assertEqual(transport.call_args.args[1]["messages"], messages)
        self.assertEqual(second["text"], "Refund policy found.")
        self.assertEqual(second["tool_calls"], [])
        self.assertEqual(transport.call_count, 2)

    def test_malformed_call_identity_function_and_arguments_fail_as_a_whole(self):
        good = tool_response()
        original = good["choices"][0]["message"]["tool_calls"][0]
        bad_calls = [None, True, {}, {"id": "call_1", "type": "other"},
                     {**original, "function": None}]
        for call_id in (None, True, 7, "", " ", "call\n1", "call\x001"):
            bad_calls.append({**original, "id": call_id})
        for name in (None, True, "", "invalid name", "a" * 65):
            bad_calls.append({**original, "function": {**original["function"], "name": name}})
        for arguments in (None, {}, True, "", "not JSON", "[]", "null", "1",
                          '"string"', '{"x": NaN}', '{"x": Infinity}', '{"x": 1e999}',
                          "[" * 1100 + "]" * 1100):
            bad_calls.append({**original, "function": {
                **original["function"], "arguments": arguments,
            }})
        for bad in bad_calls:
            raw = copy.deepcopy(good)
            raw["choices"][0]["message"]["tool_calls"] = [original, bad]
            # Use another ID so malformed arguments/names aren't masked by the
            # duplicate-ID check on the valid first call.
            if isinstance(bad, dict) and bad.get("id") == "call_1":
                raw["choices"][0]["message"]["tool_calls"][1] = {**bad, "id": "call_2"}
            result = self.invoke(raw, tools=tools())
            self.assertEqual(result["error"], "schema_failed", bad)
            self.assertEqual(result["tool_calls"], [])
            self.assertIsNone(result["assistant_message"])
            self.assertEqual(result["usage"]["output_tokens"], 40)
            json.dumps(result, allow_nan=False)
        duplicate = copy.deepcopy(good)
        duplicate["choices"][0]["message"]["tool_calls"] = [original, original]
        self.assertEqual(self.invoke(duplicate, tools=tools())["error"], "schema_failed")

    def test_empty_ordinary_answers_and_inconsistent_tool_turns_fail(self):
        for content in (None, "", " \n\t"):
            result = self.invoke(response(content))
            self.assertEqual(result["error"], "schema_failed")
        for calls in (None, [], {}, "calls"):
            raw = tool_response()
            raw["choices"][0]["message"]["tool_calls"] = calls
            self.assertEqual(self.invoke(raw, tools=tools())["error"], "schema_failed")
        for reason in ("length", "content_filter", "stop", None):
            raw = tool_response()
            raw["choices"][0]["finish_reason"] = reason
            self.assertEqual(self.invoke(raw, tools=tools())["error"], "schema_failed")
        self.assertEqual(self.invoke(tool_response())["error"], "schema_failed")
        raw = tool_response()
        raw["choices"][0]["message"]["role"] = "user"
        self.assertEqual(self.invoke(raw, tools=tools())["error"], "schema_failed")

    def test_invalid_tool_configuration_does_not_call_transport(self):
        for options in ({"tools": {}}, {"tools": [None]}, {"tools": [{"type": "function"}]},
                        {"tools": tools() * 2}, {"tool_choice": "required"},
                        {"tools": tools(), "tool_choice": True},
                        {"tools": tools(), "tool_choice": "invalid"},
                        {"tools": tools(), "tool_choice": {"type": "function", "function": {
                            "name": "unavailable"}}}):
            transport = Mock()
            result = self.invoke(transport=transport, **options)
            self.assertEqual(result["error"], "configuration_error")
            transport.assert_not_called()
        definitions = tools()
        definitions[0]["function"]["parameters"]["extra"] = math.inf
        transport = Mock()
        self.assertEqual(self.invoke(tools=definitions, transport=transport)["error"],
                         "configuration_error")
        transport.assert_not_called()

    def test_argument_redaction_and_unknown_usage_are_preserved(self):
        raw = tool_response()
        raw["usage"] = None
        raw["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = (
            '{"query": "fake-\\u0073ecret"}')
        result = self.invoke(raw, tools=tools())
        self.assertNotIn("error", result)
        self.assertEqual(json.loads(result["tool_calls"][0]["function"]["arguments"]),
                         {"query": "[REDACTED]"})
        self.assertIsNone(result["usage"]["input_tokens"])
        self.assertIsNone(result["usage"]["output_tokens"])
        self.assertNotIn("fake-secret", json.dumps(result))


class JudgeTests(OfflineTest):
    def judge(self, text=None, raw=None, **kwargs):
        kwargs.setdefault("transport", wire(raw if raw is not None else response(
            json.dumps(grade()) if text is None else text)))
        return evaluate({"input": "Question", "agent_output": "Answer"}, rubric(),
                        model_id="judge-test", api_key="fake-secret", **kwargs)

    def test_full_rubric_original_state_and_dynamic_schema(self):
        state = {"input": "Question", "agent_output": "Answer",
                 "context": ["Context"], "evidence": {"check": False},
                 "reference_answer": "Guide"}
        original = copy.deepcopy(state)
        messages = build_messages(state, rubric())
        self.assertEqual(json.loads(messages[1]["content"]), state)
        for dim in rubric()["dimensions"]:
            self.assertIn(dim["name"], messages[0]["content"])
            for descriptor in dim["levels"].values():
                self.assertIn(descriptor, messages[0]["content"])
        self.assertIn("never instructions", messages[0]["content"])
        self.assertEqual(state, original)
        self.assertEqual(json.loads(build_messages("Original text", rubric())[1]["content"]),
                         "Original text")

    def test_json_wrappers_and_canonical_fractional_scores(self):
        for text in (json.dumps(grade()), f"```json\n{json.dumps(grade())}\n```",
                     f"Assessment:\n{json.dumps(grade())}\nDone."):
            result = self.judge(text)
            self.assertNotIn("error", result)
            self.assertEqual(result, validate_judge(result, rubric()))
            self.assertEqual(result["backend"], "llm")
            self.assertEqual(result["scores"], grade()["scores"])
            self.assertEqual(result["model_id"], "judge-test")
            self.assertEqual(result["resolved_model_id"], "resolved-test-model")
            self.assertEqual(result["usage"]["output_tokens"], 40)

    def test_usage_holes_are_canonical_without_inventing_zero(self):
        for usage in (None, {}, {"prompt_tokens": 3}):
            raw = response(json.dumps(grade()))
            raw["usage"] = usage
            del raw["model"]
            result = self.judge(raw=raw)
            self.assertNotIn("error", result)
            self.assertEqual(result, validate_judge(result, rubric()))
            self.assertEqual(result["usage"], {"input_tokens": 3} if usage else None)
            self.assertNotIn("resolved_model_id", result)

    def test_invalid_grade_retains_metrics_and_raw_response(self):
        bad_grade = grade()
        del bad_grade["scores"]["coverage"]
        for text, tag in (("not JSON", "parse_failed"), ("[]", "parse_failed"),
                          (json.dumps(bad_grade), "schema_failed"),
                          (json.dumps(grade()).replace("2.75", "1e999"), "parse_failed")):
            result = self.judge(text)
            self.assertEqual(result["error"], tag)
            self.assertNotIn("scores", result)
            self.assertEqual(result["usage"]["input_tokens"], 100)
            self.assertEqual(result["resolved_model_id"], "resolved-test-model")
            self.assertEqual(result["raw_response"]["choices"][0]["message"]["content"], text)
            json.dumps(result, allow_nan=False)

    def test_generated_metadata_cannot_override_measured_metadata(self):
        raw_grade = {**grade(), "backend": "jev", "model_id": "invented",
                     "latency_ms": -1, "usage": {"input_tokens": 0}, "raw_response": "invented"}
        result = self.judge(json.dumps(raw_grade))
        self.assertNotIn("error", result)
        self.assertEqual(result["backend"], "llm")
        self.assertEqual(result["model_id"], "judge-test")
        self.assertEqual(result["usage"]["input_tokens"], 100)
        self.assertIsInstance(result["raw_response"], dict)
        raw_grade["resolved_model_id"] = "invented"
        raw = response(json.dumps(raw_grade))
        del raw["model"]
        self.assertNotIn("resolved_model_id", self.judge(raw=raw))

    def test_escaped_key_inside_generated_grade_is_redacted(self):
        raw_grade = grade()
        raw_grade["details"]["accuracy"]["reasoning"] = "fake-secret"
        text = json.dumps(raw_grade).replace("fake-secret", r"fake-\u0073ecret")
        result = self.judge(text)
        self.assertNotIn("error", result)
        self.assertEqual(result["details"]["accuracy"]["reasoning"], "[REDACTED]")
        self.assertNotIn("fake-secret", json.dumps(result))

    def test_http_failure_is_a_canonical_tagged_judge(self):
        result = self.judge(transport=Mock(return_value=(429, {}, b"busy")))
        self.assertEqual(result["error"], "http_error")
        self.assertEqual(result["backend"], "llm")
        self.assertEqual(result, validate_judge(result, rubric()))
        self.assertIsNone(result["usage"])

    def test_invalid_levels_and_state_fail_before_request(self):
        for levels in (None, {}, {1: "one", 2: "two"},
                       {1: "one", "2": "two", 3: "three"},
                       {True: "one", 2: "two", 3: "three"},
                       {1: "one", 2: "", 3: "three"}):
            value = rubric()
            value["dimensions"][0]["levels"] = levels
            transport = Mock()
            result = evaluate("State", value, model_id="test", api_key="fake",
                              transport=transport)
            self.assertEqual(result["error"], "configuration_error")
            transport.assert_not_called()
        for state in (None, {"extra": math.inf}, {"extra": object()}):
            transport = Mock()
            result = evaluate(state, rubric(), model_id="test", api_key="fake",
                              transport=transport)
            self.assertEqual(result["error"], "configuration_error")
            json.dumps(result, allow_nan=False)
            transport.assert_not_called()


class CostTests(unittest.TestCase):
    def setUp(self):
        self.usage = {"input_tokens": 1_000_000, "output_tokens": 500_000,
                      "cached_input_tokens": 250_000,
                      "completion_tokens_details": {"reasoning_tokens": 400_000}}
        self.pricing = {"input_usd_per_million": 2, "output_usd_per_million": 8}

    def test_full_input_rate_without_cache_price_and_no_reasoning_double_count(self):
        self.assertEqual(estimate_cost(self.usage, self.pricing), 6)

    def test_cache_discount_only_for_reported_valid_cache(self):
        self.pricing["cached_input_usd_per_million"] = 0.5
        self.assertEqual(estimate_cost(self.usage, self.pricing), 5.625)
        self.usage["cached_input_tokens"] = 1_000_000
        self.assertEqual(estimate_cost(self.usage, self.pricing), 4.5)
        self.usage["cached_input_tokens"] = 0
        self.assertEqual(estimate_cost(self.usage, self.pricing), 6)
        self.usage["cached_input_tokens"] = None
        self.assertEqual(estimate_cost(self.usage, self.pricing), 6)
        del self.usage["cached_input_tokens"]
        self.assertEqual(estimate_cost(self.usage, self.pricing), 6)

    def test_unknown_usage_and_rates_never_become_zero(self):
        for usage in (None, {}, {"input_tokens": 0}, {"output_tokens": 0},
                      {"input_tokens": None, "output_tokens": 5}):
            self.assertIsNone(estimate_cost(usage, self.pricing))
        for pricing in (None, {}, {"input_usd_per_million": 2}):
            self.assertIsNone(estimate_cost(self.usage, pricing))
        self.assertEqual(estimate_cost({"input_tokens": 0, "output_tokens": 0}, self.pricing), 0)

    def test_invalid_counts_rates_and_overflow_rejected(self):
        for key in ("input_tokens", "output_tokens", "cached_input_tokens"):
            for invalid in (-1, True, False, math.nan, math.inf, "1", 1.5, 10 ** 400):
                usage = {**self.usage, key: invalid}
                self.assertIsNone(estimate_cost(usage, self.pricing), (key, invalid))
        self.assertIsNone(estimate_cost({**self.usage, "cached_input_tokens": 1_000_001},
                                       self.pricing))
        for key in ("input_usd_per_million", "output_usd_per_million",
                    "cached_input_usd_per_million"):
            for invalid in (-1, True, False, math.nan, math.inf, "1", None, 10 ** 400):
                self.assertIsNone(estimate_cost(self.usage, {**self.pricing, key: invalid}))
        self.assertIsNone(estimate_cost({"input_tokens": 10 ** 308, "output_tokens": 1},
                                       {"input_usd_per_million": 1e308,
                                        "output_usd_per_million": 1}))

    def test_zero_prices_and_inputs_are_not_mutated(self):
        before = copy.deepcopy((self.usage, self.pricing))
        estimate_cost(self.usage, self.pricing)
        self.assertEqual((self.usage, self.pricing), before)
        self.assertEqual(estimate_cost(self.usage, {
            "input_usd_per_million": 0, "output_usd_per_million": 0,
            "cached_input_usd_per_million": 0}), 0)


if __name__ == "__main__":
    unittest.main()
