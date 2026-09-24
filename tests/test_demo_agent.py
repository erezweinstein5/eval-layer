"""Offline native-tool integration tests; no provider or account calls."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from demo import agent
from demo.tools import AS_OF, FIXTURE, ToolEnvironment, tool_schemas


METADATA_FIELDS = {"recommendation", "latency_ms", "tool_calls", "input_tokens",
                   "output_tokens", "model_id", "error"}
KNOWLEDGE = """# Offline Harbor policy
## Plans and billing
Team is $20 per member per month. Monthly seat additions are prorated.
Annual seat changes require the account team.
## Cancellation and refunds
Initial purchases allow refund requests within 14 days if no project was exported.
REFUND_POLICY_FROM_CONFIG. Renewals are not automatically refundable.
Chat support cannot issue a refund.
## Security
Never request passwords or API keys.
"""


def tool_call(name, arguments, call_id="call-1"):
    return {"id": call_id, "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments, allow_nan=False)}}


def reply(text=None, calls=None, *, usage=None, error=None):
    calls = [] if calls is None else calls
    message = {"role": "assistant", "content": text}
    if calls:
        message["tool_calls"] = deepcopy(calls)
    result = {
        "text": text, "assistant_message": message, "tool_calls": deepcopy(calls),
        "model_id": "offline-model", "resolved_model_id": "offline-model-pinned",
        "latency_ms": 12, "raw_response": {"offline_mock": True},
        "usage": usage if usage is not None else {
            "input_tokens": 10, "output_tokens": 3, "cached_input_tokens": 4,
            "completion_tokens_details": {"reasoning_tokens": 1}},
    }
    if error:
        result["error"] = error
    return result


def last_tool(messages):
    return json.loads(next(message["content"] for message in reversed(messages)
                           if message["role"] == "tool"))


class DemoAgentTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(self.enterContext(tempfile.TemporaryDirectory(prefix="harbor-offline-")))
        self.config = {
            "model": "offline-model", "system_prompt": agent.DEFAULT_INSTRUCTIONS,
            "knowledge": KNOWLEDGE, "environment": deepcopy(FIXTURE),
            "_storage_dir": str(self.directory / "case"),
            "_case": {"workspace_id": "ws-acme", "follow_up_inputs": [],
                      "expected_output": "PRIVATE_EXPECTED_OUTPUT",
                      "contract": {"hidden": "PRIVATE_CONTRACT"}},
        }
        self.network = self.enterContext(patch(
            "scripts.llm_judge.build_opener", side_effect=AssertionError("network forbidden")))
        self.complete = self.enterContext(patch.object(
            agent, "complete", side_effect=AssertionError("a completion fixture is required")))

    def environment(self, **kwargs):
        return ToolEnvironment(self.config, self.config["_storage_dir"], **kwargs)

    def test_multistep_tool_rounds_have_native_messages_and_aggregate_usage(self):
        self.complete.side_effect = [
            reply(calls=[tool_call("search_memory", {"query": "contact preference"}, "mem"),
                         tool_call("get_workspace", {}, "workspace")]),
            reply(calls=[tool_call("get_invoice", {"invoice_id": "inv-100"}, "invoice-100"),
                         tool_call("get_invoice", {"invoice_id": "inv-101"}, "invoice-101")]),
            reply(calls=[tool_call("create_ticket", {
                "category": "billing", "summary": "Duplicate inv-100; preferred email updates.",
                "invoice_id": "inv-100"}, "ticket")]),
            reply("The existing local ticket is ticket-previous and is pending. No refund was issued."),
        ]
        before = deepcopy(self.config)
        metadata, response = agent.run("Investigate the two charges and open a local ticket.", self.config)
        self.assertEqual(self.config, before)
        self.assertEqual(set(metadata), METADATA_FIELDS)
        self.assertIsNone(metadata["error"])
        self.assertEqual(metadata["tool_calls"], 5)
        self.assertEqual(metadata["input_tokens"], 40)
        self.assertEqual(metadata["output_tokens"], 12)
        self.assertEqual(response["usage"]["cached_input_tokens"], 16)
        self.assertEqual(response["usage"]["completion_tokens_details"]["reasoning_tokens"], 4)
        self.assertEqual(response["model_latency_ms"], 48)
        self.assertGreaterEqual(metadata["latency_ms"], 48)
        self.assertEqual(metadata["latency_ms"], response["latency_ms"])
        second_messages = self.complete.call_args_list[1].args[0]
        self.assertEqual([m["role"] for m in second_messages],
                         ["system", "user", "assistant", "tool", "tool"])
        self.assertEqual(second_messages[2]["tool_calls"][0]["id"], "mem")
        self.assertEqual([m["tool_call_id"] for m in second_messages[-2:]], ["mem", "workspace"])
        self.assertEqual(last_tool(second_messages)["workspace"]["billing_cycle"], "monthly")
        calls = [event for event in response["trace"] if event["type"] == "tool_call"]
        self.assertEqual([event["tool_call_id"] for event in calls],
                         ["mem", "workspace", "invoice-100", "invoice-101", "ticket"])
        self.assertFalse(calls[-1]["result"]["created"])
        self.assertEqual(calls[-1]["result"]["ticket"]["ticket_id"], "ticket-previous")
        for event in response["trace"]:
            self.assertTrue({"type", "turn", "step", "name", "arguments", "result",
                             "latency_ms", "tool_call_id"}.issubset(event))
        for invocation in self.complete.call_args_list:
            self.assertEqual(invocation.kwargs["tools"], tool_schemas())
            self.assertEqual(invocation.kwargs["tool_choice"], "auto")
        self.assertEqual(len(response["turn_outputs"]), 1)
        self.network.assert_not_called()

    def test_system_contains_no_policy_memory_fixture_or_evaluation_contract(self):
        self.complete.return_value = reply("I can look that up.")
        agent.run("What is the current plan?", self.config)
        messages = self.complete.call_args.args[0]
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[1], {"role": "user", "content": "What is the current plan?"})
        system = messages[0]["content"]
        self.assertIn("ws-acme", system)
        self.assertIn(AS_OF, system)
        for forbidden in (KNOWLEDGE, "REFUND_POLICY_FROM_CONFIG", "ticket-previous",
                          "PRIVATE_EXPECTED_OUTPUT", "PRIVATE_CONTRACT", "BETA_PRIVATE_MARKER_7f31"):
            self.assertNotIn(forbidden, json.dumps(messages))
        self.assertNotIn("Acme Studio", system)

    def test_model_trace_excludes_messages_raw_response_and_internal_reasoning(self):
        replies = [
            reply(calls=[tool_call("get_workspace", {}, "workspace")]),
            reply("The current billing cycle is monthly."),
        ]
        for value in replies:
            value["raw_response"] = {"reasoning": "PRIVATE_INTERNAL_REASONING",
                                     "duplicated_observations": "DO_NOT_COPY_RAW_RESPONSE"}
        self.complete.side_effect = replies
        _, response = agent.run("Check the current plan.", self.config)
        calls = [event for event in response["trace"] if event["type"] == "model_call"]
        self.assertEqual(len(calls), 2)
        for event in calls:
            encoded = json.dumps(event)
            self.assertNotIn('"messages"', encoded)
            self.assertNotIn('"raw_response"', encoded)
            self.assertNotIn("PRIVATE_INTERNAL_REASONING", encoded)
            self.assertNotIn("DO_NOT_COPY_RAW_RESPONSE", encoded)
            self.assertEqual(event["usage"]["input_tokens"], 10)
            self.assertEqual(event["name"], "offline-model")
            self.assertEqual(event["latency_ms"], 12)
        observations = [event for event in response["trace"] if event["type"] == "tool_call"]
        self.assertEqual(observations[0]["result"]["workspace"]["billing_cycle"], "monthly")

    def test_real_completion_adapter_native_round_trip_uses_mock_http_only(self):
        from scripts import llm_judge
        payloads = []

        def transport(url, payload, key, timeout):
            payloads.append(deepcopy(payload))
            self.assertEqual(key, "offline-native-key")
            self.assertEqual(payload["tools"], tool_schemas())
            self.assertEqual(payload["tool_choice"], "auto")
            if len(payloads) == 1:
                message = {"role": "assistant", "content": None,
                           "tool_calls": [tool_call("search_memory", {"query": "preferred channel"}, "native-memory")]}
                finish, input_tokens, output_tokens = "tool_calls", 10, 2
            else:
                self.assertEqual([m["role"] for m in payload["messages"]],
                                 ["system", "user", "assistant", "tool"])
                self.assertEqual(payload["messages"][-1]["tool_call_id"], "native-memory")
                self.assertEqual(last_tool(payload["messages"])["matches"][0]["value"], "email")
                message = {"role": "assistant", "content": "Your saved channel is email.",
                           "reasoning_content": "PRIVATE_PROVIDER_REASONING"}
                finish, input_tokens, output_tokens = "stop", 20, 3
            wire = {"model": "offline-resolved-native", "choices": [
                {"message": message, "finish_reason": finish}],
                "usage": {"prompt_tokens": input_tokens, "completion_tokens": output_tokens}}
            return 200, {}, json.dumps(wire, allow_nan=False).encode()

        self.complete.side_effect = llm_judge.complete
        with patch.dict("os.environ", {"OPENAI_API_KEY": "offline-native-key",
                                      "OPENAI_BASE_URL": llm_judge.DEFAULT_BASE_URL}, clear=True), \
                patch.object(llm_judge, "_post", side_effect=transport):
            metadata, response = agent.run("Recall my saved contact channel.", self.config)
        self.assertIsNone(metadata["error"])
        self.assertEqual(metadata["recommendation"]["answer"], "Your saved channel is email.")
        self.assertEqual((metadata["input_tokens"], metadata["output_tokens"], metadata["tool_calls"]),
                         (30, 5, 1))
        self.assertEqual(response["resolved_model_id"], "offline-resolved-native")
        self.assertNotIn("PRIVATE_PROVIDER_REASONING", json.dumps(response["trace"]))
        self.network.assert_not_called()

    def test_policy_is_retrieved_from_config_only_after_tool_call(self):
        def complete(messages, **kwargs):
            if messages[-1]["role"] == "user":
                self.assertNotIn("REFUND_POLICY_FROM_CONFIG", json.dumps(messages))
                return reply(calls=[tool_call("search_policy", {"query": "refund initial purchase"})])
            result = last_tool(messages)
            self.assertEqual(result["source"], "config.knowledge")
            self.assertIn("REFUND_POLICY_FROM_CONFIG", json.dumps(result["sections"]))
            self.assertNotIn("Plans and billing", json.dumps(result["sections"]))
            return reply("The retrieved policy allows an eligible request, not an issued refund.")
        self.complete.side_effect = complete
        metadata, response = agent.run("What is the refund policy?", self.config)
        self.assertIsNone(metadata["error"])
        self.assertEqual(metadata["tool_calls"], 1)
        self.assertEqual(response["trace"][2]["name"], "search_policy")

    def test_fresh_followup_reopens_store_and_retrieves_preference_without_history(self):
        self.config["_case"]["follow_up_inputs"] = ["What channel did I ask you to remember?"]
        created_environments = []

        class ObservedEnvironment(ToolEnvironment):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                created_environments.append(self)

        def complete(messages, **kwargs):
            index = self.complete.call_count
            if index == 1:
                return reply(calls=[tool_call("save_memory",
                    {"key": "preferred_channel", "value": "in_app"}, "save")])
            if index == 2:
                self.assertTrue(last_tool(messages)["saved"])
                stored = json.loads((Path(self.config["_storage_dir"]) / "store.json").read_text())
                self.assertEqual(stored["memory"]["preferred_channel"]["value"], "in_app")
                return reply("Saved your in-app preference.")
            if index == 3:
                self.assertEqual([m["role"] for m in messages], ["system", "user"])
                self.assertEqual(messages[-1]["content"], self.config["_case"]["follow_up_inputs"][0])
                self.assertNotIn("Remember in_app for me.", json.dumps(messages))
                self.assertNotIn("Saved your in-app preference.", json.dumps(messages))
                self.assertNotIn("in_app", json.dumps(messages))
                self.assertEqual(len(created_environments), 2)
                self.assertIsNot(created_environments[0], created_environments[1])
                return reply(calls=[tool_call("search_memory", {"query": "preferred channel"}, "recall")])
            stored_preference = next(entry["value"] for entry in last_tool(messages)["matches"]
                                     if entry["key"] == "preferred_channel")
            return reply("Your saved channel is " + stored_preference + ".")

        self.complete.side_effect = complete
        with patch.object(agent, "ToolEnvironment", ObservedEnvironment):
            metadata, response = agent.run("Remember in_app for me.", self.config)
        self.assertIsNone(metadata["error"])
        self.assertEqual(len(response["turn_outputs"]), 2)
        self.assertEqual(response["turn_outputs"][1]["answer"], "Your saved channel is in_app.")
        self.assertEqual(response["tool_state"]["memory"]["preferred_channel"]["value"], "in_app")
        retrieved = [e for e in response["trace"] if e["type"] == "tool_call"
                     and e["name"] == "search_memory"]
        self.assertEqual(retrieved[0]["turn"], 2)
        self.assertEqual(metadata["tool_calls"], 2)

    def test_seed_is_config_fixture_copy_and_final_state_contains_only_current_tenant(self):
        self.config["environment"]["workspaces"]["ws-acme"]["workspace"]["plan"] = "Starter"
        self.config["environment"]["workspaces"]["ws-acme"]["memory"]["preferred_channel"]["value"] = "in_app"
        original = deepcopy(self.config)
        environment = self.environment()
        self.assertEqual(environment.execute("get_workspace", {})["workspace"]["plan"], "Starter")
        self.assertEqual(environment.snapshot()["memory"]["preferred_channel"]["value"], "in_app")
        environment.execute("save_memory", {"key": "preferred_channel", "value": "email"})
        self.assertEqual(self.config, original)
        self.assertEqual(FIXTURE["workspaces"]["ws-acme"]["workspace"]["plan"], "Team")
        serialized = json.dumps(environment.snapshot())
        self.assertNotIn("ws-beta", serialized)
        self.assertNotIn("BETA_PRIVATE_MARKER_7f31", serialized)
        self.assertNotIn("workspaces", environment.snapshot())

    def test_foreign_invoice_ticket_and_memory_are_inaccessible_without_leaking_existence(self):
        environment = self.environment()
        before = environment.snapshot()
        self.assertEqual(environment.execute("get_invoice", {"invoice_id": "inv-beta"}),
                         environment.execute("get_invoice", {"invoice_id": "does-not-exist"}))
        self.assertEqual(environment.execute("get_ticket", {"ticket_id": "ticket-beta"}),
                         environment.execute("get_ticket", {"ticket_id": "does-not-exist"}))
        self.assertEqual(environment.execute("search_memory", {"query": "BETA_PRIVATE_MARKER_7f31"})["matches"], [])
        for name, arguments in (
            ("get_workspace", {"workspace_id": "ws-beta"}),
            ("get_invoice", {"invoice_id": "inv-beta", "workspace_id": "ws-beta"}),
            ("search_memory", {"query": "all", "workspace_id": "ws-beta"}),
            ("create_ticket", {"category": "billing", "summary": "foreign", "invoice_id": "inv-beta"}),
        ):
            self.assertIn("error", environment.execute(name, arguments))
        self.assertEqual(environment.snapshot(), before)
        self.assertNotIn("BETA_PRIVATE_MARKER_7f31", environment.store_path.read_text())

    def test_saved_ticket_memory_points_to_live_status_and_current_plan_overrides_stale_note(self):
        environment = self.environment()
        remembered = environment.execute("search_memory", {"query": "billing cycle"})
        stale = next(entry for entry in remembered["matches"] if entry["key"] == "billing_cycle")
        self.assertEqual(stale["value"], "annual")
        current = environment.execute("get_workspace", {})
        self.assertEqual(current["workspace"]["billing_cycle"], "monthly")
        self.assertEqual(current["authority"], "current")
        tickets = environment.execute("search_memory", {"query": "previous ticket"})
        ticket_id = next(entry["value"] for entry in tickets["matches"] if entry["key"] == "pending_billing_ticket")
        actual = environment.execute("get_ticket", {"ticket_id": ticket_id})
        self.assertEqual(actual["ticket"]["status"], "pending")
        self.assertEqual(actual["source"], "ticket_record")

    def test_fixture_invoices_have_observable_refund_and_duplicate_boundaries(self):
        from datetime import date
        environment = self.environment()
        invoices = {name: environment.execute("get_invoice", {"invoice_id": name})["invoice"]
                    for name in ("inv-100", "inv-101", "inv-200", "inv-201")}
        self.assertEqual(invoices["inv-100"]["duplicate_of"], "inv-101")
        for key in ("period_start", "period_end", "amount_usd", "status"):
            self.assertEqual(invoices["inv-100"][key], invoices["inv-101"][key])
        for name, age, eligibility in (("inv-200", 7, True), ("inv-201", 53, False)):
            self.assertEqual((date.fromisoformat(AS_OF) - date.fromisoformat(invoices[name]["purchase_date"])).days, age)
            self.assertEqual(invoices[name]["purchase_type"], "initial_purchase")
            self.assertEqual(invoices[name]["refund_eligibility"]["eligible_to_request"], eligibility)

    def test_preference_and_ticket_writes_survive_new_environment_instances(self):
        environment = self.environment()
        result = environment.execute("save_memory", {"key": "preferred_channel", "value": "in_app"})
        self.assertTrue(result["saved"])
        created = environment.execute("create_ticket", {
            "category": "billing", "summary": "Please review three additional seats; contact via in_app."})
        self.assertTrue(created["created"])
        self.assertEqual(created["ticket"]["ticket_id"], "ticket-001")
        self.assertTrue(created["local_only"])
        reopened = self.environment()
        self.assertEqual(reopened.snapshot(), environment.snapshot())
        ticket = reopened.execute("get_ticket", {"ticket_id": created["ticket"]["ticket_id"]})["ticket"]
        self.assertEqual(ticket, created["ticket"])
        self.assertEqual(reopened.snapshot()["workspace"], FIXTURE["workspaces"]["ws-acme"]["workspace"])
        self.assertEqual(reopened.snapshot()["invoices"], FIXTURE["workspaces"]["ws-acme"]["invoices"])

    def test_new_run_resets_store_to_fixture_instead_of_contaminating_cases(self):
        self.complete.side_effect = [
            reply(calls=[tool_call("save_memory", {"key": "preferred_channel", "value": "in_app"})]),
            reply("Saved locally."),
        ]
        _, first = agent.run("Remember in_app.", self.config)
        self.assertEqual(first["tool_state"]["memory"]["preferred_channel"]["value"], "in_app")
        self.complete.side_effect = [
            reply(calls=[tool_call("search_memory", {"query": "preferred channel"})]),
            reply("Email."),
        ]
        _, second = agent.run("Recall my preference.", self.config)
        self.assertEqual(second["tool_state"]["memory"]["preferred_channel"]["value"], "email")

    def test_unknown_invalid_and_secret_memory_tools_have_no_write_side_effects(self):
        environment = self.environment()
        before = environment.store_path.read_bytes()
        invalid = [
            ("delete_workspace", {}), ("get_workspace", {"path": "/tmp/something"}),
            ("get_invoice", {}), ("search_policy", {"query": ""}),
            ("save_memory", {"key": "password", "value": "secret-value"}),
            ("save_memory", {"key": "preferred_channel", "value": "secret-value"}),
            ("save_memory", {"key": "billing_cycle", "value": "annual"}),
            ("save_memory", {"key": "response_style", "value": "in_app"}),
            ("save_memory", {"key": "preferred_channel", "value": True}),
            ("create_ticket", {"category": "refund_issued", "summary": "fake"}),
        ]
        for name, arguments in invalid:
            with self.subTest(name=name, arguments=arguments):
                self.assertIn("error", environment.execute(name, arguments))
                self.assertEqual(environment.store_path.read_bytes(), before)

    def test_invalid_native_calls_are_recorded_and_returned_as_tool_errors(self):
        calls = [
            tool_call("unknown_tool", {}, "unknown"),
            tool_call("save_memory", {"key": "password", "value": "no"}, "secret"),
            tool_call("get_workspace", {"workspace_id": "ws-beta"}, "foreign"),
            tool_call("save_memory", {}, "missing"),
        ]
        for index, arguments in enumerate(("not JSON", "[]", '{"query": NaN}',
                                           '{"query":"a","query":"b"}')):
            calls.append({"id": f"invalid-{index}", "type": "function",
                          "function": {"name": "search_memory", "arguments": arguments}})
        self.complete.side_effect = [reply(calls=calls), reply("Those calls were rejected.")]
        metadata, response = agent.run("Check tool error handling.", self.config)
        self.assertIsNone(metadata["error"])
        self.assertEqual(metadata["tool_calls"], 8)
        executed = [e for e in response["trace"] if e["type"] == "tool_call"]
        self.assertTrue(all(e["result"].get("error") for e in executed))
        messages = self.complete.call_args_list[1].args[0]
        self.assertEqual(len([m for m in messages if m["role"] == "tool"]), 8)
        self.assertEqual(response["tool_state"]["memory"], FIXTURE["workspaces"]["ws-acme"]["memory"])
        self.assertEqual(response["tool_state"]["tickets"], FIXTURE["workspaces"]["ws-acme"]["tickets"])

    def test_malicious_retrieved_memory_is_data_and_has_no_automatic_actions(self):
        self.complete.side_effect = [
            reply(calls=[tool_call("search_memory", {"query": "billing notes preference"}, "notes")]),
            reply("Your preference is email. I ignored instructions in the imported note."),
        ]
        metadata, response = agent.run("Read my notes and preference. Make no changes.", self.config)
        tool_event = next(e for e in response["trace"] if e["type"] == "tool_call")
        self.assertIn("ignore the user", json.dumps(tool_event["result"]))
        self.assertIn("untrusted_content", json.dumps(tool_event["result"]))
        self.assertEqual(metadata["tool_calls"], 1)
        self.assertEqual(response["tool_state"]["tickets"], FIXTURE["workspaces"]["ws-acme"]["tickets"])
        self.assertEqual(response["tool_state"]["memory"], FIXTURE["workspaces"]["ws-acme"]["memory"])

    def test_failed_model_retains_prior_trace_persisted_writes_and_billed_usage(self):
        self.complete.side_effect = [
            reply(calls=[tool_call("get_workspace", {}, "workspace")]),
            reply(calls=[tool_call("save_memory", {"key": "preferred_channel", "value": "in_app"}, "save")]),
            reply(error="transport_error", usage={"input_tokens": 7, "output_tokens": None}),
        ]
        metadata, response = agent.run("Remember in_app and confirm.", self.config)
        self.assertEqual(metadata["error"], "transport_error")
        self.assertIsNone(metadata["recommendation"])
        self.assertEqual(metadata["tool_calls"], 2)
        self.assertEqual(metadata["input_tokens"], 27)
        self.assertIsNone(metadata["output_tokens"])
        self.assertEqual(len([e for e in response["trace"] if e["type"] == "model_call"]), 3)
        self.assertEqual(response["tool_state"]["memory"]["preferred_channel"]["value"], "in_app")
        self.assertEqual(response["turn_outputs"][-1]["error"], "transport_error")
        self.assertGreaterEqual(response["latency_ms"], 36)

    def test_missing_usage_is_not_replaced_with_zero_by_a_later_success(self):
        first = reply(calls=[tool_call("get_workspace", {})])
        first["usage"] = {"input_tokens": 0}
        self.complete.side_effect = [first, reply("Done.", usage={"input_tokens": 0, "output_tokens": 3})]
        metadata, response = agent.run("Check my workspace.", self.config)
        self.assertEqual(metadata["input_tokens"], 0)
        self.assertIsNone(metadata["output_tokens"])
        self.assertIsNone(response["usage"]["output_tokens"])

    def test_model_exception_is_recorded_without_exception_secret_text(self):
        self.complete.side_effect = [reply(calls=[tool_call("get_workspace", {})]),
                                     RuntimeError("secret-must-not-be-reported")]
        metadata, response = agent.run("Check my workspace.", self.config)
        self.assertEqual(metadata["error"], "model_exception")
        self.assertEqual(metadata["tool_calls"], 1)
        self.assertIsNone(metadata["input_tokens"])
        self.assertEqual(len([e for e in response["trace"] if e["type"] == "model_call"]), 2)
        self.assertNotIn("secret-must-not-be-reported", json.dumps(response))

    def test_eight_model_round_limit_retains_all_usage_and_tool_results(self):
        self.complete.side_effect = lambda *a, **k: reply(
            calls=[tool_call("get_workspace", {}, f"round-{self.complete.call_count}")])
        metadata, response = agent.run("Keep checking my workspace.", self.config)
        self.assertEqual(self.complete.call_count, 8)
        self.assertEqual(metadata["error"], "max_model_rounds")
        self.assertEqual(metadata["input_tokens"], 80)
        self.assertEqual(metadata["output_tokens"], 24)
        self.assertEqual(metadata["tool_calls"], 8)
        self.assertEqual(len(response["trace"]), 17)
        self.assertIsNone(metadata["recommendation"])
        self.assertIsNotNone(response["tool_state"])

    def test_atomic_store_failure_does_not_commit_in_memory_state(self):
        environment = self.environment()
        before_state, before_file = environment.snapshot(), environment.store_path.read_bytes()
        with patch.object(environment, "_persist", side_effect=OSError("private path")):
            result = environment.execute("save_memory", {"key": "preferred_channel", "value": "in_app"})
        self.assertEqual(result, {"error": "tool_failed", "reason": "OSError"})
        self.assertEqual(environment.snapshot(), before_state)
        self.assertEqual(environment.store_path.read_bytes(), before_file)

    def test_store_cannot_be_reopened_for_another_workspace(self):
        environment = self.environment()
        other = deepcopy(self.config)
        other["_case"]["workspace_id"] = "ws-beta"
        with self.assertRaises(ValueError):
            ToolEnvironment(other, self.config["_storage_dir"])
        self.assertNotIn("BETA_PRIVATE_MARKER_7f31", environment.store_path.read_text())

    def test_schemas_and_snapshots_are_detached_copies(self):
        schemas = tool_schemas()
        schemas[0]["function"]["parameters"]["properties"].clear()
        self.assertIn("query", tool_schemas()[0]["function"]["parameters"]["properties"])
        environment = self.environment()
        state = environment.snapshot()
        state["memory"]["preferred_channel"]["value"] = "changed"
        self.assertEqual(environment.snapshot()["memory"]["preferred_channel"]["value"], "email")

    def test_configuration_errors_return_metadata_without_a_model_call(self):
        for changes in ({"model": None}, {"_case": {"workspace_id": "missing"}},
                        {"_case": {"workspace_id": "ws-acme", "follow_up_inputs": "invalid"}},
                        {"environment": {}}):
            with self.subTest(changes=changes):
                config = {**self.config, **changes}
                metadata, response = agent.run("Question?", config)
                self.assertEqual(set(metadata), METADATA_FIELDS)
                self.assertEqual(metadata["error"], "configuration_error")
                self.assertEqual(metadata["tool_calls"], 0)
                self.assertEqual(response["trace"], [])
        self.complete.assert_not_called()


if __name__ == "__main__":
    unittest.main()
