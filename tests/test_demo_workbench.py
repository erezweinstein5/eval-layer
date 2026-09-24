"""Offline workbench workflows; all completions/judges use labeled fake results.

Only temporary run directories are written. Regression tests cover failed-call
billing, isolated judge evidence, generation constraints, and secret redaction.
Generic cases bypass scenario attachment; scenario and native agent behavior
have separate test suites. Rich evidence fixtures are explicitly mocked.
"""

from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from demo import agent as demo_agent
from demo import server
from scripts import jev_judge, llm_judge


MODEL = "offline-test-model"
METADATA_FIELDS = {
    "recommendation", "latency_ms", "tool_calls", "input_tokens",
    "output_tokens", "model_id", "error",
}


def generated_suite():
    """Ten mocked cases obey the actual generator's requested coverage."""
    return {
        "rubric": {
            "name": "Offline support rubric", "version": "1.0", "pass_threshold": 0.8,
            "dimensions": [
                {"name": name, "scale": scale, "weight": weight,
                 "levels": {str(i): f"{name} concrete descriptor {i}"
                            for i in range(1, scale + 1)}}
                for name, scale, weight in (
                    ("policy_accuracy", 3, 0.4), ("coverage", 5, 0.4), ("format", 3, 0.2)
                )
            ],
        },
        "test_cases": [
            {"id": f"tc-{i + 1:02}", "input": f"Offline policy question {i + 1}?",
             "expected_output": f"Offline reference sketch {i + 1}.",
             "metadata": {"difficulty": difficulty, "category": f"boundary-{i + 1}"}}
            for i, difficulty in enumerate(["easy"] * 4 + ["medium"] * 3 + ["hard"] * 3)
        ],
    }


def completion(text, *, input_tokens=100, output_tokens=40, cached_input_tokens=20):
    return {
        "text": text, "usage": {
            "input_tokens": input_tokens, "output_tokens": output_tokens,
            "cached_input_tokens": cached_input_tokens,
            "completion_tokens_details": {"reasoning_tokens": output_tokens // 2},
        },
        "model_id": MODEL, "resolved_model_id": MODEL + "-resolved",
        "latency_ms": 12, "raw_response": {"offline_mock": True},
        "tool_calls": [], "assistant_message": {"role": "assistant", "content": text},
    }


def judge_result(backend, rubric):
    return {
        "backend": backend,
        "model_id": "jev-1.13.0" if backend == "jev" else MODEL,
        "resolved_model_id": f"offline-{backend}-resolved",
        "scores": {d["name"]: d["scale"] for d in rubric["dimensions"]},
        "details": {
            d["name"]: {
                "reasoning": None if backend == "jev" else "Offline descriptor match.",
                "evidence": None if backend == "jev" else ["Offline recorded observation."],
                "suggestion": None if backend == "jev" else "Offline suggestion.",
                "confidence": 0.9 if backend == "jev" else "high",
            } for d in rubric["dimensions"]
        },
        "overall_reasoning": None if backend == "jev" else "Offline mock assessment.",
        "usage": ({"input_tokens": 150, "output_tokens": 0} if backend == "jev" else
                  {"input_tokens": 200, "output_tokens": 60, "cached_input_tokens": 50}),
        "latency_ms": 4 if backend == "jev" else 20,
        "raw_response": {"offline_mock": True},
    }


def evidence_for(row, knowledge):
    """Every saved evidence field must reach each judge without substitutions."""
    return {
        "input": row["input"], "context": knowledge, "agent_output": row["agent_output"],
        "expected_output": row["expected_output"], "tool_trace": row["trace"],
        "session_outputs": row["turn_outputs"], "independent_checks": row["checks"],
        "final_tool_state": row["tool_state"],
    }


class WorkbenchTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.directory = Path(self.stack.enter_context(tempfile.TemporaryDirectory(
            prefix="eval-workbench-offline-")))
        self.stack.enter_context(patch.dict("os.environ", {
            "OPENAI_API_KEY": "offline-llm-key",
            "TYPESAFE_API_KEY": "offline-jev-key",
        }, clear=True))
        # Guard the underlying adapters too: a missing high-level mock cannot
        # accidentally reach a live provider or use the machine's credentials.
        for adapter in (llm_judge, jev_judge):
            self.stack.enter_context(patch.object(
                adapter, "build_opener", side_effect=AssertionError("network forbidden")))
        self.payload = generated_suite()
        self.attach_contracts = self.stack.enter_context(patch.object(
            server, "attach_contracts", side_effect=lambda cases: cases))
        self.observed_judges = []
        self.generator = self.stack.enter_context(patch.object(
            server, "complete", side_effect=lambda *a, **k: completion(
                json.dumps(self.payload), input_tokens=1000, output_tokens=200,
                cached_input_tokens=200)))
        # Exercise the real demo.agent.run and its seven-field metadata mapping.
        self.agent_completion = self.stack.enter_context(patch.object(
            demo_agent, "complete", side_effect=lambda messages, **kwargs: completion(
                "OFFLINE MOCK answer to " + messages[-1]["content"])))
        self.judges = {}
        for backend, target in (("llm", "evaluate_llm"), ("jev", "evaluate_jev")):
            self.judges[backend] = self.stack.enter_context(patch.object(
                server, target, side_effect=self.make_judge(backend)))
        self.pricing = {
            "llm": {"model": MODEL, "input_usd_per_million": 2, "output_usd_per_million": 8,
                    "cached_input_usd_per_million": 1, "note": "Offline fixture rates"},
            "jev": {"model": "jev-1.13.0", "input_usd_per_million": 0.5, "output_usd_per_million": 0,
                    "note": "Offline fixture rates"},
        }
        self.app = server.Workbench(self.directory, deepcopy(self.pricing), MODEL)

    def test_bedrock_judge_is_independent_of_agent_and_generation(self):
        self.work("generate")
        self.work("run-agent")
        saved_rows = deepcopy(self.app.state['agent_run']['rows'])
        self.app.pricing['agent'] = deepcopy(self.app.pricing['llm'])
        self.app.pricing['llm'].update(model='us.anthropic.claude-opus-5-5', provider='bedrock', region='us-east-1', effort='high')
        with patch.object(server, 'evaluate_bedrock', side_effect=self.make_judge('llm')) as judge:
            self.work('compare')
        self.assertEqual(judge.call_count, 10)
        self.judges['llm'].assert_not_called()
        self.assertEqual(self.app.state['agent']['model'], MODEL)
        self.assertEqual(self.app.state['agent_run']['rows'], saved_rows)
        comparison = self.app.state['comparison']
        self.assertEqual(comparison['summary']['llm']['model'], 'us.anthropic.claude-opus-5-5')
        request = comparison['rows'][0]['judge_requests']['llm']
        self.assertIn('system', request)
        self.assertNotIn('temperature', request)
        self.assertEqual(comparison['evaluation_config']['backends']['llm']['parameters']['region'], 'us-east-1')

    def make_judge(self, backend):
        def evaluate(state, rubric, **kwargs):
            self.observed_judges.append(
                (backend, deepcopy(state), deepcopy(rubric), deepcopy(kwargs)))
            return judge_result(backend, rubric)
        return evaluate

    def work(self, action):
        self.app._work(action)
        self.assertEqual(self.app.state["job"]["status"], "complete",
                         self.app.state["job"])

    def assert_no_judges(self):
        for judge in self.judges.values():
            judge.assert_not_called()

    def mock_rich_agent(self, *, failed_case=None):
        """Mock agent/scenario boundaries to exercise evidence transport and gates."""
        def run(question, config):
            case_id = config["_case"]["id"]
            answer = "OFFLINE MOCK persisted preference for " + case_id
            response = completion(answer)
            response.update(
                trace=[
                    {"type": "model_call", "turn": 1,
                     "usage": {"input_tokens": 60, "output_tokens": 20}},
                    {"type": "tool_call", "turn": 1, "id": case_id + "-write",
                     "name": "save_memory",
                     "arguments": {"key": "preferred_channel", "value": "in_app"},
                     "result": {"saved": {"preferred_channel": "in_app"}}},
                    {"type": "model_call", "turn": 2,
                     "usage": {"input_tokens": 40, "output_tokens": 20}},
                    {"type": "tool_call", "turn": 2, "id": case_id + "-read",
                     "name": "search_memory", "arguments": {"query": "preferred channel"},
                     "result": {"matches": [{"key": "preferred_channel", "value": "in_app"}]}},
                ],
                turn_outputs=[
                    {"turn": 1, "input": question, "output": {"answer": "Preference saved."}},
                    {"turn": 2, "input": "Recall my preference.",
                     "output": {"answer": "Retrieved in_app from memory."}},
                ],
                tool_state={"workspace_id": "offline-workspace",
                            "memory": {"preferred_channel": "in_app"},
                            "tickets": {}, "case_id": case_id},
            )
            return {
                "recommendation": {"answer": answer}, "latency_ms": 12,
                "tool_calls": 2, "input_tokens": 100, "output_tokens": 40,
                "model_id": MODEL, "error": None,
            }, response

        def checks(case, response):
            return [{"name": "Offline independently supplied check",
                     "expected": {"preferred_channel": "in_app"},
                     "actual": deepcopy(response["tool_state"]["memory"]),
                     "passed": case["id"] != failed_case}]

        self.stack.enter_context(patch.object(server, "run_agent_case", side_effect=run))
        self.stack.enter_context(patch.object(server, "check_execution", side_effect=checks))

    def test_generate_uses_skill_agent_policy_and_saves_valid_artifacts(self):
        self.work("generate")
        messages = self.generator.call_args.args[0]
        self.assertIs(self.generator.call_args.kwargs["json_mode"], False)
        self.assertEqual(self.generator.call_args.kwargs["model_id"], MODEL)
        supplied = json.loads(messages[1]["content"])
        agent = self.app.state["agent"]
        self.assertEqual(supplied["agent_source"], agent["source"])
        self.assertEqual(supplied["agent_instructions"], agent["system_prompt"])
        self.assertEqual(supplied["knowledge"], agent["knowledge"])
        self.assertIn((server.ROOT / "SKILL.md").read_text(), messages[0]["content"])
        suite = self.app.state["suite"]
        self.assertEqual(suite["rubric"], self.payload["rubric"])
        self.assertEqual(suite["test_cases"], self.payload["test_cases"])
        self.attach_contracts.assert_called_once_with(self.payload["test_cases"])
        self.assertEqual(suite["agent_hash"], server.digest(agent))
        self.assertEqual(suite["generator_model"], MODEL + "-resolved")
        self.assertAlmostEqual(suite["cost_usd"], 0.0034)
        for filename, expected in (("rubric.json", suite["rubric"]),
                                   ("test_cases.json", suite["test_cases"]),
                                   ("suite.json", suite)):
            self.assertEqual(json.loads((self.directory / "generated" / filename).read_text()),
                             expected)
        self.generator.assert_called_once()
        self.agent_completion.assert_not_called()
        self.assert_no_judges()

    def test_run_all_saves_agent_metadata_identical_evidence_and_comparison(self):
        self.work("run-all")
        state = self.app.snapshot()
        rows = state["agent_run"]["rows"]
        self.assertEqual(len(rows), 10)
        self.assertEqual(self.agent_completion.call_count, 10)
        self.assertEqual(self.generator.call_count, 1)
        self.assertAlmostEqual(state["agent_run"]["cost_usd"], 0.005)
        self.assertEqual(state["agent_run"]["usage"],
                         {"input_tokens": 1000, "output_tokens": 400})
        for case, row, agent_call in zip(self.payload["test_cases"], rows,
                                        self.agent_completion.call_args_list):
            self.assertEqual(set(row["agent_metadata"]), METADATA_FIELDS)
            self.assertEqual(row["agent_metadata"]["tool_calls"], 0)
            self.assertIsNone(row["agent_metadata"]["error"])
            self.assertEqual(row["agent_output"], row["agent_metadata"]["recommendation"])
            self.assertEqual(agent_call.args[0][-1]["content"], case["input"])
            exposed = agent_call.kwargs["tools"]
            self.assertIsInstance(exposed, list)
            self.assertTrue(exposed)
            self.assertTrue({"search_policy", "search_memory"} <=
                            {tool["function"]["name"] for tool in exposed})
            self.assertNotIn(state["agent"]["knowledge"], agent_call.args[0][0]["content"])
            self.assertNotIn(case["expected_output"], agent_call.args[0][0]["content"])
        comparison = state["comparison"]
        self.assertEqual(comparison["agent_outputs_hash"], server.digest(rows))
        self.assertEqual(comparison["agent"], state["agent"])
        self.assertEqual(comparison["rubric"], state["suite"]["rubric"])
        self.assertEqual(comparison["agent_run"],
                         {k: v for k, v in state["agent_run"].items() if k != "rows"})
        self.assertEqual(comparison["generation"],
                         {k: v for k, v in state["suite"].items()
                          if k not in ("rubric", "test_cases")})
        for index, row in enumerate(rows):
            pair = self.observed_judges[2 * index:2 * index + 2]
            self.assertEqual([call[0] for call in pair],
                             ["jev", "llm"] if index % 2 == 0 else ["llm", "jev"])
            expected = evidence_for(row, state["agent"]["knowledge"])
            for backend, evidence, rubric, kwargs in pair:
                self.assertEqual(evidence, expected)
                self.assertEqual(rubric, self.payload["rubric"])
                self.assertEqual(kwargs["model_id"], "jev-1.13.0" if backend == "jev" else MODEL)
                if backend == "jev":
                    self.assertEqual(kwargs["max_attempts"], 1)
        for backend, expected_cost in (("llm", 0.0083), ("jev", 0.00075)):
            summary = comparison["summary"][backend]
            self.assertEqual((summary["n_scored"], summary["n_failed"], summary["n_total"]),
                             (10, 0, 10))
            self.assertEqual(summary["pass_rate"], 1)
            self.assertAlmostEqual(summary["cost_usd"], expected_cost)
        self.assertEqual(comparison["summary"]["paired_n"], 10)
        self.assertEqual(comparison["summary"]["verdict_disagreements"], 0)
        self.assertEqual(comparison["summary"]["mean_score_gap"], 0)
        self.assertAlmostEqual(comparison["summary"]["cost_savings_pct"],
                               (1 - 0.00075 / 0.0083) * 100)
        saved_rows = [json.loads(line) for line in
                      (self.directory / "agent_outputs.jsonl").read_text().splitlines()]
        self.assertEqual(saved_rows, rows)
        self.assertEqual(json.loads((self.directory / (comparison["run_id"] + ".json")).read_text()),
                         comparison)

    def test_rich_evidence_and_failed_checks_survive_comparison_and_history(self):
        self.mock_rich_agent(failed_case="tc-01")
        self.work("run-all")
        snapshot = self.app.snapshot()
        run, comparison = snapshot["agent_run"], snapshot["comparison"]
        self.assertEqual(run["tool_calls"], 20)
        self.assertEqual(run["model_calls"], 20)
        self.assertAlmostEqual(run["cost_usd"], 0.005)
        self.assertEqual(comparison["agent_outputs_hash"], server.digest(run["rows"]))
        for index, row in enumerate(run["rows"]):
            expected = evidence_for(row, snapshot["agent"]["knowledge"])
            for field in ("tool_trace", "session_outputs", "independent_checks", "final_tool_state"):
                self.assertTrue(expected[field], f"{field} must exercise nonempty evidence")
            for _, actual, _, _ in self.observed_judges[index * 2:index * 2 + 2]:
                self.assertEqual(actual, expected)
            self.assertEqual(evidence_for(comparison["rows"][index], snapshot["agent"]["knowledge"]),
                             expected)
        for backend, cost in (("llm", 0.0083), ("jev", 0.00075)):
            failed = comparison["rows"][0]["judges"][backend]
            self.assertTrue(failed["rubric_passed"])
            self.assertFalse(failed["checks_passed"])
            self.assertFalse(failed["passed"])
            self.assertEqual(comparison["summary"][backend]["n_scored"], 10)
            self.assertEqual(comparison["summary"][backend]["pass_rate"], 0.9)
            self.assertAlmostEqual(comparison["summary"][backend]["cost_usd"], cost)
        saved_rows = [json.loads(line) for line in
                      (self.directory / "agent_outputs.jsonl").read_text().splitlines()]
        self.assertEqual(saved_rows, run["rows"])
        self.app.configure({key: snapshot["agent"][key] + "\nNew configuration."
                            for key in ("name", "system_prompt", "knowledge")})
        archived = json.loads((self.directory / (comparison["run_id"] + ".json")).read_text())
        self.assertEqual(archived, comparison)
        self.assertEqual(archived["agent_run"]["tool_calls"], 20)
        self.assertEqual(self.app.state["history"][0]["summary"], comparison["summary"])

    def test_fractional_mixed_scale_scores_and_disagreement_are_reported(self):
        call = 0

        def judge(state, rubric, **kwargs):
            nonlocal call
            result = judge_result("llm", rubric)
            if call == 0:
                result["scores"]["policy_accuracy"] = 1.25
            call += 1
            return result

        self.judges["llm"].side_effect = judge
        self.work("run-all")
        comparison = self.app.state["comparison"]
        first = comparison["rows"][0]["judges"]["llm"]
        self.assertEqual(first["scores"]["policy_accuracy"], 1.25)
        self.assertAlmostEqual(first["weighted_score"], 0.4 * 1.25 / 3 + 0.4 + 0.2)
        self.assertFalse(first["passed"])
        summary = comparison["summary"]
        self.assertEqual(summary["llm"]["pass_rate"], 0.9)
        self.assertEqual(summary["verdict_disagreements"], 1)
        self.assertAlmostEqual(summary["mean_score_gap"], (1 - first["weighted_score"]) / 10)

    def test_recompare_uses_saved_answers_without_regenerating_or_running_agent(self):
        self.work("run-all")
        before = deepcopy(self.app.state)
        original_file = (self.directory / (before["comparison"]["run_id"] + ".json")).read_bytes()
        self.work("compare")
        self.generator.assert_called_once()
        self.assertEqual(self.agent_completion.call_count, 10)
        self.assertEqual(self.app.state["agent_run"], before["agent_run"])
        self.assertEqual(self.observed_judges[:20], self.observed_judges[20:])
        self.assertEqual(len(self.app.state["history"]), 2)
        self.assertNotEqual(self.app.state["history"][0]["run_id"],
                            self.app.state["history"][1]["run_id"])
        self.assertEqual((self.directory / (before["comparison"]["run_id"] + ".json")).read_bytes(),
                         original_file)

    def test_failed_agent_row_is_retained_and_skips_both_judges(self):
        normal = self.agent_completion.side_effect
        call = 0

        def answer(messages, **kwargs):
            nonlocal call
            call += 1
            result = normal(messages, **kwargs)
            if call == 1:
                result.update(error="schema_failed", text=None)
            return result

        self.agent_completion.side_effect = answer
        self.work("run-all")
        state = self.app.state
        row = state["comparison"]["rows"][0]
        self.assertIsNone(row["agent_output"])
        self.assertEqual(row["agent_metadata"]["error"], "schema_failed")
        self.assertAlmostEqual(state["agent_run"]["cost_usd"], 0.005)
        for backend in ("llm", "jev"):
            self.assertEqual(self.judges[backend].call_count, 9)
            self.assertEqual(row["judges"][backend]["error"], "agent_failed")
            summary = state["comparison"]["summary"][backend]
            self.assertEqual((summary["n_scored"], summary["n_total"], summary["n_failed"]),
                             (9, 10, 1))
            self.assertEqual(summary["pass_rate"], 0.9)
        self.assertEqual(state["comparison"]["summary"]["paired_n"], 9)
        self.assertIsNone(state["comparison"]["summary"]["cost_savings_pct"])

    def test_billed_failed_judge_calls_keep_metrics_and_count_against_pass_rate(self):
        def failed(state, rubric, **kwargs):
            result = judge_result("llm", rubric)
            result.pop("scores")
            result.pop("details")
            result.update(error="schema_failed", reason="Offline malformed grade")
            return result

        self.judges["llm"].side_effect = failed
        self.work("run-all")
        comparison = self.app.state["comparison"]
        summary = comparison["summary"]["llm"]
        self.assertEqual((summary["n_scored"], summary["n_failed"]), (0, 10))
        self.assertEqual(summary["pass_rate"], 0)
        self.assertAlmostEqual(summary["cost_usd"], 0.0083)
        self.assertEqual(summary["input_tokens"], 2000)
        self.assertEqual(summary["output_tokens"], 600)
        self.assertEqual(summary["mean_latency_ms"], 20)
        self.assertEqual(comparison["summary"]["paired_n"], 0)
        self.assertIsNone(comparison["summary"]["speedup"])
        self.assertIsNone(comparison["summary"]["cost_savings_pct"])
        for row in comparison["rows"]:
            self.assertNotIn("passed", row["judges"]["llm"])

    def test_unknown_usage_stays_unknown_across_later_successes(self):
        normal = self.agent_completion.side_effect
        call = 0

        def answer(messages, **kwargs):
            nonlocal call
            call += 1
            result = normal(messages, **kwargs)
            if call == 1:
                result["usage"] = {"input_tokens": 100, "output_tokens": None}
            return result

        self.agent_completion.side_effect = answer
        normal_judge = self.judges["llm"].side_effect

        def judge(state, rubric, **kwargs):
            result = normal_judge(state, rubric, **kwargs)
            if state["input"] == self.payload["test_cases"][0]["input"]:
                result["usage"] = {"input_tokens": 200}
            return result

        self.judges["llm"].side_effect = judge
        self.work("run-all")
        run = self.app.state["agent_run"]
        self.assertEqual(run["usage"]["input_tokens"], 1000)
        self.assertIsNone(run["usage"]["output_tokens"])
        self.assertIsNone(run["cost_usd"])
        summary = self.app.state["comparison"]["summary"]
        self.assertEqual(summary["llm"]["input_tokens"], 2000)
        self.assertIsNone(summary["llm"]["output_tokens"])
        self.assertIsNone(summary["llm"]["cost_usd"])
        self.assertIsNone(summary["cost_savings_pct"])

    def test_configuration_invalidates_active_artifacts_but_keeps_history(self):
        self.work("run-all")
        history = deepcopy(self.app.state["history"])
        archived = self.directory / (history[0]["run_id"] + ".json")
        previous = archived.read_bytes()
        original_agent = deepcopy(self.app.state["agent"])
        original_rubric = deepcopy(self.app.state["suite"]["rubric"])
        body = {key: self.app.state["agent"][key] + "\nChanged offline."
                for key in ("name", "system_prompt", "knowledge")}
        self.app.configure(body)
        for field in ("suite", "agent_run", "comparison"):
            self.assertIsNone(self.app.state[field])
        self.assertEqual(self.app.state["history"], history)
        self.assertEqual(archived.read_bytes(), previous)
        archived_run = json.loads(archived.read_text())
        self.assertEqual(archived_run["agent"], original_agent)
        self.assertEqual(archived_run["rubric"], original_rubric)
        self.assertAlmostEqual(archived_run["generation"]["cost_usd"], 0.0034)
        self.assertAlmostEqual(archived_run["agent_run"]["cost_usd"], 0.005)
        reloaded = server.Workbench(self.directory, deepcopy(self.pricing), MODEL)
        self.assertEqual(reloaded.state["history"], history)
        self.assertIsNone(reloaded.state["suite"])
        for field, value in body.items():
            self.assertEqual(reloaded.state["agent"][field], value)
        with self.assertRaisesRegex(ValueError, "Generate"):
            self.app.launch("run-agent")
        with self.assertRaisesRegex(ValueError, "Run the agent"):
            self.app.launch("compare")

    def test_bad_configuration_is_atomic_and_does_not_destroy_saved_results(self):
        self.work("run-all")
        before = deepcopy(self.app.state)
        saved = (self.directory / "state.json").read_bytes()
        good = {key: self.app.state["agent"][key] for key in ("name", "system_prompt", "knowledge")}
        for body in ({}, {**good, "name": ""}, {**good, "knowledge": " "},
                     {**good, "system_prompt": "x" * 40001}):
            with self.subTest(body_keys=list(body)), self.assertRaises(ValueError):
                self.app.configure(body)
            self.assertEqual(self.app.state, before)
            self.assertEqual((self.directory / "state.json").read_bytes(), saved)

    def test_snapshot_is_detached_and_restart_refreshes_prices_without_storing_keys(self):
        self.work("run-all")
        snapshot = self.app.snapshot()
        self.assertEqual(snapshot["credentials"], {"llm": True, "jev": True})
        snapshot["comparison"]["rows"].clear()
        snapshot["agent"]["knowledge"] = "Modified snapshot"
        self.assertEqual(len(self.app.state["comparison"]["rows"]), 10)
        self.assertNotEqual(self.app.state["agent"]["knowledge"], "Modified snapshot")
        for path in self.directory.rglob("*"):
            if path.is_file():
                self.assertNotIn("offline-llm-key", path.read_text())
                self.assertNotIn("offline-jev-key", path.read_text())
        self.assertNotIn("credentials", json.loads((self.directory / "state.json").read_text()))
        updated_prices = deepcopy(self.pricing)
        updated_prices["llm"]["input_usd_per_million"] = 4
        reloaded = server.Workbench(self.directory, updated_prices, MODEL)
        self.assertEqual(reloaded.state["job"]["status"], "idle")
        self.assertEqual(reloaded.state["pricing"], updated_prices)
        self.assertEqual(reloaded.state["comparison"]["pricing"], self.pricing)
        self.assertEqual(reloaded.state["history"], self.app.state["history"])

    def test_suite_changes_reject_stale_agent_outputs_without_judge_calls(self):
        self.work("generate")
        self.work("run-agent")
        self.app.state["suite"]["test_cases"][0]["expected_output"] = "Changed reference"
        self.app._work("compare")
        self.assertEqual(self.app.state["job"]["status"], "failed")
        self.assertIn("Eval set changed", self.app.state["job"]["error"])
        self.assert_no_judges()

    def test_agent_changes_reject_stale_suite_without_running_agent(self):
        self.work("generate")
        self.app.state["agent"]["knowledge"] += "\nChanged policy."
        self.app._work("run-agent")
        self.assertEqual(self.app.state["job"]["status"], "failed")
        self.assertIn("Agent changed", self.app.state["job"]["error"])
        self.agent_completion.assert_not_called()

    def test_launch_preconditions_and_missing_credentials_do_not_start_work(self):
        for action in ("other", "run-agent", "compare"):
            with self.subTest(action=action), self.assertRaises(ValueError):
                self.app.launch(action)
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(ValueError, "OPENAI_API_KEY"):
                self.app.launch("generate")
        with patch.dict("os.environ", {"OPENAI_API_KEY": "offline-llm-key"}, clear=True):
            with self.assertRaisesRegex(ValueError, "TYPESAFE_API_KEY"):
                self.app.launch("run-all")
        self.generator.assert_not_called()
        self.agent_completion.assert_not_called()
        self.assert_no_judges()

    def test_running_job_rejects_configuration_and_concurrent_launch(self):
        entered, release = threading.Event(), threading.Event()
        normal = self.generator.side_effect

        def paused(*args, **kwargs):
            entered.set()
            if not release.wait(3):
                raise AssertionError("offline test did not release generator")
            return normal(*args, **kwargs)

        self.generator.side_effect = paused
        self.app.launch("generate")
        try:
            self.assertTrue(entered.wait(2))
            with self.assertRaisesRegex(ValueError, "already in progress"):
                self.app.launch("generate")
            body = {key: self.app.state["agent"][key]
                    for key in ("name", "system_prompt", "knowledge")}
            with self.assertRaisesRegex(ValueError, "current run"):
                self.app.configure(body)
        finally:
            release.set()
            self.app.worker.join(3)
        self.assertFalse(self.app.worker.is_alive())
        self.assertEqual(self.app.state["job"]["status"], "complete")
        self.generator.assert_called_once()

    def test_duplicate_case_ids_stop_before_agent_or_judges(self):
        self.payload["test_cases"][1]["id"] = self.payload["test_cases"][0]["id"]
        self.app._work("run-all")
        self.assertEqual(self.app.state["job"]["status"], "failed")
        self.assertIsNone(self.app.state["suite"])
        self.agent_completion.assert_not_called()
        self.assert_no_judges()

    def test_failed_regeneration_preserves_previous_run_and_history(self):
        self.work("run-all")
        before = deepcopy(self.app.state)
        self.generator.side_effect = None
        self.generator.return_value = {
            **completion(None), "error": "http_error", "http_status": 429,
        }
        self.app._work("generate")
        self.assertEqual(self.app.state["job"]["status"], "failed")
        for field in ("suite", "agent_run", "comparison", "history"):
            self.assertEqual(self.app.state[field], before[field])
        self.assertEqual(self.agent_completion.call_count, 10)
        for judge in self.judges.values():
            self.assertEqual(judge.call_count, 10)
        reloaded = server.Workbench(self.directory, deepcopy(self.pricing), MODEL)
        self.assertEqual(reloaded.state["comparison"], before["comparison"])

    def test_model_price_mismatch_is_rejected_before_spending(self):
        with self.assertRaisesRegex(ValueError, "pricing"):
            server.Workbench(self.directory / "mismatch", deepcopy(self.pricing), "other-model")
        self.work("generate")
        alternate_prices = deepcopy(self.pricing)
        alternate_prices["llm"]["model"] = "other-model"
        with self.assertRaisesRegex(ValueError, "Saved agent model differs"):
            server.Workbench(self.directory, alternate_prices, "other-model")
        self.generator.assert_called_once()
        self.agent_completion.assert_not_called()
        self.assert_no_judges()

    def test_skipped_judges_have_zero_spend_not_unknown_spend(self):
        """Skipped judges contribute zero cost while billed calls stay counted."""
        self.work("generate")
        self.work("run-agent")
        self.app.state["agent_run"]["rows"][0]["agent_metadata"]["error"] = "transport_error"
        self.work("compare")
        summary = self.app.state["comparison"]["summary"]
        self.assertIsNotNone(summary["llm"]["cost_usd"],
                             "A skipped judge made the cost of nine known calls unknown")
        self.assertAlmostEqual(summary["llm"]["cost_usd"], 9 * 0.00083)
        self.assertAlmostEqual(summary["jev"]["cost_usd"], 9 * 0.000075)
        for judge in self.app.state["comparison"]["rows"][0]["judges"].values():
            self.assertEqual(judge["usage"], {"input_tokens": 0, "output_tokens": 0})
            self.assertEqual(judge["cost_usd"], 0)

    def test_new_schema_failure_retains_the_billed_judge_usage_and_cost(self):
        """Rejecting a grade preserves metrics from the completed provider call."""
        def malformed(state, rubric, **kwargs):
            result = judge_result("llm", rubric)
            del result["scores"]["coverage"]
            return result

        self.judges["llm"].side_effect = malformed
        self.work("run-all")
        comparison = self.app.state["comparison"]
        self.assertEqual(comparison["summary"]["llm"]["n_failed"], 10)
        self.assertIsNotNone(comparison["summary"]["llm"]["cost_usd"],
                             "Schema validation discarded usage from billed calls")
        self.assertAlmostEqual(comparison["summary"]["llm"]["cost_usd"], 0.0083)
        self.assertEqual(comparison["summary"]["llm"]["input_tokens"], 2000)

    def test_judges_cannot_change_each_others_frozen_evidence(self):
        """Each judge gets independent copies of the original evidence and rubric."""
        self.mock_rich_agent()
        self.work("generate")
        self.work("run-agent")
        rows = deepcopy(self.app.state["agent_run"]["rows"])
        original_rubric = deepcopy(self.app.state["suite"]["rubric"])
        normal = self.judges["jev"].side_effect

        def mutating_judge(state, rubric, **kwargs):
            result = normal(state, rubric, **kwargs)
            state["agent_output"]["answer"] = "Mutated by first judge"
            state["tool_trace"][0]["usage"]["input_tokens"] = -1
            state["session_outputs"][0]["output"]["answer"] = "Mutated session output"
            state["independent_checks"][0]["passed"] = False
            state["final_tool_state"]["memory"]["preferred_channel"] = "mutated"
            rubric["dimensions"][0]["levels"]["1"] = "Mutated descriptor"
            return result

        self.judges["jev"].side_effect = mutating_judge
        self.work("compare")
        for index, row in enumerate(rows):
            expected = evidence_for(row, self.app.state["agent"]["knowledge"])
            for _, actual, _, _ in self.observed_judges[index * 2:index * 2 + 2]:
                self.assertEqual(actual, expected)
            self.assertEqual(
                evidence_for(self.app.state["comparison"]["rows"][index],
                             self.app.state["agent"]["knowledge"]), expected)
        self.assertEqual(self.app.state["agent_run"]["rows"], rows)
        self.assertEqual(self.app.state["suite"]["rubric"], original_rubric)
        for _, _, observed_rubric, _ in self.observed_judges:
            self.assertEqual(observed_rubric, original_rubric)

    def test_generation_rejects_wrong_difficulty_distribution(self):
        for case in self.payload["test_cases"]:
            case["metadata"]["difficulty"] = "easy"
        self.app._work("generate")
        self.assertEqual(self.app.state["job"]["status"], "failed")
        self.assertIsNone(self.app.state["suite"])

    def test_generation_rejects_wrong_dimension_count(self):
        """A rubric must satisfy the demo's three-dimension generation contract."""
        self.payload["rubric"]["dimensions"].pop()
        for dimension in self.payload["rubric"]["dimensions"]:
            dimension["weight"] = 0.5
        self.app._work("generate")
        self.assertEqual(self.app.state["job"]["status"], "failed")
        self.assertIsNone(self.app.state["suite"])

    def test_unexpected_worker_exception_does_not_persist_provider_secrets(self):
        """Neither actionable validation errors nor unexpected errors expose keys."""
        for error_type in (RuntimeError, ValueError):
            with self.subTest(error_type=error_type.__name__):
                self.generator.side_effect = error_type(
                    "Provider echoed offline-llm-key and offline-jev-key")
                self.app._work("generate")
                self.assertEqual(self.app.state["job"]["status"], "failed")
                for secret in ("offline-llm-key", "offline-jev-key"):
                    self.assertNotIn(secret, json.dumps(self.app.snapshot()))
                    self.assertNotIn(secret, (self.directory / "state.json").read_text())


if __name__ == "__main__":
    unittest.main()
