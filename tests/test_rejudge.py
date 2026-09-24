"""Saved-output integration tests. All Jev calls use a patched offline transport."""

import builtins
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts import rejudge


def rubric():
    return {
        "dimensions": [
            {"name": "quality", "scale": 5, "weight": 0.75,
             "levels": {str(i): f"Quality level {i}" for i in range(1, 6)}},
            {"name": "format", "scale": 3, "weight": 0.25,
             "levels": {"1": "Wrong shape", "2": "Partly correct", "3": "Correct shape"}},
        ],
        "pass_threshold": 0.7,
    }


def llm_judge(quality=4, form=3, model="saved-llm"):
    return {
        "scores": {"quality": quality, "format": form},
        "details": {
            name: {"reasoning": "Matches the level.", "evidence": ["Saved output."],
                   "suggestion": "Be clearer.", "confidence": "high"}
            for name in ("quality", "format")
        },
        "model_id": model,
        "resolved_model_id": model + "-pinned",
        "latency_ms": 42,
        "usage": {"input_tokens": 50, "output_tokens": 10},
    }


def row(case="one", subject="alpha", **updates):
    value = {
        "subject": subject, "case_id": case, "trial": 1,
        "input": {"question": "Answer the task."},
        "context": {"facts": [1, False, None]},
        "agent_output": {
            "recommendation": {"answer": "done"},
            "latency_ms": 987, "tool_calls": 2, "input_tokens": 111,
            "output_tokens": 22, "model_id": "agent-model", "error": None,
        },
        "expected_output": {"answer": "done"},
        "evidence": {"test_stdout": "passed", "path": "/do/not/open/evidence.txt"},
        "artifacts_dir": "/do/not/read/artifacts",
        "labels": {"private": "not judge input"},
        "reference_scores": {"quality": 4, "format": 3},
        "reference_metadata": {"graded_by": "human",
                               "graded_rubric_sha256": rejudge.rubric_sha256(rubric())},
        "gates": {"tests": True, "scope": True},
        "judge": llm_judge(),
        "passed": True,
    }
    value.update(updates)
    return value


def wire_response(payload, quality=5, form=3, confidence=0.4):
    """Build a deterministic valid wire response independently of replay code."""
    scores = {"quality": quality, "format": form}
    answers = {}
    for name, question in payload["questions"].items():
        criteria = question["criteria"]
        selected = scores[name] - 1
        answers[name] = {
            "type": "score", "score": selected, "confidence": confidence,
            "legend": {str(i): level for i, level in enumerate(criteria)},
            "probabilities": {str(i): float(i == selected) for i in range(len(criteria))},
        }
    return {"model": "jev-resolved-snapshot", "answers": answers,
            "usage": {"input_tokens": 80, "output_tokens": 12}}


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.input = self.directory / "saved.jsonl"
        self.rubric = self.directory / "rubric.json"
        self.rubric.write_text(json.dumps(rubric(), allow_nan=False), encoding="utf-8")
        self.output = self.directory / "replay"
        self.payloads = []
        # Fake credentials and an unconditional network-client guard apply to
        # every test. Only the explicitly patched Jev _post may be exercised.
        self.enterContext(patch.dict("os.environ", {"TYPESAFE_API_KEY": "offline-test-key"}))
        self.network = self.enterContext(patch(
            "urllib.request.OpenerDirector.open", side_effect=AssertionError("network forbidden")))
        self.agent = self.enterContext(patch(
            "scripts.codex_adapter.run", side_effect=AssertionError("agent invocation forbidden")))

    def transport(self, payload, key, timeout):
        self.assertEqual(key, "offline-test-key")
        self.payloads.append(deepcopy(payload))
        return 200, {}, json.dumps(wire_response(payload), allow_nan=False).encode()

    def write_rows(self, rows):
        self.input.write_text("".join(json.dumps(r, allow_nan=False) + "\n" for r in rows),
                              encoding="utf-8")

    def invoke(self, *flags):
        argv = ["--input", str(self.input), "--rubric", str(self.rubric),
                "--output", str(self.output), *flags]
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return rejudge.main(argv)

    def reports(self):
        results = [json.loads(line) for line in (self.output / "results.jsonl").read_text().splitlines()]
        summary = json.loads((self.output / "summary.json").read_text())
        markdown = (self.output / "report.md").read_text()
        return results, summary, markdown

    def live(self, rows, transport=None, *flags):
        self.write_rows(rows)
        with patch("scripts.jev_judge._post", side_effect=transport or self.transport) as post:
            self.assertEqual(self.invoke("--judge-backend", "jev",
                                         "--judge-model", "jev-requested-alias", *flags), 0)
        return self.reports(), post

    def test_two_subject_replay_preserves_outputs_and_reports_gate_failures(self):
        rows = [row(), row(subject="beta", gates={"tests": False})]
        before = deepcopy(rows)
        (results, summary, markdown), post = self.live(rows)
        self.assertEqual(post.call_count, 2)
        self.assertEqual(rows, before)
        self.assertEqual([json.loads(s) for s in self.input.read_text().splitlines()], before)
        self.assertEqual(summary["n_total"], 2)
        self.assertEqual(summary["n_scored"], 2)
        self.assertEqual(summary["n_passed"], 1)
        self.assertEqual(summary["n_failed"], 1)
        self.assertEqual(summary["pass_rate"], 0.5)
        self.assertEqual(summary["weighted_overall"], 1.0)
        self.assertEqual(summary["gate_counts"], {"passed": 1, "failed": 1})
        self.assertEqual([s["subject"] for s in summary["subjects"]], ["alpha", "beta"])
        self.assertEqual(summary["baseline_comparison"]["paired_n"], 2)
        self.assertEqual(summary["baseline_comparison"]["paired_rubric_provenance"], {"unverified": 2})
        self.assertEqual(summary["rubric_sha256"], rejudge.rubric_sha256(rubric()))
        self.assertEqual(summary["hash_method"]["algorithm"], "sha256")
        self.assertEqual(summary["review_threshold"], 0.5)
        self.assertAlmostEqual(summary["baseline_comparison"]["normalized_mae"], 0.15)
        self.assertEqual(summary["human_reference"]["paired_n"], 0)
        for original, result, record in zip(before, results, summary["records"]):
            for field in ("agent_output", "input", "context", "evidence", "gates",
                          "subject", "case_id", "trial", "reference_scores", "reference_metadata"):
                self.assertEqual(result[field], original[field])
            self.assertEqual(result["baseline_judge"], original["judge"])
            self.assertEqual(result["judge"]["backend"], "jev")
            self.assertEqual(result["judge"]["model_id"], "jev-requested-alias")
            self.assertEqual(result["judge"]["resolved_model_id"], "jev-resolved-snapshot")
            self.assertEqual(result["rejudge"]["rubric_sha256"], summary["rubric_sha256"])
            self.assertEqual(result["judge"]["rubric_sha256"], summary["rubric_sha256"])
            self.assertEqual(record["judge_identity"], {
                "backend": "jev", "requested_model_id": "jev-requested-alias",
                "resolved_model_id": "jev-resolved-snapshot"})
            self.assertEqual(record["baseline_identity"]["requested_model_id"], "saved-llm")
            self.assertEqual(record["usage"], {"input_tokens": 80, "output_tokens": 12})
            self.assertEqual(record["baseline_latency_ms"], 42)
            self.assertEqual(record["baseline_usage"], {"input_tokens": 50, "output_tokens": 10})
            self.assertGreaterEqual(record["latency_ms"], 0)
            self.assertEqual(record["confidence"], {"quality": 0.4, "format": 0.4})
            self.assertEqual(record["review_dimensions"], ["quality", "format"])
            self.assertEqual(record["review_threshold"], 0.5)
            self.assertTrue(all(v is None for detail in record["explanations"].values()
                                for v in detail.values()))
        self.assertFalse(results[1]["passed"])
        self.assertIn("Paired n: 2", markdown)
        self.assertIn("jev-requested-alias", markdown)
        self.assertIn("jev-resolved-snapshot", markdown)
        self.assertIn("Baseline latency ms: 42", markdown)
        self.assertIn("review threshold: 0.5", markdown)
        self.assertIn("| quality | unavailable | unavailable | unavailable |", markdown)
        self.assertNotIn("cost", summary)
        self.agent.assert_not_called()
        self.network.assert_not_called()

    def test_custom_review_threshold_and_unknown_baseline_usage_are_reported(self):
        saved = row()
        del saved["judge"]["latency_ms"]
        del saved["judge"]["usage"]
        (_, summary, markdown), _ = self.live([saved], None, "--review-threshold", "0.3")
        record = summary["records"][0]
        self.assertIsNone(record["baseline_latency_ms"])
        self.assertIsNone(record["baseline_usage"])
        self.assertEqual(summary["review_threshold"], 0.3)
        self.assertEqual(record["review_threshold"], 0.3)
        self.assertEqual(record["review_dimensions"], [])
        self.assertIn("Baseline latency ms: unavailable; baseline usage: unavailable", markdown)
        self.assertIn("review threshold: 0.3", markdown)

    def test_state_allowlist_preserves_types_and_does_not_open_evidence_paths(self):
        saved = row(input=False, context=0, expected_output=["expected", {"x": False}],
                    evidence={"path": "/etc/passwd", "inline": {"passed": True}})
        before = deepcopy(saved)
        with patch("pathlib.Path.open", side_effect=AssertionError("path read forbidden")), \
                patch("scripts.jev_judge._post", side_effect=self.transport):
            results = rejudge.replay_rows([saved], rubric(), judge_backend="jev",
                                          judge_model="jev-requested-alias")
        self.assertEqual(saved, before)
        state = self.payloads[0]["state"]
        self.assertEqual(state, {
            "input": False, "context": 0, "expected_output": ["expected", {"x": False}],
            "evidence": saved["evidence"], "agent_output": {"answer": "done"},
        })
        self.assertTrue({"reference_scores", "reference_metadata", "judge", "baseline_judge",
                         "labels", "artifacts_dir"}.isdisjoint(state))
        self.assertEqual(results[0]["agent_output"]["model_id"], "agent-model")

    def test_plain_scalar_object_and_empty_outputs_are_preserved(self):
        for output in ("", 0, False, [], {}, {"answer": [1, "a"]}):
            with self.subTest(output=output):
                saved = row(agent_output=output)
                del saved["context"]
                state = rejudge.build_state(saved)
                self.assertEqual(state["agent_output"], output)
                self.assertEqual(type(state["agent_output"]), type(output))
                self.assertNotIn("context", state)

    def test_missing_input_or_output_is_retained_without_a_judge_call(self):
        rows = []
        for index, field in enumerate(("input", "agent_output")):
            for missing in (True, False):
                saved = row(case=f"{field}-{missing}")
                if missing:
                    del saved[field]
                else:
                    saved[field] = None
                rows.append(saved)
        rows.append(row(case="empty-recommendation", agent_output={"recommendation": None, "error": None}))
        (results, summary, _), post = self.live(rows)
        post.assert_not_called()
        self.assertEqual(summary["n_total"], 5)
        self.assertEqual(summary["n_scored"], 0)
        self.assertEqual(summary["n_failed"], 5)
        self.assertEqual(summary["pass_rate"], 0)
        self.assertEqual(summary["status_counts"], {"invalid_record": 5})
        self.assertTrue(all(r["judge"] is None for r in results))

    def test_agent_failures_skip_judging_and_keep_original_error_metadata(self):
        wrapped = row(case="wrapped")
        wrapped["agent_output"].update(error="timeout", recommendation=None)
        rows = [wrapped, row(case="top", error="agent provider failed"),
                row(case="agent-error", agent_error="parse failed"),
                row(case="result", agent_result={"error": "failed"})]
        (results, summary, _), post = self.live(rows)
        post.assert_not_called()
        self.assertEqual(summary["status_counts"], {"agent_failed": 4})
        self.assertEqual(summary["n_total"], 4)
        self.assertEqual(summary["n_failed"], 4)
        self.assertEqual(results[0]["agent_output"], wrapped["agent_output"])
        self.assertEqual(results[1]["error"], rows[1]["error"])
        self.assertTrue(all(r["judge"] is None and not r["passed"] for r in results))

    def test_malformed_and_failed_judges_remain_in_pass_rate_denominator(self):
        rows = [row(case=name, subject=subject) for name, subject in
                (("pass", "alpha"), ("malformed", "alpha"), ("failure", "beta"), ("agent", "beta"))]
        rows[-1]["error"] = "agent failure"

        def transport(payload, *args):
            index = len(self.payloads)
            self.payloads.append(payload)
            if index == 1:
                return 200, {}, b'{"model": "wrong", "answers": {}}'
            if index == 2:
                return 403, {}, b'permission denied'
            return 200, {}, json.dumps(wire_response(payload)).encode()

        (results, summary, _), post = self.live(rows, transport)
        self.assertEqual(post.call_count, 3)
        self.assertEqual(summary["n_total"], 4)
        self.assertEqual(summary["n_scored"], 1)
        self.assertEqual(summary["n_passed"], 1)
        self.assertEqual(summary["n_failed"], 3)
        self.assertEqual(summary["pass_rate"], 0.25)
        self.assertEqual(summary["status_counts"], {"judged": 1, "judge_failed": 2, "agent_failed": 1})
        self.assertEqual(summary["baseline_comparison"]["paired_n"], 1)
        self.assertEqual(results[1]["judge"]["error"], "schema_failed")
        self.assertEqual(results[2]["judge"]["error"], "http_error")

    def test_unexpected_judge_exception_is_recorded_without_sensitive_exception_text(self):
        def transport(*args):
            raise RuntimeError("credential-that-must-not-be-logged")
        (results, summary, _), _ = self.live([row()], transport)
        self.assertEqual(results[0]["judge"]["error"], "judge_exception")
        self.assertEqual(summary["n_failed"], 1)
        self.assertNotIn("credential-that-must-not-be-logged", json.dumps(results))

    def test_no_judge_imports_no_provider_and_needs_neither_key_nor_model(self):
        self.write_rows([row(), row(case="gate", gates={"tests": False})])
        real_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if any(word in name for word in ("jev_judge", "anthropic", "urllib", "httpx", "codex_adapter")):
                raise AssertionError(f"provider or agent import: {name}")
            return real_import(name, *args, **kwargs)

        with patch.dict("os.environ", {}, clear=True), \
                patch("builtins.__import__", side_effect=guarded_import), \
                patch.object(rejudge, "_evaluate_jev") as evaluate:
            self.assertEqual(self.invoke("--no-judge", "--judge-backend", "jev"), 0)
        evaluate.assert_not_called()
        results, summary, markdown = self.reports()
        self.assertEqual(summary["n_total"], 2)
        self.assertEqual(summary["n_scored"], 0)
        self.assertEqual(summary["n_unassessed"], 1)
        self.assertEqual(summary["status_counts"], {"judge_skipped": 2})
        self.assertIsNone(summary["weighted_overall"])
        self.assertIsNone(summary["pass_rate"])
        self.assertEqual(summary["baseline_comparison"]["paired_n"], 0)
        self.assertTrue(all(r["judge"] is None and r["baseline_judge"] for r in results))
        self.assertIsNone(results[0]["passed"])
        self.assertFalse(results[1]["passed"])
        self.assertIn("unavailable", markdown)

    def test_default_llm_validates_saved_judge_and_does_not_relabel_model(self):
        saved = row()
        before = deepcopy(saved)
        with patch.object(rejudge, "_evaluate_jev") as evaluate:
            results = rejudge.replay_rows([saved], rubric(), judge_model="unrelated-cli-model")
        evaluate.assert_not_called()
        self.assertEqual(saved, before)
        self.assertEqual(results[0]["judge"]["backend"], "llm")
        self.assertEqual(results[0]["judge"]["model_id"], "saved-llm")
        self.assertEqual(results[0]["baseline_judge"], before["judge"])
        self.assertTrue(results[0]["passed"])
        comparison = results[0]["rejudge"]["baseline_comparison"]
        self.assertEqual(comparison["normalized_mae"], 0)
        self.assertEqual(comparison["current"]["requested_model_id"], "saved-llm")

    def test_llm_rejects_non_llm_missing_and_malformed_saved_judges(self):
        rows = [row(case="jev", judge={"backend": "jev", **llm_judge()}),
                row(case="other", judge={"backend": "other", **llm_judge()}),
                row(case="missing", judge=None), row(case="bad", judge={"scores": {}})]
        with patch.object(rejudge, "_evaluate_jev") as evaluate:
            results = rejudge.replay_rows(rows, rubric())
        evaluate.assert_not_called()
        self.assertEqual([r["judge"]["error"] for r in results],
                         ["backend_mismatch", "backend_mismatch", "judge_missing", "schema_failed"])
        self.assertTrue(all(not r["passed"] for r in results))

    def test_existing_baseline_is_preserved_and_comparison_is_strictly_row_local(self):
        rows = [
            row(case="same", subject="alpha", judge=llm_judge(5, 3),
                baseline_judge=llm_judge(1, 1, model="older-a")),
            row(case="same", subject="beta", judge=llm_judge(1, 1),
                baseline_judge=llm_judge(5, 3, model="older-b")),
            row(case="same", subject="alpha", trial=2, baseline_judge=None),
        ]
        before = deepcopy(rows)
        results = rejudge.replay_rows(rows, rubric())
        summary = rejudge.summarize(results, rubric())
        self.assertEqual(rows, before)
        self.assertEqual(results[0]["baseline_judge"], rows[0]["baseline_judge"])
        comparison = summary["baseline_comparison"]
        self.assertEqual(comparison["paired_n"], 2)
        self.assertEqual(comparison["unmatched_n"], 1)
        self.assertAlmostEqual(comparison["normalized_mae"], 0.75 * 4 / 5 + 0.25 * 2 / 3)
        self.assertEqual(comparison["signed_normalized_delta"], 0)
        self.assertEqual(len(comparison["by_model_pair"]), 2)
        self.assertEqual([p["paired_n"] for p in comparison["by_model_pair"]], [1, 1])

    def test_baseline_explicit_wrong_output_or_identity_is_unmatched(self):
        rows = [row(case="hash", baseline_judge_metadata={"graded_output_sha256": "wrong"}),
                row(case="subject", baseline_judge_metadata={"subject": "other"}),
                row(case="trial", baseline_judge={**llm_judge(), "trial": 999})]
        results = rejudge.replay_rows(rows, rubric())
        self.assertTrue(all(r["rejudge"]["baseline_comparison"]["status"] == "unmatched" for r in results))
        summary = rejudge.summarize(results, rubric())
        self.assertEqual(summary["baseline_comparison"]["paired_n"], 0)
        self.assertIsNone(summary["baseline_comparison"]["normalized_mae"])

    def test_human_metrics_require_actual_grader_and_exact_output_hash(self):
        matched = row(case="matched", agent_output={"answer": "שלום"}, judge=llm_judge(5, 2))
        raw = json.dumps(matched["agent_output"], ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
        output_hash = hashlib.sha256(raw).hexdigest()
        matched["reference_metadata"]["graded_output_sha256"] = output_hash
        cases = [matched]
        for case, metadata in (
            ("sketch", {"graded_by": "human", "graded_output_sha256": "expected-output-hash"}),
            ("assistant", {"graded_by": "codex", "graded_output_sha256": output_hash}),
            ("unknown", {"graded_output_sha256": output_hash}),
            ("missing-hash", {"graded_by": "human"}),
        ):
            cases.append({**deepcopy(matched), "case_id": case, "reference_metadata": metadata})
        results = rejudge.replay_rows(cases, rubric())
        summary = rejudge.summarize(results, rubric())
        reference = summary["human_reference"]
        self.assertEqual(reference["paired_n"], 1)
        self.assertEqual(reference["unmatched_n"], 4)
        self.assertAlmostEqual(reference["normalized_mae"], 0.75 / 5 + 0.25 / 3)
        self.assertAlmostEqual(reference["leniency"], 0.75 / 5 - 0.25 / 3)
        self.assertEqual(reference["status"], "available")
        self.assertEqual(reference["unavailable_reasons"], {
            "reference_output_unmatched": 2, "reference_not_human": 1, "grader_unavailable": 1})
        self.assertEqual(results[0]["rejudge"]["agent_output_sha256"], output_hash)
        self.assertEqual(rejudge.agent_output_sha256({"b": 1, "a": 2}),
                         rejudge.agent_output_sha256({"a": 2, "b": 1}))

    def test_reference_hash_covers_complete_metadata_wrapper_not_expected_output(self):
        saved = row()
        saved["reference_metadata"]["graded_output_sha256"] = rejudge.agent_output_sha256(
            saved["agent_output"]["recommendation"])
        result = rejudge.replay_rows([saved], rubric())[0]
        self.assertEqual(result["rejudge"]["human_reference"]["status"], "unmatched")
        saved["reference_metadata"]["graded_output_sha256"] = rejudge.agent_output_sha256(saved["agent_output"])
        result = rejudge.replay_rows([saved], rubric())[0]
        self.assertEqual(result["rejudge"]["human_reference"]["status"], "matched")

    def test_incomplete_or_invalid_human_scores_are_not_partially_averaged(self):
        for reference in ({}, {"quality": 4}, {"quality": True, "format": 3},
                          {"quality": 6, "format": 3}, {"quality": 4, "format": 3, "extra": 5}):
            with self.subTest(reference=reference):
                saved = row(reference_scores=reference)
                saved["reference_metadata"]["graded_output_sha256"] = rejudge.agent_output_sha256(saved["agent_output"])
                result = rejudge.replay_rows([saved], rubric())[0]
                self.assertEqual(result["rejudge"]["human_reference"]["reason"], "reference_scores_invalid")

    def test_rubric_hash_normalizes_yaml_level_keys_and_detects_changed_grading(self):
        original = rubric()
        yaml_form = deepcopy(original)
        for dimension in yaml_form["dimensions"]:
            dimension["levels"] = {int(k): v for k, v in dimension["levels"].items()}
        original_hash = rejudge.rubric_sha256(original)
        self.assertEqual(rejudge.rubric_sha256(yaml_form), original_hash)
        saved = row()
        saved["reference_metadata"]["graded_output_sha256"] = rejudge.agent_output_sha256(saved["agent_output"])
        for change in ("descriptors", "weights", "threshold"):
            modified = deepcopy(original)
            if change == "descriptors":
                modified["dimensions"][0]["levels"]["5"] = "A materially different criterion"
            elif change == "weights":
                modified["dimensions"][0]["weight"] = 0.5
                modified["dimensions"][1]["weight"] = 0.5
            else:
                modified["pass_threshold"] = 0.9
            with self.subTest(change=change):
                self.assertNotEqual(rejudge.rubric_sha256(modified), original_hash)
                result = rejudge.replay_rows([saved], modified)[0]
                reference = result["rejudge"]["human_reference"]
                self.assertEqual(reference["status"], "unmatched")
                self.assertEqual(reference["reason"], "reference_rubric_unmatched")
        del saved["reference_metadata"]["graded_rubric_sha256"]
        result = rejudge.replay_rows([saved], original)[0]
        self.assertEqual(result["rejudge"]["human_reference"]["reason"], "reference_rubric_unmatched")

    def test_baseline_rubric_provenance_verified_unverified_and_mismatched(self):
        rubric_hash = rejudge.rubric_sha256(rubric())
        rows = [
            row(case="legacy"),
            row(case="metadata", baseline_judge_metadata={"graded_rubric_sha256": rubric_hash}),
            row(case="judge", baseline_judge={**llm_judge(), "rubric_sha256": rubric_hash}),
            row(case="wrong", baseline_judge_metadata={"graded_rubric_sha256": "old-rubric"}),
            row(case="wrong-judge", baseline_judge={**llm_judge(), "rubric_sha256": "old-rubric"}),
        ]
        summary = rejudge.summarize(rejudge.replay_rows(rows, rubric()), rubric())
        comparison = summary["baseline_comparison"]
        self.assertEqual(comparison["paired_n"], 3)
        self.assertEqual(comparison["paired_rubric_provenance"], {"unverified": 1, "verified": 2})
        self.assertEqual(comparison["unavailable_reasons"], {"baseline_rubric_mismatch": 2})
        self.assertIn("rubric provenance is unverified", rejudge.render_markdown(summary))

    def test_invalid_jev_response_preserves_models_and_transport_metadata(self):
        # The LLM-shaped details deliberately violate the Jev detail contract.
        candidate = {"backend": "jev", **llm_judge(model="original-request")}
        with patch.object(rejudge, "_evaluate_jev", return_value=candidate):
            results = rejudge.replay_rows([row()], rubric(), judge_backend="jev", judge_model="fallback-model")
        judge = results[0]["judge"]
        self.assertEqual(judge["error"], "schema_failed")
        self.assertEqual(judge["backend"], "jev")
        for key in ("model_id", "resolved_model_id", "latency_ms", "usage"):
            self.assertEqual(judge[key], candidate[key])
        summary = rejudge.summarize(results, rubric(), judge_backend="jev", judge_model="fallback-model")
        self.assertEqual(summary["records"][0]["judge_identity"]["requested_model_id"], "original-request")
        self.assertEqual(summary["records"][0]["judge_identity"]["resolved_model_id"], "original-request-pinned")
        checked = rejudge._checked_judge({"backend": "jev", "scores": {}}, rubric(), "jev",
                                         expected_model="fallback-model")
        self.assertEqual(checked["model_id"], "fallback-model")
        self.assertNotIn("resolved_model_id", checked)

    def test_gates_require_boolean_mapping_and_cannot_be_overridden_by_perfect_judge(self):
        gates = [None, [], True, {"tests": 1}, {"tests": "true"}, {"tests": None},
                 {"": True}, {"tests": False}, {}, {"tests": True}]
        rows = [row(case=str(i), gates=g, judge=llm_judge(5, 3)) for i, g in enumerate(gates)]
        absent = row(case="absent", judge=llm_judge(5, 3))
        del absent["gates"]
        rows.append(absent)
        results = rejudge.replay_rows(rows, rubric())
        self.assertEqual([r["passed"] for r in results], [False] * 8 + [True] * 3)
        self.assertEqual([r["rejudge"]["gate_status"] for r in results],
                         ["malformed"] * 7 + ["failed", "not_applicable", "passed", "not_applicable"])

    def test_duplicate_identity_rejects_entire_run_before_judge_or_output_creation(self):
        self.write_rows([row(), row(agent_output="different saved output")])
        with patch.object(rejudge, "_evaluate_jev") as evaluate:
            self.assertEqual(self.invoke("--judge-backend", "jev", "--judge-model", "jev"), 2)
        evaluate.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_live_jev_requires_model_but_no_judge_does_not(self):
        self.write_rows([row()])
        with self.assertRaises(SystemExit) as exc:
            self.invoke("--judge-backend", "jev")
        self.assertEqual(exc.exception.code, 2)
        self.assertFalse(self.output.exists())

    def test_missing_api_key_is_reported_as_failure_with_no_network(self):
        self.write_rows([row()])
        with patch.dict("os.environ", {}, clear=True), patch("scripts.jev_judge._post") as post:
            self.assertEqual(self.invoke("--judge-backend", "jev", "--judge-model", "jev"), 0)
        post.assert_not_called()
        results, summary, _ = self.reports()
        self.assertEqual(results[0]["judge"]["error"], "configuration_error")
        self.assertEqual(summary["pass_rate"], 0)

    def test_existing_output_directory_or_symlink_is_never_overwritten(self):
        self.write_rows([row()])
        self.output.mkdir()
        sentinel = self.output / "summary.json"
        sentinel.write_text("keep me")
        with patch.object(rejudge, "_evaluate_jev") as evaluate:
            self.assertEqual(self.invoke("--judge-backend", "jev", "--judge-model", "jev"), 2)
        evaluate.assert_not_called()
        self.assertEqual(sentinel.read_text(), "keep me")
        link = self.directory / "link"
        link.symlink_to(self.output, target_is_directory=True)
        self.output = link
        self.assertEqual(self.invoke("--no-judge"), 2)
        self.assertEqual(sentinel.read_text(), "keep me")

    def test_malformed_jsonl_lines_and_nonobjects_are_retained_in_reports(self):
        self.input.write_text(
            json.dumps(row()) + '\nnot JSON\n[]\n\n{"case_id":"x","input":NaN}\n'
            '{"case_id":"x","input":1e999}\n{"case_id":"x","case_id":"y"}\n',
            encoding="utf-8")
        self.assertEqual(self.invoke("--no-judge"), 0)
        results, summary, _ = self.reports()
        self.assertEqual(len(results), 7)
        self.assertEqual(summary["n_total"], 7)
        self.assertEqual(summary["status_counts"], {"judge_skipped": 1, "invalid_record": 6})
        self.assertEqual(results[1]["source_raw_line"], "not JSON")
        self.assertEqual(results[2]["source_row"], [])
        self.assertEqual(results[3]["source_raw_line"], "")
        self.assertEqual([r["rejudge"]["source_line"] for r in results], list(range(1, 8)))
        json.dumps(summary, allow_nan=False)
        json.dumps(results, allow_nan=False)

    def test_malformed_subject_stays_visible_and_unscored(self):
        self.write_rows([row(subject={"bad": "shape"})])
        self.assertEqual(self.invoke(), 0)
        results, summary, _ = self.reports()
        self.assertEqual(results[0]["subject"], {"bad": "shape"})
        self.assertEqual(summary["n_total"], 1)
        self.assertEqual(summary["n_failed"], 1)
        self.assertEqual(summary["status_counts"], {"invalid_record": 1})

    def test_invalid_rubric_rejected_before_calls_or_creating_output(self):
        self.write_rows([row()])
        invalid = rubric()
        invalid["dimensions"][0]["weight"] = 2
        self.rubric.write_text(json.dumps(invalid))
        with patch.object(rejudge, "_evaluate_jev") as evaluate:
            self.assertEqual(self.invoke("--judge-backend", "jev", "--judge-model", "jev"), 2)
        evaluate.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_yaml_rubric_when_optional_pyyaml_is_available(self):
        try:
            import yaml
        except ImportError:
            self.skipTest("optional PyYAML not installed")
        self.write_rows([row()])
        self.rubric = self.directory / "rubric.yaml"
        self.rubric.write_text(yaml.safe_dump(rubric()))
        self.assertEqual(self.invoke("--no-judge"), 0)
        self.assertEqual(self.reports()[1]["n_total"], 1)

    def test_empty_input_reports_unavailable_statistics_without_nonfinite_json(self):
        self.write_rows([])
        self.assertEqual(self.invoke(), 0)
        results, summary, _ = self.reports()
        self.assertEqual(results, [])
        self.assertEqual(summary["n_total"], 0)
        self.assertIsNone(summary["weighted_overall"])
        self.assertIsNone(summary["baseline_comparison"]["normalized_mae"])
        self.assertIsNone(summary["human_reference"]["leniency"])
        self.assertIsNone(summary["pass_rate"])

    def test_direct_script_cli_works_offline(self):
        self.write_rows([row()])
        command = [sys.executable, str(Path(rejudge.__file__)), "--input", str(self.input),
                   "--rubric", str(self.rubric), "--output", str(self.output),
                   "--judge-backend", "jev", "--no-judge"]
        completed = subprocess.run(command, capture_output=True, text=True, timeout=15,
                                   env={"PATH": "/usr/bin:/bin"})
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(self.reports()[1]["mode"], "no-judge")

    def test_nonfinite_provider_metadata_fails_closed(self):
        candidate = {"backend": "jev", **llm_judge(), "usage": {"input_tokens": float("nan")}}
        with patch.object(rejudge, "_evaluate_jev", return_value=candidate):
            results = rejudge.replay_rows([row()], rubric(), judge_backend="jev", judge_model="jev")
        self.assertEqual(results[0]["judge"]["error"], "schema_failed")
        self.assertFalse(results[0]["passed"])
        json.dumps(results, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
