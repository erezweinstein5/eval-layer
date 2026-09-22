"""No network or model calls: exercise the adapter through a mock executable."""

import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

from scripts.codex_adapter import run


class CodexAdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repo = self.root / "repo"
        self.repo.mkdir()
        subprocess.run(["git", "init", "-q", str(self.repo)], check=True)
        subprocess.run([
            "git", "-C", str(self.repo), "-c", "user.name=Test",
            "-c", "user.email=test@example.invalid", "commit", "-q",
            "--allow-empty", "-m", "fixture",
        ], check=True)
        self.mock = self.root / "mock-codex"
        self.artifacts = self.root / "evidence"

    def invoke(self, events, *, final="Done.", exit_code=0, timeout=False,
               raw=None, write_final=True, edits=None):
        payload = raw if raw is not None else "".join(
            json.dumps(event) + "\n" for event in events
        )
        self.mock.write_text(
            f"#!{sys.executable}\n"
            "import json, pathlib, sys, time\n"
            "args = sys.argv[1:]\n"
            "assert args[:6] == ['exec', '--json', '--sandbox', "
            "'workspace-write', '--model', 'test-model']\n"
            "assert args[-1] == '-'\n"
            "assert sys.stdin.read() == 'full prompt\\nwith unicode: λ'\n"
            "assert pathlib.Path('.git').is_dir()\n"
            + "".join(f"pathlib.Path({name!r}).write_text({text!r})\n"
                      for name, text in (edits or {}).items())
            +
            f"sys.stdout.write({payload!r}); sys.stdout.flush()\n"
            "sys.stderr.write('mock diagnostic\\n')\n"
            + ("time.sleep(10)\n" if timeout else "")
            + (f"pathlib.Path(args[args.index('-o') + 1]).write_text({final!r})\n"
               if write_final else "")
            + f"sys.exit({exit_code})\n",
            encoding="utf-8",
        )
        self.mock.chmod(0o755)
        result = run(
            "full prompt\nwith unicode: λ", workspace=self.repo,
            model_id="test-model", artifacts_dir=self.artifacts,
            executable=self.mock, timeout_s=2 if timeout else 5,
        )
        self.assertEqual(set(result), {
            "recommendation", "latency_ms", "tool_calls", "input_tokens",
            "output_tokens", "model_id", "error",
        })
        self.assertGreaterEqual(result["latency_ms"], 0)
        return result

    @staticmethod
    def completion(input_tokens=10, output_tokens=2):
        return {"type": "turn.completed", "usage": {
            "input_tokens": input_tokens, "output_tokens": output_tokens,
            "cached_input_tokens": 8,
        }}

    def test_counts_usage_and_preserves_final_output(self):
        events = [{"type": "turn.started"}]
        for item_id, kind in enumerate((
            "command_execution", "mcp_tool_call", "web_search", "file_change",
        )):
            for state in ("started", "updated", "completed"):
                events.append({"type": f"item.{state}", "item": {
                    "id": str(item_id), "type": kind, "changes": [{}, {}],
                }})
        events += [self.completion(), {"type": "turn.started"}, self.completion(5, 3)]
        result = self.invoke(events, final="Summary\n")
        self.assertIsNone(result["error"])
        self.assertEqual(result["recommendation"], {"final_message": "Summary\n"})
        self.assertEqual((result["input_tokens"], result["output_tokens"]), (15, 5))
        self.assertEqual(result["tool_calls"], 3)
        self.assertEqual(result["model_id"], "test-model")
        self.assertTrue((self.artifacts / "invocation.json").is_file())
        self.assertIn("mock diagnostic", (self.artifacts / "stderr.txt").read_text())

    def test_zero_usage_and_no_tools_are_valid(self):
        result = self.invoke([self.completion(0, 0)])
        self.assertIsNone(result["error"])
        self.assertEqual(result["input_tokens"], 0)
        self.assertEqual(result["tool_calls"], 0)

    def test_missing_usage_is_unknown(self):
        result = self.invoke([self.completion(), {"type": "turn.completed"}])
        self.assertIsNone(result["error"])
        self.assertIsNone(result["input_tokens"])
        self.assertIsNone(result["output_tokens"])

    def test_failure_streams(self):
        cases = [
            ([{"type": "turn.started"}], "no completion"),
            ([self.completion(), {"type": "turn.started"}], "no completion"),
            ([{"type": "turn.failed", "error": {"message": "failed"}}], "turn.failed"),
            ([{"type": "error", "message": "failed"}, self.completion()], "error:"),
            ([{"type": "item.started", "item": {"type": "web_search"}}], "missing id"),
            ([{"type": "turn.completed", "usage": {"input_tokens": "10"}}], "invalid"),
            ([{"type": "turn.completed", "usage": []}], "usage must"),
        ]
        for index, (events, error) in enumerate(cases):
            with self.subTest(error=error):
                self.artifacts = self.root / f"case-{index}"
                result = self.invoke(events)
                self.assertIn(error, result["error"])
                self.assertIsNone(result["recommendation"])

    def test_malformed_json_even_after_completion(self):
        result = self.invoke([], raw=json.dumps(self.completion()) + "\nnot json\n")
        self.assertIn("malformed stream line 2", result["error"])
        self.assertIsNone(result["recommendation"])

    def test_nonzero_exit(self):
        result = self.invoke([self.completion()], exit_code=9)
        self.assertIn("nonzero exit 9", result["error"])
        self.assertIsNone(result["recommendation"])

    def test_timeout_preserves_partial_usage(self):
        result = self.invoke([self.completion()], timeout=True)
        self.assertIn("timeout", result["error"])
        self.assertEqual(result["input_tokens"], 10)
        self.assertIsNone(result["recommendation"])

    def test_missing_final_file(self):
        result = self.invoke([self.completion()], write_final=False)
        self.assertIn("FileNotFoundError", result["error"])

    def test_empty_final_file(self):
        self.assertIn("missing final output", self.invoke(
            [self.completion()], final=" \n",
        )["error"])

    def test_launch_failure(self):
        result = run("x", workspace=self.repo, model_id="test-model",
                     artifacts_dir=self.artifacts, executable=self.root / "absent")
        self.assertIn("FileNotFoundError", result["error"])

    def test_reject_stale_artifacts_and_implicit_model(self):
        self.artifacts.mkdir()
        result = run("x", workspace=self.repo, model_id="test-model",
                     artifacts_dir=self.artifacts)
        self.assertIn("FileExistsError", result["error"])
        result = run("x", workspace=self.repo, model_id="",
                     artifacts_dir=self.artifacts)
        self.assertIn("model_id must", result["error"])

    def test_documented_fixture_patch_and_external_test_recipe(self):
        doc = (Path(__file__).resolve().parents[1] / "references" / "codex.md").read_text()
        helpers = next(block for block in re.findall(r"```python\n(.*?)```", doc, re.S)
                       if "def capture_patch" in block)
        namespace = {}
        exec(compile(helpers, "codex.md", "exec"), namespace)
        (self.repo / "answer.py").write_text("answer = 0\n")
        subprocess.run(["git", "add", "answer.py"], cwd=self.repo, check=True)
        subprocess.run([
            "git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
            "commit", "-qm", "bug",
        ], cwd=self.repo, check=True)
        baseline_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=self.repo, text=True,
        ).strip()
        external = self.root / "external_test.py"
        external.write_text(
            "import pathlib, sys\n"
            "scope = {}\n"
            "exec(pathlib.Path(sys.argv[1], 'answer.py').read_text(), scope)\n"
            "assert scope['answer'] == 42, 'wrong answer'\n"
        )
        argv = [sys.executable, str(external), str(self.repo)]
        baseline_evidence = self.root / "baseline-evidence"
        baseline_evidence.mkdir()
        before = namespace["external_test"](argv, self.repo, baseline_evidence)
        self.assertEqual(before["exit_code"], 1)
        self.assertIn("wrong answer", (baseline_evidence / "test.stderr").read_text())
        result = self.invoke([self.completion()], edits={
            "answer.py": "answer = 42\n", "new file.txt": "new deliverable\n",
        })
        self.assertIsNone(result["error"])
        namespace["capture_patch"](self.repo, self.artifacts, baseline_sha)
        patch = (self.artifacts / "changes.patch").read_text()
        self.assertIn("+answer = 42", patch)
        self.assertIn("+new deliverable", patch)
        self.assertIn(b"new file.txt\0", (self.artifacts / "untracked.z").read_bytes())
        after = namespace["external_test"](argv, self.repo, self.artifacts)
        self.assertEqual(after["exit_code"], 0)


if __name__ == "__main__":
    unittest.main()
