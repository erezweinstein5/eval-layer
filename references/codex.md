# Codex CLI adapter

Use `scripts/codex_adapter.py` for a Codex coding subject. It uses only Python's
stdlib and returns the existing seven metadata fields. The harness owns fixture
creation, external tests, patch evidence, pass gates, and optional judging.
See `references/cross-subject-benchmarking.md` for subject dispatch.

## Invocation and output

The adapter executes this argument vector in the case's repository:

```sh
codex exec --json --sandbox workspace-write --model "$MODEL_ID" \
  -o "$FINAL_OUTPUT" - < "$PROMPT_FILE"
```

Set all three variables explicitly for a manual run. `-` makes stdin the full
prompt. `--json` emits JSONL events; `-o` saves the final message separately.
These flags and the optional `--output-schema` are documented in official
sources [1–2]. The adapter passes arguments without a shell.

Copy `scripts/codex_adapter.py` next to the generated `eval_harness.py`, then:

```python
import json
import os
from pathlib import Path
from codex_adapter import run

# Supply an existing fresh committed fixture and a NEW evidence directory.
workspace = Path(os.environ["CASE_WORKSPACE"]).resolve()
evidence = Path(os.environ["CASE_EVIDENCE"]).resolve()
result = run(
    Path(os.environ["PROMPT_FILE"]).read_text(encoding="utf-8"),
    workspace=workspace,
    model_id=os.environ["MODEL_ID"],
    artifacts_dir=evidence,
    timeout_s=300,
)
print(json.dumps(result, ensure_ascii=False))
```

The evidence directory must be outside the workspace and must not exist yet.
An existing directory is an error, preventing stale final output from making a
failed trial look successful. Preserve each directory under its subject, case,
and trial identifiers. No session is resumed.

| Field | Adapter meaning |
|---|---|
| `recommendation` | `{"final_message": "<raw final text>"}` on success; `None` on error |
| `latency_ms` | Wall time of invocation and output collection; excludes external tests and judging |
| `tool_calls` | Unique observed call item IDs in this invocation |
| `input_tokens` | Sum of `usage.input_tokens` from `turn.completed` |
| `output_tokens` | Sum of `usage.output_tokens` from `turn.completed` |
| `model_id` | Required explicit requested model; not a claim about a server-resolved alias |
| `error` | Failure description or `None` |

The stream documents `turn.completed` usage and item lifecycle events [1].
This adapter counts `command_execution`, `mcp_tool_call`, and `web_search` IDs
once across `item.started`, `item.updated`, and `item.completed`. A failed
tool invocation still counts. `file_change` items and their change arrays do
not count as calls. One shell invocation may execute several commands; this
metric counts observed call items, not subprocesses or files.

Input already includes cached input: do not add `cached_input_tokens` to it.
Likewise, do not add reasoning tokens to output totals. Retain those breakdowns
in raw evidence if needed. Missing usage stays `None`; a reported zero remains
zero. Counts observed before a failure remain available, but are partial.

Keep `recommendation` a JSON object even when the final response is plain text.
An optional schema can request an object with a string `final_message` [1–2].
If adding that option to the adapter, parse the final file and validate its
required keys/types before assigning `recommendation`. Do not require a second
model call to turn a coding summary into JSON. The supplied adapter simply
wraps the raw text, so it needs no schema library.

## Coding case format

Extend the shared seed format with harness-owned fixture and check settings:

```yaml
test_cases:
  - id: easy-01
    input: "Fix discount_total so an empty cart returns zero. Preserve the public API."
    expected_output: "An empty cart returns zero and existing discount behavior is preserved."
    metadata:
      difficulty: easy
      category: bug-fix
    fixture:
      repository: /absolute/path/to/fixture-repository
      commit: "<full starting commit SHA>"
    allowed_paths: ["src/cart.py"]
    checks:
      - id: empty-cart-regression
        argv: ["python3", "/absolute/path/to/trusted-checks/check_empty_cart.py"]
        timeout_s: 60
        required: true
        baseline_expectation: fails_with_target_assertion
      - id: existing-behavior
        argv: ["python3", "/absolute/path/to/trusted-checks/check_cart_behavior.py"]
        timeout_s: 60
        required: true
        baseline_expectation: passes
```

Replace the fixture SHA and paths with real values when generating a harness.
Each trusted check executes or imports the candidate from its working directory.
Run each check into a separate evidence directory so logs are not overwritten.
Record the intended baseline assertion explicitly in the generated case. Store
reference scores and their grader metadata alongside these fields only when
available; an expected-output sketch is not proof that the candidate passed.

## Fresh fixture for every subject × case × trial

1. Freeze each case's starting commit, full prompt, expected failure, external
   test command, dependency versions, allowed paths, and acceptance conditions.
2. Create a new clone or independent repository from that commit for **every**
   subject/case/trial. Commit any case setup before invocation. Save its baseline
   SHA outside the fixture. Start clean, including no leftover untracked files.
3. Verify the external regression test fails for the intended bug on a separate
   baseline copy. A missing dependency, collection error, or timeout is a fixture
   failure, not the expected bug. Record the failing test identity and diagnostic.
   For non-bug tasks, define an equivalent unmet acceptance condition.
4. Run the subject once in the untouched trial fixture. Give every comparable
   subject the same prompt, timeout, tools, dependencies, and starting contents.
5. Capture the resulting files and patch, then run the trusted external tests
   against those files. Preserve all evidence before disposing of the fixture.

Do not reset and reuse a dirty working directory between trials. Avoid sharing
mutable build outputs or caches that can carry a prior solution. A new Codex
process does not itself provide a new repository. Codex's normal Git repository
check is documented in [1]; this recipe keeps it enabled.

External tests should live outside the editable fixture and import or execute
the candidate code. Keep their commands and expected results under harness
control. Agent-authored tests can supplement them. A final message saying
"tests passed" cannot satisfy a test gate.

## Patch and test evidence

The following stdlib helper runs in the harness after the adapter. It captures
staged and unstaged edits relative to the saved baseline, plus ordinary untracked
files by adding intent-to-add entries **only in the disposable fixture**.

```python
import json
import subprocess
from pathlib import Path

def capture_patch(workspace, evidence, baseline_sha):
    workspace, evidence = Path(workspace), Path(evidence)

    def git(*args):
        return subprocess.check_output(["git", *args], cwd=workspace)

    # Save original status before the intent-to-add operation changes the index.
    (evidence / "status.z").write_bytes(
        git("status", "--porcelain=v1", "-z", "--untracked-files=all"))
    untracked = git("ls-files", "--others", "--exclude-standard", "-z")
    (evidence / "untracked.z").write_bytes(untracked)
    for name in untracked.split(b"\0"):
        if name:
            subprocess.run(
                ["git", "add", "-N", "--", name.decode("utf-8", "surrogateescape")],
                cwd=workspace, check=True,
            )
    (evidence / "changes.patch").write_bytes(
        git("diff", "--binary", "--no-ext-diff", baseline_sha, "--"))
    (evidence / "changed-paths.z").write_bytes(
        git("diff", "--name-only", "-z", baseline_sha, "--"))
    (evidence / "ignored.z").write_bytes(
        git("ls-files", "--others", "--ignored", "--exclude-standard", "-z"))

def external_test(argv, workspace, evidence, timeout_s=120):
    evidence = Path(evidence)
    with (evidence / "test.stdout").open("wb") as out, \
         (evidence / "test.stderr").open("wb") as err:
        try:
            proc = subprocess.run(
                argv, cwd=workspace, stdout=out, stderr=err, timeout=timeout_s,
            )
            outcome = {"argv": argv, "exit_code": proc.returncode, "timeout": False}
        except subprocess.TimeoutExpired:
            outcome = {"argv": argv, "exit_code": None, "timeout": True}
    (evidence / "tests.json").write_text(json.dumps(outcome), encoding="utf-8")
    return outcome
```

Supply the baseline SHA captured **before** Codex runs, not the possibly changed
post-run `HEAD`. Capture before testing so test-generated files do not obscure
the candidate patch. Include ignored candidate deliverables explicitly according
to the case policy; the ignored manifest prevents silently overlooking them.
For fixtures with submodules, capture their changes separately or disallow edits.
Archive the final candidate tree if the fixture will be deleted.

Use a process/container supervisor for test suites that spawn persistent children;
the small test helper times out the direct test process. The adapter kills the
process group on POSIX timeouts; Windows requires external supervision for child
process cleanup. Neither helper is a replacement for an isolated test worker.

## Gates, configuration, and reporting

Require all applicable hard gates before marking a trial passed:

- Valid baseline: the intended bug/acceptance failure is demonstrated.
- Invocation success: no adapter error, a completion event, and nonempty final output.
- External regression and required existing tests pass; timeout/collection failure fails.
- Patch stays within allowed paths and meets required artifact/behavior checks.
- Tests and protected fixture/configuration files have not been weakened or removed.

A weighted judge score cannot override a failed gate. Keep fixture failures,
agent failures, test failures, and judge failures distinguishable; never drop
failed trials from the denominator. A zero tool count is valid when a case does
not need tools.

Store the seven-field result unchanged inside a harness row alongside
`subject`, `case_id`, `trial`, `artifacts_dir`, `gates`, `judge`, and `judge_status`.
The adapter writes `prompt.txt`, `invocation.json`, `events.jsonl`, `stderr.txt`,
`process.json`, and, when produced, `final.txt`. The harness adds baseline SHA,
patch/status manifests, test evidence, and configuration evidence. Do not put
paths, diffs, or test outcomes inside `recommendation` as extra metadata fields.

Record the CLI version, executable identity, requested model, provider, reasoning
effort, sandbox/approval policy, tool and MCP availability, network/search mode,
dependency/runtime versions, and timeout. Record relevant instruction files,
skills, config layers, and prompt hashes/content without copying credentials.
The CLI inherits configuration and supports per-run overrides [2]; flags alone
do not prove the entire effective configuration. Label unavailable settings
unknown. Hold settings constant unless they define the subject under comparison.

Keep generated harness flags `--framework`, `--test-case`, `-v`/`--verbose`, and
`--trials`. Add `--no-judge` to skip judge calls while still running gates and
writing evidence. Report `judge: null`, `judge_status: "skipped"`, and null judge scores, not
invented grades. Judging is optional for deterministic coding acceptance; enable
it only for rubric dimensions that need qualitative assessment.

## Local validation without paid calls

From this skill directory:

```sh
python3 -m unittest discover -s tests -p 'test_codex_adapter.py' -v
```

The tests use a mock executable and fresh committed Git fixtures. They cover the
argument vector, full stdin prompt, final file, event deduplication, usage totals,
missing usage, zero counts, timeout, nonzero exit, malformed stream, failed turns,
missing completion/output, launch failure, and stale artifact rejection.
Live CLI/model compatibility is not established by these mock tests.

## Official sources

Verified 2026-09-22:

1. [OpenAI: Non-interactive mode](https://developers.openai.com/codex/noninteractive/)
   — JSONL lifecycle, usage example, final output, schema, stdin, Git requirement.
2. [OpenAI: CLI reference](https://developers.openai.com/codex/cli/reference/)
   — `exec` flags, explicit model/sandbox, configuration overrides.
