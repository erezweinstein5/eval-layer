# Optional results UI: customer-support example

The core product is the skill-generated eval layer: context record, rubric, cases/checks, CLI harness, and Jev/LLM results. This workbench is an optional example and viewer. Its fixed support scenarios are not the skill's generic generation logic.

This demo starts with the [Harbor support agent](agent.py) and its
[support policy](knowledge.md). The agent uses native LLM tool calls, sandbox
support tools, and a durable memory store across multiple conversation turns.
Each evaluation case has isolated tool data and memory. The server generates a
rubric and customer wording for exactly 10 authored
[scenario contracts](scenarios.py), using the agent source and policy. It binds
the expected behavior from those fixture-backed contracts, runs the actual
agent, and compares Jev with a live LLM judge on the same recorded evidence.
Tool data is synthetic and local; LLM calls and sandbox tool executions are
real. The model chooses the tool calls, and the recorded answers are not
prewritten.
The [UI](ui/index.html) is a single HTML file with no external
assets or JavaScript dependencies; the Python server uses the standard library.

## Setup

Run from the repository root with Python 3.10 or newer. Load these credentials
into the server process's environment through your existing credential setup:

| Variable | Purpose |
|---|---|
| `OPENAI_API_KEY` | Existing bearer credential for the AWS Bedrock Mantle endpoint below; used for generation, the agent, and the LLM judge |
| `TYPESAFE_API_KEY` | TypeSafe credential for the Jev judge |

Keep credentials in the environment, outside the source, agent instructions,
and support knowledge. The setup below reuses `OPENAI_API_KEY`; it does not
replace it or require an OpenAI-hosted endpoint.

```bash
export OPENAI_BASE_URL='https://bedrock-mantle.us-east-2.api.aws/v1'
export DEMO_LLM_MODEL='openai.gpt-oss-120b'
python3 demo/server.py --port 8771
```

Open `http://127.0.0.1:8771`. The server binds to localhost. Its default port is
`8766` when `--port` is omitted. The default model is `openai.gpt-oss-120b`;
`DEMO_LLM_MODEL` or `--model` can select another model. Set `OPENAI_BASE_URL`
explicitly for this AWS setup.

The server loads [pricing.json](pricing.json), which records the verified
list-price rates, sources, model IDs, and verification date. If you change the
model or pricing assumptions, update that file accordingly. An existing data
directory restores the saved agent configuration, including its model; use a
fresh `--data-dir` when testing a different model.

Opening the workbench does not call a model. Starting generation, an agent run,
or a judge comparison makes live provider requests and incurs API costs.

## Workflow

1. **Agent:** inspect or edit the name, system instructions, and support
   knowledge, then select **Save agent**. Saving clears the current suite and
   outputs; previously completed comparisons remain in history.
2. **Eval set:** select **Generate eval set**. The server sends the full
   [SKILL.md](../SKILL.md), [rubric design reference](../references/rubric-design.md),
   agent source, instructions, policy, and the 10 scenario contracts to the
   generator. The skill generates the rubric and customer wording, one case
   per contract. After generation, the server replaces proposed reference
   answers with the independently authored contract expectations and binds
   their checks, follow-up inputs, and difficulty metadata. These expectations
   are fixture-backed test contracts, not human quality labels.
3. Select **Run agent on cases** to execute each conversation. The live model
   chooses its support and memory tools through native tool calls. The server
   records user turns, model calls, tool arguments and results, answers,
   deterministic checks, timing, and usage. Case isolation keeps another
   conversation's tool data and memory from influencing the result.
4. **Compare:** select **Compare judges**. Jev and the LLM judge independently
   score the same saved answers against the same rubric. Requests are
   sequential, one per case per judge, with the backend order alternating by
   case. Repeating this stage reuses the recorded agent answers.
5. Inspect speed, estimated cost, and scoring differences. Sort by score gap,
   filter disagreements, and open a case to see the input, actual answer,
   expected answer, execution trace, deterministic checks, and both judges'
   dimension scores and confidence. Jev's textual explanations are marked
   unavailable.

**Run full evaluation** performs generation, agent execution, and comparison
in order. Progress updates while a stage runs. Provider errors remain visible;
missing measurements are not filled with synthetic values.

## Inspect tools, memory, and checks

The **Eval set** and **Compare** views report total tool calls and model turns.
A model turn is a recorded model invocation, including an invocation that
selects tools; it is separate from a user turn. Tool counts represent recorded
tool executions, including unsuccessful attempts.

Open a case to inspect:

- **Activity counts:** tool calls, model turns, and the number of recorded
  deterministic checks.
- **Deterministic checks:** each check's name, pass/fail result, expected value,
  and actual value. An unrecorded result stays unknown. These checks are
  separate from rubric scores and gate overall pass. Failed-check counts appear
  in the run summary, case rows, and drawer; a high rubric score does not erase
  a failed check.
- **Execution trace:** user turns and model/tool events in recorded order.
  Expand model calls for usage, or tool calls for arguments, results, latency,
  and call ID. Memory operations are labeled, with returned memory entries
  retained in their full result.
- **Conversation outputs:** recorded answers for each turn, including
  fresh-session follow-ups used to check memory retrieval.
- **Final case-scoped tool and memory state:** only the saved `tool_state`
  attached to this row, when provided.

The UI reads `row.trace` events of type `user_turn`, `model_call`, or `tool_call`,
`row.checks` entries with `name`, `passed`, `expected`, and `actual`, and optional
`row.tool_state` and `row.turn_outputs`. Run totals use recorded `tool_calls` and `model_calls`, or
derive them from available row traces. Older results without traces or checks
remain readable: existing tool-count metadata is retained, while unavailable
model turns, checks, and state are not invented.

## Saved artifacts

The default data directory is `artifacts/tool-memory-demo/`, relative to the repository:

```text
artifacts/tool-memory-demo/
  state.json                 # current agent, suite, outputs, comparison, history
  generated/
    rubric.json
    test_cases.json
    suite.json               # includes generator metadata and skill/agent hashes
  agent_outputs.jsonl        # recorded answers, traces, checks, and agent metadata
  <run-id>.json              # saved completed judge comparison
```

Use `--data-dir /path/to/run-directory` to keep a separate workbench. Restarting
with the same directory restores saved state. **Export current JSON** downloads
the current server state; the run selector opens saved comparisons.

## Share a read-only HTML report

Select **Export HTML** to download `eval-layer-report.html`. The button fetches
the current saved state immediately before export. It includes the agent,
suite, recorded answers, comparison, pricing, and capture time; unsaved edits,
other historical runs, and provider credential status are excluded.

Open the downloaded file directly in a browser, with no server running.
**Agent**, **Eval set**, **Compare**, sorting, disagreement filtering, and the
case evidence drawer remain available, including tool/memory traces, checks,
and final case state. Configuration inputs and mutation buttons are disabled.
The file makes no network requests or model calls.
**Export snapshot JSON** downloads the embedded data locally.

The export embeds JSON in the template's
`<script id="eval-layer-snapshot" type="application/json">` element.
Serialization escapes `<`, `>`, `&`, and Unicode line/paragraph separators.
On opening, the UI serves state reads from that embedded snapshot, rejects
mutation API requests, disables polling, and hides credential status.
A Content Security Policy also blocks connections and form submissions.
All JavaScript and CSS are inline.

For a scripted archive, use the final [UI template](ui/index.html) and the
actual `artifacts/tool-memory-demo/state.json`: embed the same sanitized JSON in that
marker, omit credentials, set `history` to `[]`, and include
`_snapshot: {version: 1, read_only: true, captured_at: "<ISO timestamp>"}`.
Use the template's snapshot CSP. Save the result as
`artifacts/tool-memory-demo/eval-layer-report.html` or `demo/report.html`.
Regenerate the report after template changes; an exported HTML file is a fixed
snapshot. Exporting during a run preserves partial results and labels them
as such. The report includes the supplied policy, prompts, and agent outputs.

## Interpreting the comparison

- This is a **10-case benchmark of one support agent**, not a broad model
  leaderboard. The same configured LLM generates the suite, runs the agent,
  and acts as the LLM judge, so these roles are not independent.
- Expected behavior comes from **authored scenario contracts backed by
  synthetic local fixtures**. These are not human quality labels or calibrated
  rubric scores. Older runs may retain generated reference answers; their
  original provenance remains visible. Agreement, confidence, and pass rate
  do not establish human accuracy or which judge is right.
- Scores use `sum(weight × score / dimension scale)`. A dimension-score
  difference counts as a disagreement. Different dimension scores can yield
  the same weighted score.
- **Overall pass = rubric pass AND deterministic checks pass.**
  `judge.rubric_passed` records whether the rubric threshold is met,
  `judge.checks_passed` records the deterministic gate, and `judge.passed` is
  their combined result. The UI separates rubric score, rubric pass, checks
  pass, and overall pass. A failed check can fail both judges even when their
  rubric scores are high. An overall pass/fail split compares these combined
  outcomes, not just rubric thresholds.
- Speedup is `LLM judge wall time / Jev judge wall time`. Mean request latency
  is shown separately. Agent time is separate from judge time. Relative speed
  and cost savings require completed, successful judging by both backends.
- Costs are **token-based estimates, not invoices**. The checked-in pricing
  snapshot excludes account-specific discounts. Judge comparison costs exclude
  generation and agent execution, whose estimates appear separately;
  missing usage or rates stays unrecorded.

## Troubleshooting

Missing provider credentials appear in the UI. Export both variables into the
shell that starts the server, then restart it. If generation fails, inspect the
reported error and retry **Generate eval set**; a valid suite must contain
exactly 10 cases. If the agent or suite changes, regenerate or rerun the
dependent stage before comparing. If the port is occupied, choose another
`--port` and open its matching localhost address.
