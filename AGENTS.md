# AGENTS.md — notes for sessions editing this skill

This file is for Codex, Claude Code, and other assistant sessions working **on** the `eval-layer` skill. Sessions *using* the skill load `SKILL.md` directly — don't duplicate that content here.

---

## Load-bearing invariants (change these carefully)

### The 7-field metadata contract

```python
{"recommendation", "latency_ms", "tool_calls",
 "input_tokens", "output_tokens", "model_id", "error"}
```

Appears in all of these files — if you change the shape, update **all** of them in the same commit:

- `SKILL.md` → "Metadata contract" section
- `README.md` → "The 7-field metadata contract" section
- `references/framework-adapters.md` → the opening contract block + every per-framework recipe
- `references/judge-robustness.md` → "Never drop a result" JSONL schema + "Failure categories to report" table
- `references/cross-subject-benchmarking.md` → SUBJECTS dispatch explanation
- `references/html-report-template.html` → column headers in the leaderboard + failure-categories table
- `scripts/codex_adapter.py` and `references/codex.md`
- Any example harness code in the repo

### Required CLI flags on generated harnesses

`--framework`, `--test-case`, `-v`/`--verbose`, `--trials`, `--no-judge`. Baked into `SKILL.md` step 2d. Don't drop one without updating the skill's validation checklist too.

### Leniency thresholds

`<0.10` / `0.10–0.25` / `>0.25`. Defined in `references/rubric-design.md` and referenced by both `SKILL.md` step 3 and the harness's "recalibrate" warning. Keep them aligned.

---

## How to test changes to this skill

Validate skill metadata with the skill-creator validator when available. Check relative links and run `git diff --check`.

For Codex adapter changes, run `python3 -m unittest discover -s tests -v`. These tests use a fake CLI and require no model credentials. Check success, event accounting, missing usage, malformed output, failures, and timeout behavior. Keep tests independent of a particular locally installed Codex model.

When workflow guidance changes, exercise it in a fresh temporary agent fixture: generate the rubric, cases, judge prompt, and harness; parse YAML; compile Python; and run a case with `--no-judge`. For multi-subject changes, run two variants and verify raw result rows and report generation. A bounded offline smoke test may use one case per subject; the generated production seed suite still follows SKILL.md coverage requirements. Mock the target agent when only checking harness wiring and label the result as a mock run. Live model evaluation is separate; do not describe offline checks as live quality validation.

Never call a paid judge in a test pass unless the user explicitly asks. `--no-judge` skips only the judge; a live Codex target still invokes a model.


## Format conventions for `references/`

- **Copy-paste-ready.** Every code snippet should run standalone (minus framework imports the user is already using). No pseudocode.
- **~150–300 lines per file.** Longer and the reference becomes a book; shorter and it's a footnote.
- **Match existing voice.** Brief, direct, ⚠️ for pitfalls, tables for comparison, no hype.
- **Cross-reference by path.** `see references/foo.md` — not "see the other doc". Paths survive renames better than prose references.
- **No framework worship.** The skill is framework-agnostic. When a framework misbehaves, document the fix; don't steer users away from it.

---

## What NOT to add

- **New runtime deps in generated harnesses.** Stick to `anthropic`, `pyyaml`, and stdlib. Users have to `pip install` whatever we generate — no Jinja2, no click, no rich, no tqdm. The existing `make_html_report.py` uses `string.Template` specifically to avoid Jinja2.
- **Opinionated rubrics.** The dimension catalog in `rubric-design.md` is a menu, not a mandate. Don't hardcode "correctness + completeness + faithfulness" into the skill.
- **Framework favoritism.** If a recipe in `framework-adapters.md` is longer/uglier for one framework, that's how it is. Don't hide the complexity.
- **Auto-running the judge.** Always gate judge calls behind `--no-judge` off by default *in user docs*, but leave it as opt-in in test passes. The user spends real money when the judge runs.
- **Unmotivated test variants.** Add cases that cover observable behavior or a demonstrated failure.

---

## Known open gaps (live TODOs, not closed)

1. **`TOTAL_COST_USD` is hardcoded `"n/a"`** in the HTML dashboard. Fixing it requires a model-pricing table in the skill. Options: (a) drop the field, (b) add `pricing.yaml` and teach `make_html_report.py` to look up `model_id`. Lean (a) unless someone asks.
2. **Radar axes mix 3-point and 5-point scales.** The current template plots raw scores with `suggestedMax: 5`, so a perfect 3/3 on `schema_compliance` looks like 60%. Fix: normalize every dim to `[0, 1]` before plotting. Hasn't bitten anyone yet because most rubrics are all-5-point.
3. **No Rogan–Gladen correction.** `hamelsmu/evals-skills` corrects weighted scores for judge bias using the leniency measurement. We expose leniency but don't fold it back. Open design question: does correction belong in the harness or stay a post-hoc adjustment in the report?
4. **No bootstrap confidence intervals.** Pass rates are point estimates. With 8–10 test cases, that's noisy. Consider adding bootstrap CI to the leaderboard.
5. **Client construction outside the try block.** The `framework-adapters.md` recipes mostly put client init at module scope. A client-construction failure then escapes the 7-field contract. Fix is a one-liner per recipe (move inside the function, wrap in try) — worth a pass when next editing that file.

---

## Sources and prior art

- **`hamelsmu/evals-skills`** (https://github.com/hamelsmu/evals-skills) — public eval skill. Borrowed the "reference scores on easy cases" pattern. Not yet imported: Rogan–Gladen correction, error-analysis-first gate, train/dev/test calibration splits.
- **`~/dev/agent-frameworks-bench/`** — earlier multi-framework SEO benchmark. The HTML dashboard template in `references/html-report-template.html` is a generalized version of the one that lives in that repo.
- **"Rubric Is All You Need"** (arxiv 2503.23989) — source for the leniency formulation.

---

## Commit hygiene

- Attribution disabled globally; don't add co-author lines.
- Commit messages: `type: description` (feat / fix / docs / refactor).
- `main` is protected on GitLab — no force-push. Pull-rebase-push if the remote moves.
- No secrets in this repo. Nothing should ever be gitignored for that reason; `.gitignore` is for `__pycache__` and editor junk only.
