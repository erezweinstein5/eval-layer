# Interactive Human Calibration

**Opt-in only.** The skill's default is to auto-generate `reference_scores` with Claude and tag them `graded_by: claude`. Leniency computed against Claude-graded references is *directional only* — it can catch gross judge drift but cannot detect shared LLM bias between reference and judge. Most users don't need more than that.

Run this flow only when the user explicitly asks — e.g., invokes the harness with `--calibrate`, says "I want human-graded references", the project has regulatory / high-stakes requirements for judge bias detection, or leniency stays suspiciously near zero (`|leniency| < 0.05` across runs is a smell for LLM-on-LLM echo).

---

## When to run it

After the agent exists, the skill has generated the default (Claude-graded) rubric + test cases + harness, and the user has opted in. The flow:

```
1. Skill generates rubric + seed.yaml with Claude-graded reference_scores (graded_by: claude)
2. User invokes --calibrate (or explicitly asks for human grading)
3. Skill runs the agent on the 5 calibration cases with --no-judge
4. Skill walks the user through grading each one interactively   ← this file
5. Skill overwrites reference_scores in seed.yaml and flips graded_by to "human"
6. Next eval sweep — leniency is now human-anchored
```

---

## How many cases to grade

**5 cases.** Tradeoffs:

| N  | Quality of leniency signal | User patience |
|----|---|---|
| 3  | Noisy — one odd grade skews the mean | Easy |
| 5  | Solid — 5 × D dimensions = 20+ samples | ~20-25 min total |
| 8+ | Diminishing returns; fatigue lowers grading quality | Users skip or rush |

All 5 should come from the **easy** pool. Hard/ambiguous cases are where humans disagree — using them as reference would inject noise into the calibration baseline.

---

## What to grade against

**Real agent output on the specific case, not the `expected_output` sketch.**

The `expected_output` field in `seed.yaml` is a *hint* for what a strong brief might look like. It's written by Claude during generation. Grading against it is circular.

So the calibration flow must run the agent first:

```bash
for case_id in easy-01 easy-02 easy-03 easy-04 easy-05; do
    python evals/eval_harness.py --framework <name> --test-case $case_id --no-judge -v
done
```

These runs populate `evals/reports/raw/<framework>.jsonl` with the real agent outputs. The interactive calibration session reads them from there.

---

## The interactive dialog

When the user invokes `/eval-layer --calibrate` (or Claude decides calibration is next), the session loops case × dimension:

### Per-case opening

```
📋 Case 1 of 5: easy-01
─────────────────────────────────────
Input: "ai coding assistants"

Context: Canned data: trend=rising, volume=22000, competition=high.
Rising queries include 'best ai coding assistant 2026', 'claude code vs cursor'.

Agent output:
{
  "target_keyword": "best ai coding assistants 2026",
  "search_intent": "commercial",
  "content_outline": [
    "What an AI coding assistant does",
    "Evaluation criteria",
    "Head-to-head comparison",
    "When not to use one",
    "Verdict"
  ],
  "call_to_action": "Try our top pick free for 14 days.",
  "estimated_word_count": 2200
}

Rubric dimensions: keyword_relevance, outline_actionability, completeness, schema_compliance
I'll walk you through one dimension at a time. You can ask questions at any point.
```

### Per-dimension prompt

```
─── Dimension 1/4: keyword_relevance (scale 1-5) ───

Level descriptors:
  1 — Keyword is off-topic, tautological, or contradicts trend data
  2 — Topic-adjacent but generic; outline touches few rising_queries
  3 — Reasonable but could be more specific
  4 — Specific and well-chosen; outline reflects rising_queries
  5 — Precisely scoped realistic search phrase; clearly synthesizes trend + SERP

Your score? (1-5, or ask a question)
```

### User can ask questions mid-session

The session is a dialog, not a form. Expected patterns:

- **"What's the difference between 3 and 4 here?"** → Claude quotes the two descriptors and points to specific parts of the agent output that lean either way.
- **"Is the outline length supposed to affect this dimension?"** → Claude clarifies: outline length is `outline_actionability`, not `keyword_relevance`.
- **"What does 'synthesizes trend + SERP' mean?"** → Claude explains in the context of the specific rising_queries.
- **"Can I go back and change easy-01 keyword_relevance?"** → Yes; Claude re-opens that slot.

Claude must **not** suggest a score unless the user explicitly asks "what would you score this?" — and even then, frame it as *one possible interpretation*, not the answer. The whole point is independent human judgment.

### Per-dimension confirmation

```
You scored keyword_relevance = 4.

Optional: briefly, why? (recorded as a comment on the case — helps future you
understand this grade if you re-run in 3 months. Skip with Enter.)
```

### Per-case close

```
✓ easy-01 graded:
  keyword_relevance: 4
  outline_actionability: 4
  completeness: 4
  schema_compliance: 3

Moving to case 2 of 5...
```

### Session close

```
✓ All 5 calibration cases graded.

Updating evals/test_cases/seed.yaml with reference_scores.
Total time: 18 min.

Next: run the full eval sweep with
  python evals/eval_harness.py --framework <name>

Leniency will now be meaningful. If it's > ±0.25, recalibrate the judge
(see references/rubric-design.md).
```

---

## What to write back to `seed.yaml`

For each graded case:

```yaml
- id: "easy-01"
  input: "ai coding assistants"
  # ... existing fields ...
  reference_scores:
    keyword_relevance: 4
    outline_actionability: 4
    completeness: 4
    schema_compliance: 3
  reference_metadata:
    graded_by: human
    graded_at: "2026-04-18T14:23:00Z"
    agent_output_from: "evals/reports/raw/raw_anthropic.jsonl"  # provenance
    notes:                                                      # optional per-dim rationale
      keyword_relevance: "Matches a rising query directly."
      outline_actionability: "Sections are specific enough but missing a pricing angle."
```

The `reference_metadata` block makes it possible to:
- Re-grade later (you know what output you graded against)
- Flag stale references when the agent changes substantially
- Audit who graded what

---

## Anti-patterns

1. **Claude grades on the user's behalf.** Defeats the purpose. Ask; don't assume.
2. **Grading before running the agent.** You'd be grading `expected_output`, which Claude wrote. Circular.
3. **Using hard cases for reference.** Human inter-rater disagreement on hard cases is high — that uncertainty leaks into the leniency signal.
4. **Skipping the "why" prompt every time.** At least 2-3 per session; helps detect rubric ambiguity for future you.
5. **Grading more than 8 cases in one sitting.** Fatigue makes late scores inconsistent. Cap sessions at 5-6.

---

## Re-calibration triggers

Re-run the interactive calibration when:

- The rubric changes (new dimension, different scale, reworded descriptor)
- The agent changes substantially (new model, new prompt, new tools) — the output distribution shifts, so old references may no longer be representative
- Leniency drifts > ±0.25 across runs even when the judge hasn't changed — suggests reference staleness

Expected cadence: once per rubric version. Not every eval run.
