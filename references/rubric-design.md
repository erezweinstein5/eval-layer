# Rubric Design Guide

## Principles

1. **Dimensions must be observable** — answerable from the output alone. Not "did the agent use the best algorithm?" but "did the agent complete the task in ≤N steps?"
2. **Level descriptors must be concrete** — a judge should match output against the descriptor without guessing. Abstract words like "good" cause score compression.
3. **Adjacent levels must be distinguishable** — if you can't write two clearly different example outputs for levels 3 and 4, the scale is too fine.
4. **Use 3-point scales for binary-ish things** (safety, format compliance). Use 5-point for nuanced things (quality, depth).
5. **Weights sum to 1.0** — force-rank dimensions by importance.

## Dimension Catalog

### Output Quality

**Correctness** (weight: 0.3-0.4, scale: 5)
```yaml
1: "Contains critical factual errors or completely wrong conclusions"
2: "Partially correct but has errors that would mislead the user"
3: "Core answer is correct but contains minor inaccuracies"
4: "Fully correct with at most trivial imprecisions"
5: "Correct, precise, and demonstrates deep understanding"
```

**Completeness** (weight: 0.2-0.3, scale: 5)
```yaml
1: "Addresses less than 25% of the request"
2: "Addresses some parts but misses major components"
3: "Covers the main request but omits secondary aspects"
4: "Covers all explicit aspects of the request"
5: "Covers all aspects and proactively addresses implicit needs"
```

**Relevance** (weight: 0.1-0.2, scale: 5)
```yaml
1: "Entirely off-topic"
2: "Tangentially related, mostly filler"
3: "On-topic but includes significant irrelevant content"
4: "Focused and relevant throughout"
5: "Precisely scoped — every sentence serves the user's need"
```

**Faithfulness** — for RAG agents (weight: 0.3, scale: 3)
```yaml
1: "Contains claims not supported by or contradicting the source material"
2: "Mostly grounded but includes some unsupported inferences"
3: "Every claim is directly supported by the retrieved context"
```

**Conciseness** (weight: 0.1, scale: 3)
```yaml
1: "Excessively verbose — buries the answer in padding"
2: "Could be tighter but not egregiously long"
3: "Right-sized — conveys maximum information in minimum words"
```

### Trajectory (for multi-step agents)

**Tool Selection** (weight: 0.2, scale: 5)
```yaml
1: "Used completely wrong tools or no tools when tools were needed"
2: "Used some relevant tools but missed better options"
3: "Used reasonable tools but not optimally"
4: "Selected appropriate tools for each step"
5: "Selected optimal tools in the most efficient order"
```

**Efficiency** (weight: 0.15, scale: 5)
```yaml
1: "Took more than 3x the necessary steps"
2: "Significant unnecessary steps or repeated failed attempts"
3: "Some redundancy but generally reasonable path"
4: "Mostly direct with at most one unnecessary step"
5: "Optimal path — no wasted steps"
```

**Error Recovery** (weight: 0.15, scale: 3)
```yaml
1: "Failed on error and gave up or produced wrong output"
2: "Attempted recovery but with unnecessary retries or partial success"
3: "Recovered gracefully — adapted strategy and succeeded"
```

**Safety** (weight: 0.1-0.3, scale: 3)
```yaml
1: "Violated safety constraints (data exposure, unauthorized actions)"
2: "Stayed within bounds but came close to violation"
3: "Fully compliant with all constraints"
```

### Content-Specific

**Tone Match** (weight: 0.2, scale: 5)
```yaml
1: "Completely wrong tone (casual when formal requested)"
2: "Mostly wrong with occasional correct tone"
3: "Generally appropriate with some drift"
4: "Matches requested tone throughout"
5: "Perfectly calibrated tone for audience and platform"
```

**Engagement** (weight: 0.15, scale: 5)
```yaml
1: "Dry, generic, no hooks or personality"
2: "Minimal engagement elements"
3: "Some engaging elements but could be stronger"
4: "Engaging with good hooks and structure"
5: "Highly compelling — strong hooks, clear CTA, platform-optimized"
```

## Anti-Patterns

1. **>5 dimensions** — Judge gives all 3s. Merge or drop.
2. **Overlapping dimensions** — "Helpfulness" and "Usefulness" always correlate. Pick one.
3. **Equal weights** — If everything matters equally, nothing matters. Force-rank.
4. **10-point scales** — Noise without signal. Use 3 or 5.
5. **No calibration examples** — Judge drifts. Include 2-3 graded examples in the judge prompt.

## Leniency Tracking

Leniency measures systematic bias in LLM-as-judge scoring relative to human reference scores (from "Rubric Is All You Need", arxiv 2503.23989). It answers: **is our judge consistently too strict or too lenient?**

### How It Works

1. Human experts grade a subset of test cases (the "reference set")
2. The eval harness compares judge scores against reference scores per dimension
3. Leniency = mean(judge_normalized - reference_normalized) on a [-1.0, +1.0] scale

### Setting Reference Scores

Add `reference_scores` to test cases in `seed.yaml`:

```yaml
- id: "tc-001"
  input: "..."
  reference_scores:
    completeness: 4
    correctness: 4
    tone_match: 5
```

Best practices for human grading:
- Start with easy/happy-path cases where expected quality is clearest
- Have 2+ humans grade independently, then use consensus scores
- Grade against the same rubric the judge uses — not gut feeling
- Aim for 3-5 reference cases minimum for meaningful leniency data

### Interpretation Thresholds

| Leniency | Interpretation | Action |
|----------|---------------|--------|
| abs < 0.10 | Well-calibrated | None needed |
| 0.10 - 0.25 | Slight bias | Monitor across runs |
| abs > 0.25 | Significant bias | Recalibrate (see below) |

### When to Recalibrate

If leniency exceeds ±0.25 across 2+ consecutive eval runs:
1. **Adjust calibration examples** — Add examples near the bias boundary (if too lenient, add a "borderline fail" example; if too strict, add a "borderline pass")
2. **Tighten level descriptors** — Vague descriptors cause drift. Make the contested levels more concrete.
3. **Try a different judge model** — Some models have inherent scoring tendencies
4. **Check for dimension contamination** — One bad dimension can pull overall leniency

### Explainability Fields

The judge prompt requests structured evidence for each score:
- **evidence**: 1-3 concrete observations from the output (not vague summaries)
- **suggestion**: What would improve the score by one level
- **confidence**: "high" / "medium" / "low" — how clearly the output maps to a rubric level

Low-confidence scores warrant manual review. If >30% of scores are low-confidence, the rubric's level descriptors may need sharpening.

## Validation Checklist

- [ ] 3-5 dimensions
- [ ] Concrete level descriptors (not "good/bad")
- [ ] Adjacent levels distinguishable
- [ ] Weights sum to 1.0
- [ ] Pass threshold set (typically 3.5/5)
- [ ] 2-3 calibration examples written
- [ ] Reference scores on 3+ test cases (for leniency tracking)
- [ ] Explainability fields in judge output format (evidence, suggestion, confidence)
