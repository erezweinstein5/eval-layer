# Judge Prompt Guide

## Template

```markdown
You are an expert evaluator assessing the quality of an AI agent's output.

## Your Task

You will receive:
- **Input**: The original request given to the agent
- **Agent Output**: The agent's response
- **Reference Answer**: What a correct response looks like (guide, not ground truth)

Evaluate the agent's output against each dimension in the rubric below.

## Rubric

{RUBRIC_DIMENSIONS_HERE}

## Instructions

1. For each dimension, read the level descriptors carefully
2. Compare the agent's output against the descriptors
3. Write 1-2 sentences of reasoning BEFORE assigning a score
4. Evaluate each dimension INDEPENDENTLY — do not let one bias another
5. The reference answer is one valid response; the agent may have an equally valid alternative

## Calibration Examples

{EXAMPLES_HERE}

## Output Format

Respond with ONLY a JSON object:

{
  "dimensions": [
    {"name": "dim_name", "reasoning": "...", "score": N, "max_score": N}
  ],
  "overall_reasoning": "1-2 sentences on overall quality",
  "weighted_score": N.N,
  "pass": true/false
}
```

## Best Practices

1. **Reasoning before scores** — reduces random scoring, makes it auditable
2. **Strong judge model** — at least as capable as the agent's model. Use Claude Sonnet 4.6 or Opus.
3. **Temperature 0** — maximizes scoring consistency
4. **Independent dimensions** — explicitly instruct to avoid halo effect
5. **Reference as guide** — accept equally valid alternatives
6. **One test case at a time** — batch evaluation causes anchoring drift

## Calibration Examples

Include 2-3 examples in the judge prompt:
1. **Clear pass** (score 4-5) — what good looks like
2. **Borderline** (score 3) — where the line is
3. **Clear fail** (score 1-2) — what bad looks like

## Response Parsing

Parse defensively — try direct JSON, then code block extraction, then first `{...}` match:

```python
import json, re

def parse_judge_response(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if match:
        return json.loads(match.group(1))
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        return json.loads(match.group())
    return {"error": "Failed to parse", "dimensions": [], "pass": False}
```
