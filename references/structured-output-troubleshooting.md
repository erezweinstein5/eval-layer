# Structured Output Troubleshooting

Claude on Bedrock (Opus 4.6 and the 4.x family) rejects several structured-output
idioms that work fine on the Anthropic direct API. When you see one of these
errors, switch to the **two-stage structured output** pattern.

---

## The three errors you will hit

### 1. LangGraph `response_format=Schema`

```
ValidationException: This model does not support assistant message prefill.
The conversation must end with a user message.
```

**Cause**: `create_react_agent(..., response_format=Schema)` appends an
assistant-role prefill instructing the model to emit JSON. Bedrock's Converse
API with Opus disallows trailing assistant messages.

**Fix**: See LangGraph recipe in `framework-adapters.md` — run the ReAct loop
to completion, then call `llm.with_structured_output(Schema).invoke(...)` on
the final text in a second round-trip.

### 2. Strands `agent.structured_output()`

```
ValueError: No valid tool use or tool use input was found in the Bedrock response.
```

**Cause**: Strands emits the schema as a forced tool call. On some turns Opus
returns a text block instead of the expected tool_use block, and the parser
bails.

**Fix**: Call the agent conversationally first, then structure with a fresh
agent that has no other tools:

```python
conv_result = agent(prompt)
structurer = Agent(model=BedrockModel(...), system_prompt="Convert to schema.")
rec = structurer.structured_output(Schema, f"Convert:\n\n{conv_result.message}")
```

### 3. OpenAI Agents SDK + LiteLLM `output_type=Schema`

```
BedrockException: output_config.format.schema: For 'array' type,
'minItems' values other than 0 or 1 are not supported (got: [2, 5]).
```

**Cause**: LiteLLM translates Pydantic `min_length`/`max_length` on
`list[...]` fields into `minItems`/`maxItems` in the JSON Schema. Bedrock's
tool-schema-constrained output mode only allows values of 0 or 1 there.

**Fix**: Drop `output_type` entirely. Let the agent emit text, then
`_extract_json()` + `Schema.model_validate()` the result:

```python
agent = Agent(
    instructions=SYSTEM_PROMPT + "\n\nReply with a single JSON object.",
    # NO output_type
    ...
)
raw = (await Runner.run(agent, prompt)).final_output
rec = Schema.model_validate(_extract_json(str(raw)))
```

---

## The two-stage pattern (default for Bedrock)

For any framework targeting Claude on Bedrock, make this the **default**,
not a fallback:

```
Stage 1 (reasoning + tools):
  Run the agent conversationally — tools enabled, no output_type/response_format.
  The agent calls tools, reasons, and emits a free-form final message.

Stage 2 (structuring):
  Feed the final message into a fresh LLM call with with_structured_output /
  structured_output / model_validate(extract_json(...)).
  No tools, no system prompt beyond "Convert this to the schema verbatim".
```

### Why it's better

- **Robust**: sidesteps all three errors above
- **Debuggable**: Stage 1 text is human-readable; schema failures are isolated to Stage 2
- **Cheap**: Stage 2 is one call with small input/output
- **Portable**: works across every framework the same way

### When not to use it

- Direct Anthropic API (not Bedrock): native structured output works fine
- Agent's output schema has no list constraints: single-stage `output_type` may work
- Latency-critical paths where 1 extra round-trip is unacceptable (rare)

---

## Validating your schema before you hit these errors

Run this snippet during Step 1 of the skill to predict which frameworks will
work:

```python
from your_schemas import Schema
print(Schema.model_json_schema())
```

Look for:
- `"minItems": N` where `N > 1` → will fail on Bedrock's OpenAI-Agents path
- nested objects 3+ levels deep → structured-output reliability drops sharply
- `"prefixItems"` (tuple-like lists) → unsupported by most frameworks' structured output

If any of these appear, document the two-stage pattern requirement in your
harness and save yourself debugging time.
