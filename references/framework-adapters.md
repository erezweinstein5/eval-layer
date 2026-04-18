# Framework Adapter Recipes

Each framework exposes *invocation*, *structured output*, *token counts*, and
*tool-call counts* through a different API surface. Below are tested recipes
that conform to the harness's metadata contract (see `SKILL.md` →
"Metadata contract").

Every adapter must return an object with at minimum:

```python
{
    "recommendation": <your schema instance or None>,
    "latency_ms": int,
    "tool_calls": int,
    "input_tokens": int | None,
    "output_tokens": int | None,
    "error": str | None,
}
```

---

## PydanticAI

```python
from pydantic_ai import Agent, RunContext

_agent = Agent(
    f"bedrock:{MODEL_ID}",
    output_type=OutputSchema,
    instructions=SYSTEM_PROMPT,
)

@_agent.tool
def my_tool(_ctx: RunContext, arg: str) -> dict:
    return {...}

def run_agent(prompt: str):
    start = time.perf_counter()
    result = _agent.run_sync(prompt)

    # Output
    rec = result.output                       # already an OutputSchema instance

    # Tool calls: walk ToolCallPart in message parts
    tool_calls = sum(
        1
        for msg in result.all_messages()
        for part in getattr(msg, "parts", [])
        if type(part).__name__ == "ToolCallPart"
    )

    # Tokens
    usage = result.usage()
    input_tokens = getattr(usage, "request_tokens", None) or getattr(usage, "input_tokens", None)
    output_tokens = getattr(usage, "response_tokens", None) or getattr(usage, "output_tokens", None)
```

---

## LangGraph (`langchain-aws.ChatBedrockConverse`)

⚠️ **Do not use `response_format=Schema`** — Bedrock Opus rejects assistant
message prefill. Use the two-stage structured output pattern:

```python
from langchain_aws import ChatBedrockConverse
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

_LLM = ChatBedrockConverse(model=MODEL_ID, temperature=0)

@tool
def my_tool(arg: str) -> dict:
    return {...}

def run_agent(prompt: str):
    graph = create_react_agent(_LLM, tools=[my_tool], prompt=SYSTEM_PROMPT)
    result = graph.invoke({"messages": [("user", prompt)]})

    # Extract final conversational text
    final_text = result["messages"][-1].content
    if isinstance(final_text, list):
        final_text = "".join(p.get("text", "") for p in final_text if isinstance(p, dict))

    # Structure in a second call
    rec = _LLM.with_structured_output(OutputSchema).invoke(
        [
            ("system", "Convert the analysis into the schema. Preserve specific claims verbatim."),
            ("user", str(final_text)),
        ]
    )

    # Tool calls: sum tool_calls across messages
    messages = result.get("messages", [])
    tool_calls = sum(len(getattr(m, "tool_calls", []) or []) for m in messages)

    # Tokens: sum usage_metadata across all messages
    input_tokens = sum((m.usage_metadata or {}).get("input_tokens", 0) for m in messages if hasattr(m, "usage_metadata"))
    output_tokens = sum((m.usage_metadata or {}).get("output_tokens", 0) for m in messages if hasattr(m, "usage_metadata"))
```

---

## CrewAI (LiteLLM → Bedrock)

```python
from crewai import LLM, Agent, Crew, Process, Task
from crewai.tools import tool

@tool("my_tool")
def my_tool(arg: str) -> dict:
    return {...}

def run_agent(prompt: str):
    llm = LLM(model=f"bedrock/{MODEL_ID}", temperature=0)
    agent = Agent(role="...", goal="...", backstory=SYSTEM_PROMPT,
                  tools=[my_tool], llm=llm, allow_delegation=False)
    task = Task(description=prompt, expected_output="JSON",
                agent=agent, output_pydantic=OutputSchema)
    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential)
    output = crew.kickoff()

    # Output — pre-parsed by CrewAI
    rec = getattr(output, "pydantic", None)

    # Tokens
    usage = getattr(output, "token_usage", None)
    input_tokens = getattr(usage, "prompt_tokens", None) if usage else None
    output_tokens = getattr(usage, "completion_tokens", None) if usage else None

    # Tool calls: CrewAI doesn't reliably expose per-call counts.
    # Count them yourself by decorating the tool with a counter, or accept 0.
    tool_calls = 0
```

---

## Strands Agents (native Bedrock)

⚠️ `agent.structured_output()` intermittently fails on Opus with
`"No valid tool use or tool use input was found in the Bedrock response"`.
Use the two-stage pattern:

```python
from strands import Agent, tool
from strands.models import BedrockModel

@tool
def my_tool(arg: str) -> dict:
    return {...}

def run_agent(prompt: str):
    model = BedrockModel(model_id=MODEL_ID, temperature=0)
    agent = Agent(model=model, system_prompt=SYSTEM_PROMPT, tools=[my_tool])

    # Stage 1: conversational run with tools
    conv_result = agent(prompt)
    final_text = str(getattr(conv_result, "message", conv_result))

    # Stage 2: structure via a fresh agent (no tools needed)
    structurer = Agent(
        model=BedrockModel(model_id=MODEL_ID, temperature=0),
        system_prompt="Convert the analysis into the schema. Preserve specific claims verbatim.",
    )
    rec = structurer.structured_output(
        OutputSchema,
        f"Convert this into JSON:\n\n{final_text}",
    )

    # Tool calls: walk messages for toolUse blocks
    tool_calls = sum(
        1
        for m in (getattr(agent, "messages", []) or [])
        for block in (m.get("content", []) if isinstance(m, dict) else [])
        if isinstance(block, dict) and "toolUse" in block
    )

    # Tokens: not reliably exposed on Strands 0.1.x
    input_tokens = output_tokens = None
```

---

## OpenAI Agents SDK (LiteLLM → Bedrock)

⚠️ Do **not** use `output_type=Schema` with min/max length constraints —
LiteLLM serializes them as `minItems > 1` which Bedrock's schema-constrained
mode rejects. Use text output + manual JSON extraction.

⚠️ Also: the SDK's package name is `agents`. If your project has a top-level
`agents/` folder, you must rename it (e.g. `bench_agents/`) or imports will
clash.

```python
from agents import Agent, Runner, function_tool
from agents.extensions.models.litellm_model import LitellmModel

@function_tool
def my_tool(arg: str) -> dict:
    return {...}

def _extract_json(text: str) -> dict | None:
    # See references/judge-robustness.md
    ...

async def run_agent(prompt: str):
    agent = Agent(
        name="...",
        instructions=SYSTEM_PROMPT + "\n\nReply with a single JSON object matching the schema.",
        model=LitellmModel(model=f"bedrock/{MODEL_ID}"),
        tools=[my_tool],
        # NO output_type — let it emit text
    )
    result = await Runner.run(agent, prompt)

    # Output: parse text → JSON → schema
    raw = result.final_output
    parsed = _extract_json(str(raw))
    # Some outputs wrap the schema in a single-key object; unwrap one level:
    if parsed and "expected_field" not in parsed and len(parsed) == 1:
        parsed = next(iter(parsed.values()))
    rec = OutputSchema.model_validate(parsed) if parsed else None

    # Tool calls: walk new_items for ToolCallItem
    tool_calls = sum(
        1 for item in (getattr(result, "new_items", []) or [])
        if type(item).__name__ == "ToolCallItem"
    )

    # Tokens
    usage = getattr(result, "usage", None)
    input_tokens = getattr(usage, "input_tokens", None) if usage else None
    output_tokens = getattr(usage, "output_tokens", None) if usage else None
```

---

## Raw Anthropic SDK (baseline)

```python
import anthropic

client = anthropic.AnthropicBedrock()   # or Anthropic() for direct API

def run_agent(prompt: str):
    tools_spec = [
        {"name": "my_tool", "description": "...", "input_schema": {...}},
    ]
    messages = [{"role": "user", "content": prompt}]
    tool_calls = 0
    input_tokens = output_tokens = 0

    while True:
        resp = client.messages.create(
            model=MODEL_ID,
            system=SYSTEM_PROMPT,
            tools=tools_spec,
            messages=messages,
            max_tokens=4096,
            temperature=0,
        )
        input_tokens += resp.usage.input_tokens
        output_tokens += resp.usage.output_tokens

        if resp.stop_reason == "end_turn":
            text = next(b.text for b in resp.content if b.type == "text")
            rec = OutputSchema.model_validate_json(_extract_json_str(text))
            break

        # Execute tool_use blocks and append tool_result messages
        messages.append({"role": "assistant", "content": resp.content})
        tool_results = []
        for block in resp.content:
            if block.type == "tool_use":
                tool_calls += 1
                result = dispatch_tool(block.name, block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result),
                })
        messages.append({"role": "user", "content": tool_results})
```

---

## Adding a new framework

1. Find the framework's *invocation* entry point (`.run`, `.invoke`, `Runner.run`, etc.)
2. Find its *structured output* path. If it doesn't work against Bedrock Opus, fall back to two-stage (see `structured-output-troubleshooting.md`)
3. Find *token* metadata. May be per-message, per-result, or missing entirely
4. Find *tool-call* counts. Usually requires walking a message/content list for tool-use blocks
5. Add a recipe here so the next user doesn't re-investigate
