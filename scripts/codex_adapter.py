"""Stdlib Codex exec adapter. See references/codex.md for harness integration."""

import json
import os
from pathlib import Path
import signal
import subprocess
import time


CALL_ITEMS = {"command_execution", "mcp_tool_call", "web_search"}


def _read_events(path, result):
    """Retain observed counts even when another event makes the run invalid."""
    calls = set()
    totals = {"input_tokens": 0, "output_tokens": 0}
    known = dict.fromkeys(totals, True)
    completed = 0
    pending = False
    errors = []
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
                if not isinstance(event, dict) or not isinstance(event.get("type"), str):
                    raise ValueError("expected event object with type")
                kind = event["type"]
                if kind == "turn.started":
                    pending = True
                elif kind == "turn.completed":
                    completed += 1
                    pending = False
                    usage = event.get("usage")
                    if usage is not None and not isinstance(usage, dict):
                        raise ValueError("usage must be an object or null")
                    for key in totals:
                        value = (usage or {}).get(key)
                        if value is None:
                            known[key] = False
                        elif type(value) is not int or value < 0:
                            raise ValueError(f"invalid {key}")
                        else:
                            totals[key] += value
                elif kind in {"error", "turn.failed"}:
                    errors.append(f"{kind}: {json.dumps(event, ensure_ascii=False)}")
                elif kind in {"item.started", "item.updated", "item.completed"}:
                    item = event.get("item")
                    if not isinstance(item, dict) or not isinstance(item.get("type"), str):
                        raise ValueError("expected item object with type")
                    if item["type"] in CALL_ITEMS:
                        item_id = item.get("id")
                        if not isinstance(item_id, str) or not item_id:
                            raise ValueError("tool item missing id")
                        calls.add(item_id)
            except (ValueError, TypeError) as exc:
                errors.append(f"malformed stream line {number}: {exc}")
    result["tool_calls"] = len(calls)
    for key in totals:
        result[key] = totals[key] if completed and known[key] else None
    if not completed or pending:
        errors.append("no completion: missing terminal turn.completed")
    return errors


def run(prompt, *, workspace, model_id, artifacts_dir, timeout_s=300,
        executable="codex"):
    """Run in a fresh fixture; artifacts_dir must be new and outside it.

    Returns exactly the seven metadata fields, including on invocation failure.
    model_id records the explicitly requested model, not a resolved server ID.
    No judge is invoked. Raw final text is wrapped in a JSON-compatible object.
    """
    start = time.perf_counter()
    result = {
        "recommendation": None, "latency_ms": 0, "tool_calls": 0,
        "input_tokens": None, "output_tokens": None,
        "model_id": model_id, "error": None,
    }
    errors = []
    try:
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("model_id must be explicit and nonempty")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        workspace = Path(workspace).resolve(strict=True)
        artifacts = Path(artifacts_dir).resolve()
        if artifacts == workspace or workspace in artifacts.parents:
            raise ValueError("artifacts_dir must be outside the workspace")
        artifacts.mkdir(parents=True, exist_ok=False)
        final_path = artifacts / "final.txt"
        argv = [
            str(executable), "exec", "--json", "--sandbox", "workspace-write",
            "--model", model_id, "-o", str(final_path), "-",
        ]
        (artifacts / "prompt.txt").write_text(prompt, encoding="utf-8")
        (artifacts / "invocation.json").write_text(json.dumps({
            "argv": argv, "cwd": str(workspace), "timeout_s": timeout_s,
        }, indent=2), encoding="utf-8")
        events_path = artifacts / "events.jsonl"
        with events_path.open("wb") as stdout, (artifacts / "stderr.txt").open("wb") as stderr:
            with subprocess.Popen(
                argv, cwd=workspace, stdin=subprocess.PIPE, stdout=stdout,
                stderr=stderr, start_new_session=(os.name == "posix"),
            ) as process:
                try:
                    process.communicate(prompt.encode("utf-8"), timeout=timeout_s)
                except subprocess.TimeoutExpired:
                    errors.append(f"timeout after {timeout_s}s")
                    if os.name == "posix":
                        # Stop ordinary child commands as well as the CLI.
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                    else:
                        process.kill()
                    process.communicate()
                if process.returncode:
                    errors.append(f"nonzero exit {process.returncode}; see stderr.txt")
                (artifacts / "process.json").write_text(
                    json.dumps({"returncode": process.returncode}), encoding="utf-8",
                )
        errors.extend(_read_events(events_path, result))
        if not errors:
            final_message = final_path.read_text(encoding="utf-8")
            if not final_message.strip():
                raise ValueError("missing final output: final.txt is empty")
            result["recommendation"] = {"final_message": final_message}
    except (OSError, ValueError, TypeError, subprocess.SubprocessError) as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    finally:
        result["latency_ms"] = round((time.perf_counter() - start) * 1000)
    if errors:
        result["error"] = "; ".join(errors)
        result["recommendation"] = None
    return result
