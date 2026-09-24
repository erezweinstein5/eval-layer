#!/usr/bin/env python3
"""Prepare demo code candidates, evaluate them with live Jev, and record stdout.

The candidates are authored examples, not outputs from a live coding-agent run.
The recording contains real judge results and independently executed code checks.
"""
import argparse
from datetime import datetime, timezone
import difflib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.rejudge import replay_rows, summarize, render_markdown

TASK = ("Fix average(values) so an empty list returns 0. Preserve ordinary, negative, "
        "and fractional averages. Change only stats.py.")
BASE = 'def average(values):\n    return sum(values) / len(values)\n'
FIX = 'def average(values):\n    if not values:\n        return 0\n    return sum(values) / len(values)\n'
CHECK = '''import importlib.util, json, pathlib
spec = importlib.util.spec_from_file_location("candidate", pathlib.Path("stats.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
checks = []
for name, values, expected in [("empty", [], 0), ("ordinary", [2,4,6], 4),
                                ("negative", [-2,-4], -3), ("fractional", [1,2], 1.5)]:
    try:
        actual = module.average(values)
        checks.append({"name": name, "passed": actual == expected,
                       "expected": expected, "actual": actual})
    except Exception as exc:
        checks.append({"name": name, "passed": False, "error": type(exc).__name__})
print(json.dumps(checks))
'''
RUBRIC = {
    "name": "average-fix-demo", "version": "1.0", "pass_threshold": 0.8,
    "dimensions": [
        {"name": "correctness", "scale": 5, "weight": 0.5, "levels": {
            1: "Does not implement the requested behavior; the empty-input check fails.",
            2: "Attempts the fix but fails multiple required checks.",
            3: "Handles empty input but fails an existing numeric behavior check.",
            4: "Passes all four checks but leaves a correctness concern visible in the patch.",
            5: "Passes empty, ordinary, negative, and fractional checks with no visible correctness issue."}},
        {"name": "instruction_following", "scale": 3, "weight": 0.3, "levels": {
            1: "Violates the explicit request to change only stats.py, or does not implement the fix.",
            2: "Changes only stats.py but introduces unrelated changes inside that file.",
            3: "Changes only stats.py and only what is necessary for the requested fix."}},
        {"name": "maintainability", "scale": 3, "weight": 0.2, "levels": {
            1: "Uses an opaque workaround or introduces unnecessary complexity.",
            2: "Implementation is understandable but has avoidable complexity.",
            3: "Implementation is short, readable, and straightforward to maintain."}},
    ],
}


class Recorder:
    def __init__(self, directory, model):
        self.start = time.monotonic()
        self.cast = (directory / "demo.cast").open("w", encoding="utf-8")
        self.cast.write(json.dumps({"version": 2, "width": 100, "height": 30,
                                   "timestamp": int(time.time()),
                                   "title": "eval-layer: live Jev evaluation",
                                   "env": {"TERM": "xterm-256color"}}) + "\n")
        self.scenes = []
        self.model = model

    def line(self, text="", *, kind="line"):
        print(text, flush=True)
        now = round(time.monotonic() - self.start, 4)
        self.cast.write(json.dumps([now, "o", text + "\r\n"]) + "\n")
        self.cast.flush()
        self.scenes.append({"at": now, "kind": kind, "text": text})

    def close(self, directory):
        self.cast.close()
        (directory / "recording.json").write_text(json.dumps({
            "title": "Three fixes. One rubric. Live Jev scoring.",
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "model": self.model,
            "note": "Authored demo candidates; live Jev scores; real local checks. "
                    "Video playback is paced for readability. Raw terminal timestamps are retained in demo.cast.",
            "events": self.scenes,
        }, indent=2), encoding="utf-8")


def prepare(directory):
    rows = []
    candidates = [("broken", "Broken baseline", BASE, False),
                  ("broad", "Fix + unrelated change", FIX, True),
                  ("clean", "Clean, minimal fix", FIX, False)]
    for key, label, code, unrelated in candidates:
        with tempfile.TemporaryDirectory(prefix="jev-demo-check-") as tmp:
            root = Path(tmp)
            (root / "stats.py").write_text(code)
            (root / "settings.py").write_text("LOG_LEVEL = 'DEBUG'\n" if unrelated else "LOG_LEVEL = 'INFO'\n")
            completed = subprocess.run([sys.executable, "-c", CHECK], cwd=root,
                                       capture_output=True, text=True, timeout=10, check=True)
            checks = json.loads(completed.stdout)
        diff = ''.join(difflib.unified_diff(BASE.splitlines(True), code.splitlines(True),
                                            fromfile="a/stats.py", tofile="b/stats.py"))
        if unrelated:
            diff += "--- a/settings.py\n+++ b/settings.py\n@@ -1 +1 @@\n-LOG_LEVEL = 'INFO'\n+LOG_LEVEL = 'DEBUG'\n"
        paths = (["stats.py"] if code != BASE else []) + (["settings.py"] if unrelated else [])
        rows.append({"subject": key, "case_id": "average-empty", "trial": 1,
                     "label": label, "input": TASK,
                     "context": "Small pure Python function. Candidate code is an authored demo example.",
                     "expected_output": "Empty list returns zero; existing numeric averages are preserved; only stats.py changes.",
                     "agent_output": {"code": code, "diff": diff or "No change from baseline."},
                     "evidence": {"checks": checks, "changed_paths": paths},
                     "gates": {"functional_checks": all(c["passed"] for c in checks),
                               "allowed_paths": all(p == "stats.py" for p in paths)},
                     "judge": None})
    (directory / "rubric.json").write_text(json.dumps(RUBRIC, indent=2), encoding="utf-8")
    (directory / "saved.jsonl").write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding="utf-8")
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new artifact directory")
    parser.add_argument("--model", default="jev-1.13.0")
    parser.add_argument("--prepare-only", action="store_true", help="prepare examples and checks, without Jev")
    args = parser.parse_args(argv)
    if not args.prepare_only and not os.environ.get("TYPESAFE_API_KEY", "").strip():
        parser.error("Set TYPESAFE_API_KEY before the live demo; it is never recorded.")
    args.output.mkdir(parents=True, exist_ok=False)
    rows = prepare(args.output)
    if args.prepare_only:
        print("Prepared three demo candidates and executed four independent checks per candidate.")
        return 0
    recorder = Recorder(args.output, args.model)
    results = []
    try:
        recorder.line("eval-layer  /  LIVE JEV DEMO", kind="title")
        recorder.line("Authored code candidates. Live judge scores. Real local checks.")
        recorder.line("TASK: " + TASK, kind="task")
        recorder.line("Rubric: correctness 50% | instructions 30% | maintainability 20%")
        recorder.line("Passing score: 0.80 AND all required checks pass.")
        recorder.line("Model: " + args.model)
        for row in rows:
            recorder.line("", kind="break")
            recorder.line(row["label"].upper(), kind="candidate")
            for line in row["agent_output"]["code"].splitlines():
                recorder.line("  " + line, kind="code")
            passed = sum(c["passed"] for c in row["evidence"]["checks"])
            recorder.line(f"Independent tests: {passed}/4 pass")
            recorder.line("Change scope: " + ("PASS" if row["gates"]["allowed_paths"] else "FAIL: settings.py also changed"))
            recorder.line("Calling Jev...", kind="request")
            # This calls the same live replay path shipped with the skill.
            judged = replay_rows([row], RUBRIC, judge_backend="jev", judge_model=args.model)[0]
            results.append(judged)
            judge = judged["judge"]
            if not judge or "error" in judge:
                recorder.line("Judge failure: " + (judge or {}).get("error", "missing"), kind="error")
                continue
            score = judged["rejudge"]["weighted_score"]
            recorder.line(f"Jev score: {score:.3f} / 1.000  |  verdict: {'PASS' if judged['passed'] else 'FAIL'}", kind="result")
            for dim in RUBRIC["dimensions"]:
                name = dim["name"]
                recorder.line(f"  {name}: {judge['scores'][name]:.2f}/{dim['scale']}  confidence {judge['details'][name]['confidence']:.2f}")
            recorder.line(f"Returned model: {judge['resolved_model_id']} | {judge['latency_ms']} ms")
        recorder.line("", kind="break")
        recorder.line("RESULTS", kind="summary")
        for row in results:
            score = row["rejudge"]["weighted_score"]
            text = "unscored" if score is None else f"{score:.3f}"
            recorder.line(f"{row['label']:<25} {text:>8}   {'PASS' if row['passed'] else 'FAIL'}")
        recorder.line("Quality scores complement deterministic checks.", kind="takeaway")
        recorder.line("A failed required check always blocks a pass.")
        recorder.line("Confidence is not a probability of correctness.")
        recorder.line("No model-generated explanations are invented.")
        recorder.line("github.com/erezweinstein5/eval-layer", kind="footer")
    finally:
        recorder.close(args.output)
        (args.output / "results.jsonl").write_text(''.join(json.dumps(r, allow_nan=False) + '\n' for r in results), encoding="utf-8")
        summary = summarize(results, RUBRIC, judge_backend="jev", judge_model=args.model)
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
        (args.output / "report.md").write_text(render_markdown(summary), encoding="utf-8")
    return 1 if any(not r.get("judge") or "error" in r["judge"] for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
