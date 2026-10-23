"""How does the model open a task? Replays pi's exact first request (system prompt, tool schemas, task
instruction) to vLLM n times per task and temperature, and classifies the first tool call.
No tools are executed, so this is cheap (one short completion per candidate).

usage: python analysis/first_turn_probe.py --capture runs/grpo_r1/rollouts/<any>.json.gz \
         --tasks manifests/train100.json --n 32 --temperatures 1.0 1.3 1.5 --out analysis/first_turn_probe.json
"""
import argparse
import gzip
import json
import re
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

SMOL = Path(__file__).resolve().parents[2] / "SmolDataEnvs"
DUMP = re.compile(r"\bcat\s+\S*\.(csv|tsv|json|txt)|to_string\(|print\(df\)|pd\.set_option\(.display\.max_rows", re.I)


def classify(choice):
    calls = (choice.get("message") or {}).get("tool_calls") or []
    if not calls:
        return "no_tool", None
    fn = calls[0]["function"]
    name, raw = fn["name"], fn.get("arguments") or "{}"
    try:
        args = json.loads(raw)
    except ValueError:
        return f"{name}:bad_args", raw
    if name == "read":
        return ("read_limit" if "limit" in args else "read_full"), raw
    if name == "bash":
        cmd = args.get("command", "")
        if DUMP.search(cmd):
            return "bash_dump", cmd
        ok = subprocess.run(["bash", "-n"], input=cmd, text=True, capture_output=True).returncode == 0
        return ("bash" if ok else "bash:syntax_error"), cmd
    return name, raw


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--capture", required=True)
    p.add_argument("--tasks", required=True)
    p.add_argument("--n", type=int, default=32)
    p.add_argument("--temperatures", type=float, nargs="+", default=[1.0, 1.3, 1.5])
    p.add_argument("--model", default="LiquidAI/LFM2.5-2.6B")
    p.add_argument("--url", default="http://127.0.0.1:8000")
    p.add_argument("--out", required=True)
    p.add_argument("--task-dir", default=str(SMOL / "prepared/datasets/train/tasks"))
    a = p.parse_args()

    turn = json.load(gzip.open(a.capture))["result"]["turns"][0]
    system, tools = turn["request_messages"][0], turn["request_tools"]
    names = json.loads(Path(a.tasks).read_text())
    client = httpx.Client(timeout=600)

    def run(job):
        name, temp = job
        instr = (Path(a.task_dir) / name / "instruction.md").read_text()
        body = {"model": a.model, "messages": [system, {"role": "user", "content": [{"type": "text", "text": instr}]}],
                "tools": tools, "tool_choice": "auto", "n": a.n, "temperature": temp, "top_p": 1.0,
                "max_tokens": 1024}
        r = client.post(f"{a.url}/v1/chat/completions", json=body)
        r.raise_for_status()
        out = [classify(c) for c in r.json()["choices"]]
        return {"task": name, "temperature": temp, "classes": [c for c, _ in out],
                "examples": {c: x for c, x in reversed(out) if c not in ("read_full",)}}

    jobs = [(n, t) for t in a.temperatures for n in names]
    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(run, jobs))
    Path(a.out).write_text(json.dumps(results, indent=1))
    for t in a.temperatures:
        rs = [r for r in results if r["temperature"] == t]
        c = Counter(x for r in rs for x in r["classes"]); n = sum(c.values())
        tasks_with_alt = sum(any(x in ("bash", "read_limit") for x in r["classes"]) for r in rs)
        print(f"T={t}: " + ", ".join(f"{k} {v / n:.1%}" for k, v in c.most_common())
              + f" | tasks with >=1 bash/read_limit candidate: {tasks_with_alt}/{len(rs)}")


if __name__ == "__main__":
    main()
