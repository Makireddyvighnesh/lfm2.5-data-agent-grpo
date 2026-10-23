"""Print one rollout as readable text: question, gold vs submitted answer, metrics, then every turn.

Works on eval rollouts (runs/val118_*/rollouts/*.json) and training rollouts
(runs/p3_train1000_tool/rollouts/*.json.gz). Gold and submitted answers come from the trial's grader log.
usage: python3 analysis/dump_rollout.py <rollout file> [--max-tool 1500] [--max-text 3000]
"""
import argparse
import gzip
import json
from pathlib import Path

TRIALS = Path(__file__).resolve().parents[2] / "SmolDataEnvs" / "runs" / "trials_data_agent"


def load(path):
    p = Path(path)
    raw = gzip.open(p).read() if p.suffix == ".gz" else p.read_bytes()
    d = json.loads(raw)
    if "result" in d:  # training rollout
        m = d["metrics"]
        msgs = d["result"]["conversations"][0]["messages"]
        trial = d["result"].get("trial_name") or m.get("trial")
        info = {k: m.get(k) for k in ("task", "difficulty", "policy_version", "correct", "reward", "submitted",
                                       "context_exhausted", "truncated", "tool_calls", "model_calls",
                                       "generated_tokens", "peak_context", "seconds")}
    else:  # eval rollout
        msgs = d["trajectory"]["messages"]
        trial = d.get("trial")
        info = {k: d.get(k) for k in ("task", "difficulty", "label", "correctness", "submitted", "context_exhausted",
                                       "budget_stops", "truncated", "tool_calls", "model_calls", "generated_tokens",
                                       "peak_context", "seconds")}
    return msgs, trial, info


def text_of(content):
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return "\n".join(c.get("text", "") for c in content if isinstance(c, dict))


def clip(s, n):
    return s if len(s) <= n else s[: n // 2] + f"\n... [{len(s) - n} chars cut] ...\n" + s[-n // 2:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--max-tool", type=int, default=1500)
    ap.add_argument("--max-text", type=int, default=3000)
    a = ap.parse_args()
    msgs, trial, info = load(a.path)
    print("== ROLLOUT", a.path)
    print("== METRICS", json.dumps(info))
    grader = TRIALS / str(trial) / "verifier" / "test-stdout.txt"
    print("== GRADER", grader.read_text().strip() if grader.exists() else f"(no grader log for trial {trial})")
    for i, m in enumerate(msgs):
        role = m.get("role")
        if role == "system":
            print(f"\n--- [{i}] system prompt ({len(text_of(m.get('content')))} chars, omitted)")
            continue
        body = text_of(m.get("content"))
        if role == "tool":
            print(f"\n--- [{i}] TOOL RESULT ({len(body)} chars)\n{clip(body, a.max_tool)}")
            continue
        print(f"\n--- [{i}] {role.upper()}")
        if body.strip():
            print(clip(body, a.max_text))
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function", {})
            args = fn.get("arguments", "")
            try:
                args = json.loads(args)
                args = args.get("command", args) if isinstance(args, dict) else args
            except Exception:
                pass
            print(f">> CALL {fn.get('name')}: {clip(args if isinstance(args, str) else json.dumps(args), a.max_text)}")


if __name__ == "__main__":
    main()
