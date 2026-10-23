"""Write eval rollouts as plain text: full system prompt, task prompt, every model turn (text + tool calls), every tool
result exactly as the model saw it (proxy cap: 4,000 chars = head 2,800 + tail 1,200), then the grader result.
usage: python3 analysis/rollouts_to_txt.py <eval run dir> <out.txt> <label>=<task>--r<k> [...]
"""
import json
import sys
from pathlib import Path

TRIALS = Path(__file__).resolve().parents[2] / "SmolDataEnvs" / "runs" / "trials_data_agent"
CAP, HEAD = 4000, 2800


def text_of(c):
    if c is None:
        return ""
    return c if isinstance(c, str) else "\n".join(x.get("text", "") for x in c if isinstance(x, dict))


def as_seen(s):
    if len(s) <= CAP:
        return s
    return f"{s[:HEAD]}\n[... {len(s) - CAP} characters omitted ...]\n{s[-(CAP - HEAD):]}"


def main():
    run, out = Path(sys.argv[1]), Path(sys.argv[2])
    rows = {}
    for line in open(run / "rollouts.jsonl"):
        r = json.loads(line)
        rows[(r["task"], r["rollout_id"].rsplit("-r", 1)[1])] = r
    W = []
    for spec in sys.argv[3:]:
        label, key = spec.split("=", 1)
        task, k = key.split("--r")
        r = rows[(task, k)]
        msgs = r["trajectory"]["messages"]
        W.append("#" * 100)
        W.append(f"# {label}")
        W.append(f"# task {task} (attempt r{k}) | run {run.name}")
        W.append(f"# correct={r.get('correctness')} submitted={r.get('submitted')} context_exhausted={r.get('context_exhausted')}"
                 f" truncated={r.get('truncated')} tool_calls={r.get('tool_calls')} peak_context={r.get('peak_context')}"
                 f" tokens seconds={r.get('seconds'):.0f}")
        W.append("#" * 100)
        step = 0
        for m in msgs:
            role, body = m.get("role"), text_of(m.get("content"))
            if role == "system":
                W.append(f"\n===== SYSTEM PROMPT ({len(body)} chars) =====\n{body}")
            elif role == "user":
                W.append(f"\n===== USER / TASK PROMPT =====\n{body}")
            elif role == "assistant":
                step += 1
                tag = "   <<< NATURAL-LANGUAGE TEXT" if body.strip() else ""
                W.append(f"\n===== MODEL TURN {step} ====={tag}")
                W.append(body.strip() if body.strip() else "(no natural-language text)")
                for tc in m.get("tool_calls") or []:
                    fn = tc.get("function", {})
                    try:
                        args = json.loads(fn.get("arguments", ""))
                    except Exception:
                        args = fn.get("arguments", "")
                    if isinstance(args, dict) and "command" in args:
                        W.append(f"\n>>> TOOL CALL: {fn.get('name')}\n{args['command']}")
                    else:
                        W.append(f"\n>>> TOOL CALL: {fn.get('name')} {json.dumps(args)}")
            elif role == "tool":
                note = f"raw {len(body)} chars; model saw the 4,000-char cut" if len(body) > CAP else f"{len(body)} chars"
                W.append(f"\n----- TOOL RESULT ({note}) -----\n{as_seen(body)}")
        g = TRIALS / str(r.get("trial")) / "verifier" / "test-stdout.txt"
        W.append("\n===== RESULT (grader) =====")
        W.append(g.read_text().strip() if g.exists() else "(no grader log)")
        W.append("\n\n")
    out.write_text("\n".join(W))
    print("wrote", out, f"{out.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
