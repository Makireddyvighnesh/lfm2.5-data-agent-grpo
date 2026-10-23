"""Write all rollouts of one training task at one update as plain text (prompt, every turn, tool results as the model
saw them after the 4,000-char proxy cap, grader line, reward). usage:
python3 analysis/train_rollouts_to_txt.py runs/<run> <update> <task> <out.txt>"""
import gzip, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rollouts_to_txt import TRIALS, as_seen, text_of, CAP

run, step, task, out = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3], Path(sys.argv[4])
W = []
for line in open(run / "rollouts.jsonl"):
    r = json.loads(line)
    if r.get("step") != step or r["task"] != task:
        continue
    msgs = json.load(gzip.open(run / r["capture_file"]))["result"]["conversations"][0]["messages"]
    W += ["#" * 100, f"# rollout {r['rollout_id']} | task {task} | update {step}",
          f"# correct={r.get('correct')} reward={r.get('reward'):.3f} submitted={r.get('submitted')} tool_calls={r.get('tool_calls')} "
          f"agent_seconds={r.get('agent_seconds')} generated_tokens={r.get('generated_tokens')} peak_context={r.get('peak_context')} "
          f"ended_on_call_text={r.get('ended_on_call_text')}", "#" * 100]
    n = 0
    for m in msgs:
        role, body = m.get("role"), text_of(m.get("content"))
        if role == "system":
            W.append("\n===== SYSTEM PROMPT (Pi harness, omitted) =====")
        elif role == "user":
            W.append(f"\n===== USER / TASK PROMPT =====\n{body}")
        elif role == "assistant":
            n += 1
            W.append(f"\n===== MODEL TURN {n} =====\n" + (body.strip() or "(no natural-language text)"))
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function", {})
                try:
                    a = json.loads(fn.get("arguments", ""))
                except Exception:
                    a = fn.get("arguments", "")
                W.append(f"\n>>> TOOL CALL: {fn.get('name')}\n" + (a["command"] if isinstance(a, dict) and "command" in a else json.dumps(a)))
        elif role == "tool":
            W.append(f"\n----- TOOL RESULT ({len(body)} chars{'; model saw 4,000-char cut' if len(body) > CAP else ''}) -----\n{as_seen(body)}")
    g = TRIALS / str(r.get("trial")) / "verifier" / "test-stdout.txt"
    W += ["\n===== GRADER =====", g.read_text().strip() if g.exists() else "(no grader log)", "\n"]
out.write_text("\n".join(W)); print("wrote", out, f"{out.stat().st_size // 1024} KB")
