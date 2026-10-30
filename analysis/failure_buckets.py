"""Bucket test250 rollouts by first failure mode (context out, invalid tool, killed command, background fork, no submit)."""
import json, os, re, sys, collections
TR = "/home/ubuntu/STT/LLMPerfLab/MiMo_rl/SmolDataEnvs/runs/trials_data_agent"
NOTOOL = re.compile(r"No such tool available|tool_use_error>Error: No such tool|unknown tool|is not a valid tool|Model tried to call unavailable tool", re.I)
KILL = re.compile(r"exceeding timeout \d+ ?ms", re.I)
ORDER = ["context out", "invalid tool", "cmd killed (timeout ms)", "background Agent fork", "no answer written", "clean"]
def audit(label):
    rows = [json.loads(l) for l in open(f"runs/test250_{label}_r1/rollouts.jsonl")]
    b = collections.defaultdict(list); bad = collections.Counter()
    for r in rows:
        p = os.path.join(TR, r["trial"], "agent", "trajectory.json")
        steps = json.load(open(p))["steps"] if os.path.exists(p) else []
        f = set()
        for s in steps:
            obs = json.dumps(s.get("observation") or "")
            for m in NOTOOL.finditer(obs): f.add("invalid tool")
            if KILL.search(obs): f.add("cmd killed (timeout ms)")
            if "Fork started" in obs: f.add("background Agent fork")
            if NOTOOL.search(obs):
                for tc in s.get("tool_calls") or []: bad[tc["function_name"]] += 1
        if r.get("context_exhausted"): f.add("context out")
        if not r.get("submitted"): f.add("no answer written")
        cat = next((c for c in ORDER if c in f), "clean")
        b[cat].append(r.get("correctness") or 0)
    n = len(rows); tot = sum(sum(v) for v in b.values())
    clean = sum(b["clean"]) / len(b["clean"]) if b["clean"] else 0
    print(f"{label}: pass {tot/n:.3f}  (clean-rollout pass {clean:.2f})")
    for c in ORDER:
        v = b.get(c, [])
        if v: print(f"  {c:26s} {len(v):4d} rollouts  pass {sum(v)/len(v):.2f}  pts lost vs clean {100*(clean*len(v)-sum(v))/n:5.1f}")
    if bad: print("  invalid tool names:", bad.most_common(8))
for lab in sys.argv[1:]: audit(lab)
