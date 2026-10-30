"""Audit tool calls in Harbor ATIF trajectories of one test250 run: timeout args, killed commands, tool errors."""
import json, re, sys, os, collections
TR = "/home/ubuntu/STT/LLMPerfLab/MiMo_rl/SmolDataEnvs/runs/trials_data_agent"
KILL = re.compile(r"exceeding timeout (\d+) ?ms|timed out after|Command timed out|terminated command", re.I)
def audit(label):
    rows = [json.loads(l) for l in open(f"runs/test250_{label}_r1/rollouts.jsonl")]
    c = collections.Counter(); per = []
    for r in rows:
        p = os.path.join(TR, r["trial"], "agent", "trajectory.json")
        if not os.path.exists(p): c["no_traj"] += 1; continue
        t = json.load(open(p)); killed = 0; small = 0; calls = 0; errs = 0
        for s in t["steps"]:
            if s.get("source") != "agent": continue
            obs = json.dumps(s.get("observation") or "")
            for tc in s.get("tool_calls") or []:
                calls += 1; a = tc.get("arguments") or {}
                to = a.get("timeout") if isinstance(a, dict) else None
                if isinstance(to, (int, float)): c["calls_with_timeout"] += 1; small += to < 1000
            killed += len(KILL.findall(obs)); errs += obs.lower().count("error")
        c["calls"] += calls; c["small_timeout"] += small; c["killed"] += killed
        per.append((r.get("correctness") or 0, killed > 0))
    hit = [x for x in per if x[1]]; miss = [x for x in per if not x[1]]
    pas = lambda v: sum(x[0] for x in v) / len(v) if v else float("nan")
    print(f"{label:28s} calls {c['calls']:5d} timeout-arg {c['calls_with_timeout']:4d} (<1000: {c['small_timeout']:4d}) killed {c['killed']:4d} | rollouts with a kill {len(hit):3d} pass {pas(hit):.2f} vs {pas(miss):.2f} without")
for lab in sys.argv[1:]: audit(lab)
