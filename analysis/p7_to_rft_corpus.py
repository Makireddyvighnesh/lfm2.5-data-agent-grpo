"""Convert correct p7 TRAINING rollouts into the evaluate.py-style rollouts/*.json that pirl/self_rft.py reads.
Each rollout becomes one 'call' = its chained training sequence (exactly what GRPO trained: input_ids +
completion_mask over the model's own sampled tokens). Per task, the top-K correct rollouts by reward (fast, short,
few calls) are kept. usage: python3 analysis/p7_to_rft_corpus.py runs/p7_think_bash <out_dir> [K]"""
import gzip, json, sys, collections
from pathlib import Path
src, out, K = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 3
(out / "rollouts").mkdir(parents=True, exist_ok=True)
by = collections.defaultdict(list)
for f in (src / "rollouts").glob("*.json.gz"):
    d = json.load(gzip.open(f)); m = d["metrics"]
    if m.get("correct") == 1 and m.get("atif") == "match" and not m.get("context_exhausted") \
            and len(d.get("training_sequences") or []) == 1:
        by[m["task"]].append((m["reward"], f))
n = 0
for task, rs in by.items():
    for reward, f in sorted(rs, key=lambda x: -x[0])[:K]:
        d = json.load(gzip.open(f)); m = d["metrics"]; s = d["training_sequences"][0]
        row = {"task": task, "rollout_id": m["rollout_id"], "correctness": 1.0, "trainable": True, "atif": "match",
               "context_exhausted": False, "peak_context": m["peak_context"], "reward": reward,
               "policy_version": m.get("policy_version"),
               "trajectory": {"calls": [{"prompt_token_ids": [], "completion_token_ids": s["input_ids"],
                                         "loss_mask": s["completion_mask"]}]}}
        (out / "rollouts" / f"{m['rollout_id']}.json").write_text(json.dumps(row)); n += 1
print("tasks", len(by), "rollouts written", n)
