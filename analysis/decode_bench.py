"""Decode throughput of the serving vLLM under the training load shape: 64 concurrent mid-rollout requests.

Replays captured requests (prompt 4-9k tokens, from runs/p2_A_tool archives) concurrently with a fixed
256-token output (ignore_eos), and reports wall time, aggregate output tokens/s and mean time to first token.
usage: python analysis/decode_bench.py [--n 64] [--model LiquidAI/LFM2.5-2.6B]
"""
import argparse
import glob
import gzip
import json
import time
from concurrent.futures import ThreadPoolExecutor

import httpx

p = argparse.ArgumentParser()
p.add_argument("--n", type=int, default=64)
p.add_argument("--model", default="LiquidAI/LFM2.5-2.6B")
p.add_argument("--run", default="runs/p2_A_tool")
a = p.parse_args()

reqs = []
for f in sorted(glob.glob(f"{a.run}/rollouts/*.json.gz")):
    d = json.load(gzip.open(f))
    for t in d["result"]["turns"]:
        if 4000 <= len(t["prompt_token_ids"]) <= 9000:
            reqs.append({"messages": t["request_messages"], "tools": t["request_tools"]})
            break
    if len(reqs) >= a.n:
        break
client = httpx.Client(timeout=900)


def one(body):
    t0 = time.time()
    r = client.post("http://127.0.0.1:8000/v1/chat/completions", json={
        "model": a.model, **body, "temperature": 1.0, "max_tokens": 256, "ignore_eos": True, "stream": False})
    r.raise_for_status()
    u = r.json()["usage"]
    return time.time() - t0, u["completion_tokens"], u["prompt_tokens"]


t0 = time.time()
with ThreadPoolExecutor(len(reqs)) as pool:
    res = list(pool.map(one, reqs))
wall = time.time() - t0
out = sum(r[1] for r in res)
print(json.dumps({"requests": len(res), "mean_prompt_tokens": round(sum(r[2] for r in res) / len(res)),
                  "wall_s": round(wall, 1), "output_tok_per_s": round(out / wall), "mean_request_s": round(sum(r[0] for r in res) / len(res), 1)}))
