"""Stratified root sampling between OpenEnv's capture proxy and vLLM (exploration of the first action).

Every rollout of a task starts from the same Pi request, and the first action decides most outcomes:
a whole-file `read` overflows the 12k prompt window, while `bash`/`read(limit)` inspection fits
(about 5% of first turns at T=1.0, measured by analysis/first_turn_probe.py). GRPO groups of 8 rarely
contain one, so the policy gets almost no signal about the decision that matters.

For the FIRST agent turn of each rollout (system prompt + tools, no assistant message yet) this proxy:
  1. draws one pool of N candidate first turns per (policy version, prompt) with a single vLLM call
     (n=N, the requested T=1.0 sampling, no tools executed), shared by the whole GRPO group;
  2. splits the pool into strata: "alt" (first tool call is bash, or read with a limit) and "common";
  3. assigns the group's G rollouts: k_alt = max(--alt-slots, round(G * p_alt)) (capped at G-1 while common
     candidates exist) drawn uniformly from "alt", the rest uniformly from "common"; and returns each rollout one untouched candidate (its own token ids and
     logprobs from vLLM), so the capture records exactly what was sampled;
  4. logs the stratified importance weight W = p_stratum * G / n_stratum_slots for that candidate,
     keyed by (policy model, sha1 of prompt ids, sha1 of completion ids), for grpo.py --root-weights.
With W, sum_i W_i f(a_i) / G is an unbiased estimate of E_mu[f] (stratified sampling), so the policy
gradient stays correct while rare first actions appear in (almost) every group. Later turns pass
through unchanged. No prompt, tool or observation is altered.

usage: python root_proxy.py --port 8002 --upstream http://127.0.0.1:8000 --pool 128 --group 8 --alt-slots 2
"""

import argparse
import asyncio
import copy
import hashlib
import json
import random
import time
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, Request, Response

HOP = {"host", "content-length", "transfer-encoding", "connection", "accept-encoding"}


def first_action(choice):
    calls = (choice.get("message") or {}).get("tool_calls") or []
    if not calls:
        return "no_tool"
    fn = calls[0].get("function") or {}
    name = fn.get("name")
    try:
        args = json.loads(fn.get("arguments") or "{}")
    except ValueError:
        return f"{name}:bad_args"
    if name == "read":
        return "read_limit" if "limit" in args else "read_full"
    return name


ALT = {"bash", "read_limit"}


def ids_key(token_ids):
    return hashlib.sha1(json.dumps(list(token_ids)).encode()).hexdigest()


def is_agent_first_turn(payload):
    msgs = payload.get("messages") or []
    return bool(payload.get("tools")) and bool(msgs) and msgs[0].get("role") == "system" \
        and not any(m.get("role") == "assistant" for m in msgs)


def build_app(a):
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    client = httpx.AsyncClient(base_url=a.upstream, timeout=httpx.Timeout(1800.0))
    log = Path(a.log).open("a")
    rng = random.Random(a.seed)
    pools = {}  # key -> asyncio.Task resolving to pool dict
    stats = {"pools": 0, "assigned": 0, "overflow": 0, "later_turns": 0, "pool_s": 0.0}

    async def make_pool(body, headers):
        t0 = time.time()
        req = dict(body, n=a.pool)
        r = await client.post("/v1/chat/completions", json=req, headers=headers)
        r.raise_for_status()
        resp = r.json()
        choices = resp["choices"]
        classes = [first_action(c) for c in choices]
        alt = [i for i, c in enumerate(classes) if c in ALT]
        common = [i for i, c in enumerate(classes) if c not in ALT]
        n, g = len(choices), a.group
        if not alt:
            k_alt = 0
        elif not common:
            k_alt = g
        else:  # oversample rare openings, but follow the policy once it prefers them; keep >=1 common control
            k_alt = min(len(alt), g - 1, max(a.alt_slots, round(g * len(alt) / n)))
        k_common = g - k_alt
        picks = [(i, "alt") for i in rng.sample(alt, k_alt)]
        pool_common = common or alt
        picks += [(i, "common") for i in (rng.sample(pool_common, k_common) if len(pool_common) >= k_common
                                         else rng.choices(pool_common, k=k_common))]
        rng.shuffle(picks)
        weight = {"alt": (len(alt) / n) * g / k_alt if k_alt else 0.0,
                  "common": (len(common) / n) * g / k_common if k_common and common else 1.0}
        stats["pools"] += 1
        stats["pool_s"] += time.time() - t0
        prompt_sha1 = ids_key(resp.get("prompt_token_ids") or [])
        return {"resp": resp, "classes": classes, "queue": picks, "weight": weight, "prompt_sha1": prompt_sha1,
                "p_alt": len(alt) / n, "n": n, "k_alt": k_alt}

    @app.get("/root_stats")
    async def root_stats():
        return {**stats, "open_pools": len(pools)}

    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
    async def forward(path: str, request: Request):
        body = await request.body()
        headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP}
        if request.method == "POST" and path.rstrip("/").endswith("chat/completions") and body:
            payload = json.loads(body)
            if is_agent_first_turn(payload) and not payload.get("stream") and payload.get("n", 1) == 1:
                key = hashlib.sha1(json.dumps([payload.get("model"), payload.get("messages"),
                                               payload.get("tools")], sort_keys=True).encode()).hexdigest()
                if key not in pools:
                    pools[key] = asyncio.ensure_future(make_pool(payload, headers))
                try:
                    pool = await pools[key]
                except Exception:
                    pools.pop(key, None)  # never cache a failed pool (e.g. engine asleep); the rollout retries
                    raise
                if pool["queue"]:
                    idx, stratum = pool["queue"].pop()
                    w = pool["weight"][stratum]
                    stats["assigned"] += 1
                    if not pool["queue"]:
                        pools.pop(key, None)  # group fully assigned: free the ~2 MB pool
                else:  # more first turns than the group size: plain on-policy draw
                    idx, stratum, w = rng.randrange(pool["n"]), "overflow", 1.0
                    stats["overflow"] += 1
                choice = copy.deepcopy(pool["resp"]["choices"][idx])
                choice["index"] = 0
                out = {k: v for k, v in pool["resp"].items() if k != "choices"}
                out["choices"] = [choice]
                ntok = len(choice.get("token_ids") or [])
                if isinstance(out.get("usage"), dict):
                    out["usage"] = dict(out["usage"], completion_tokens=ntok,
                                        total_tokens=out["usage"].get("prompt_tokens", 0) + ntok)
                log.write(json.dumps({"t": time.time(), "key": key, "model": payload.get("model"),
                                      "prompt_sha1": pool["prompt_sha1"],
                                      "ids_sha1": ids_key(choice.get("token_ids") or []),
                                      "class": pool["classes"][idx], "stratum": stratum, "weight": w,
                                      "p_alt": pool["p_alt"], "pool_n": pool["n"], "k_alt": pool["k_alt"],
                                      "group": a.group}) + "\n")
                log.flush()
                return Response(content=json.dumps(out), status_code=200, media_type="application/json")
            stats["later_turns"] += 1
        resp = await client.request(request.method, "/" + path, params=request.query_params,
                                    content=body, headers=headers)
        out_headers = {k: v for k, v in resp.headers.items() if k.lower() not in HOP}
        return Response(content=resp.content, status_code=resp.status_code, headers=out_headers)

    return app


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", type=int, default=8002)
    p.add_argument("--upstream", default="http://127.0.0.1:8000")
    p.add_argument("--pool", type=int, default=128)
    p.add_argument("--group", type=int, default=8)
    p.add_argument("--alt-slots", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log", default=str(Path(__file__).resolve().parent / "runs" / "root_assign.jsonl"))
    a = p.parse_args()
    uvicorn.run(build_app(a), host="127.0.0.1", port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
