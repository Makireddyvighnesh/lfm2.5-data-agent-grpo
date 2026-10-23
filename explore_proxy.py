"""First-turn exploration proxy between OpenEnv's capture proxy and vLLM.

Every rollout of a task starts from the same prompt, and LFM picks a whole-file `read` there 98.6% of the
time; Python inspection via `bash` (1.2%) solves 33% vs 4%. Raising the sampling temperature for the
FIRST model call only (a chat request with no assistant message yet) makes the rare first actions
appear in GRPO groups. Later calls are untouched.

Why this stays a correct policy gradient: vLLM runs with `--logprobs-mode processed_logprobs`, so the
logprobs it returns (and the capture proxy records as the behaviour policy) are those of the
temperature-scaled distribution that actually sampled the tokens. The trainer scores the same tokens
at T=1 and, with `--is-correction tis`, weights each token by min(pi_T1 / mu_Tfirst, cap): ordinary
truncated importance sampling. No prompt, tool or observation is changed.

usage: python explore_proxy.py --port 8002 --upstream http://127.0.0.1:8000 --first-temperature 1.5
Point grpo.py at it with --vllm-url http://127.0.0.1:8002 (keep evaluation on :8000, plain T=1).
"""

import argparse
import json
import time
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, Request, Response

HOP = {"host", "content-length", "transfer-encoding", "connection", "accept-encoding"}


def build_app(upstream, first_temperature, log_path):
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    client = httpx.AsyncClient(base_url=upstream, timeout=httpx.Timeout(1800.0))
    log = Path(log_path).open("a") if log_path else None
    counts = {"first": 0, "later": 0}

    @app.get("/explore_stats")
    async def stats():
        return {"first_temperature": first_temperature, **counts}

    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
    async def forward(path: str, request: Request):
        body = await request.body()
        if request.method == "POST" and path.rstrip("/").endswith("chat/completions") and body:
            payload = json.loads(body)
            first = not any(m.get("role") == "assistant" for m in payload.get("messages") or [])
            counts["first" if first else "later"] += 1
            if first and first_temperature is not None:
                original = payload.get("temperature")
                payload["temperature"] = first_temperature
                body = json.dumps(payload).encode()
                if log:
                    log.write(json.dumps({"t": time.time(), "first": True, "temperature": first_temperature,
                                          "requested": original, "stream": payload.get("stream")}) + "\n")
                    log.flush()
        headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP}
        resp = await client.request(request.method, "/" + path, params=request.query_params,
                                    content=body, headers=headers)
        out_headers = {k: v for k, v in resp.headers.items() if k.lower() not in HOP}
        return Response(content=resp.content, status_code=resp.status_code, headers=out_headers)

    return app


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", type=int, default=8002)
    p.add_argument("--upstream", default="http://127.0.0.1:8000")
    p.add_argument("--first-temperature", type=float, default=1.5)
    p.add_argument("--log", default=str(Path(__file__).resolve().parent / "runs" / "explore_proxy.jsonl"))
    a = p.parse_args()
    uvicorn.run(build_app(a.upstream, a.first_temperature, a.log), host="127.0.0.1", port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
