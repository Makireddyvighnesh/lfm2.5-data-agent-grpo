"""Local OpenAI-compatible shim in front of Tinker's OpenAI endpoint (evaluation tier).

Harbor's capture proxy forwards only temperature/top_p/top_k upstream, and Qwen3.5 on Tinker thinks
by default, so this shim injects `reasoning_effort: false` and keeps the Tinker key server-side
(read from .env as TINKER_API_KEY; never logged). Tinker's endpoint returns logprobs but no token
ids, so rollouts through it are eval-only. Training needs a SamplingClient-backed shim instead.

    python pirl/tinker_shim.py --port 8300 --model Qwen/Qwen3.5-4B
"""

import argparse
import json
import os
import sys
from pathlib import Path

import httpx
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

TINKER_OAI = "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1"
ROOT = Path(__file__).resolve().parents[1]


def build_app(models, thinking):
    load_dotenv(ROOT / ".env")
    key = os.environ.get("TINKER_API_KEY")
    if not key:
        sys.exit("TINKER_API_KEY missing from environment/.env")
    headers = {"Authorization": f"Bearer {key}"}
    client = httpx.AsyncClient(base_url=TINKER_OAI, headers=headers, timeout=httpx.Timeout(900, connect=30))
    app = FastAPI()
    stats = {"requests": 0, "prompt_tokens": 0, "cached_tokens": 0, "completion_tokens": 0, "errors": 0}

    @app.get("/v1/models")
    async def list_models():
        return {"object": "list", "data": [{"id": m, "object": "model", "owned_by": "tinker"} for m in models]}

    @app.get("/health")
    async def health():
        return {"ok": True}

    @app.get("/stats")
    async def get_stats():
        return stats

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        body = await request.json()
        body["reasoning_effort"] = thinking
        # Tinker rejects vLLM-only fields some clients send.
        for k in ("return_token_ids", "chat_template_kwargs", "skip_special_tokens", "top_k"):
            body.pop(k, None)
        stats["requests"] += 1
        if body.get("stream"):
            async def gen():
                async with client.stream("POST", "/chat/completions", json=body) as r:
                    async for chunk in r.aiter_raw():
                        yield chunk
            return StreamingResponse(gen(), media_type="text/event-stream")
        r = await client.post("/chat/completions", json=body)
        if r.status_code != 200:
            stats["errors"] += 1
            return Response(r.content, status_code=r.status_code, media_type="application/json")
        data = r.json()
        u = data.get("usage") or {}
        stats["prompt_tokens"] += u.get("prompt_tokens") or 0
        stats["completion_tokens"] += u.get("completion_tokens") or 0
        stats["cached_tokens"] += ((u.get("prompt_tokens_details") or {}).get("cached_tokens")) or 0
        return JSONResponse(data)

    return app


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", type=int, default=8300)
    p.add_argument("--model", action="append", default=None)
    p.add_argument("--thinking", action="store_true")
    a = p.parse_args()
    app = build_app(a.model or ["Qwen/Qwen3.5-4B"], a.thinking)
    uvicorn.run(app, host="127.0.0.1", port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
