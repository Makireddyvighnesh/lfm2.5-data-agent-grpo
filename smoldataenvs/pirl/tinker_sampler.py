"""vLLM-compatible OpenAI chat endpoint backed by Tinker's native SamplingClient (training tier).

Harbor's capture proxy talks to it exactly as to vLLM: the response carries the engine's own
`prompt_token_ids`, the sampled `token_ids`, and per-token logprobs as `token_id:N` entries, so pi
rollouts on Tinker are trainable and token-faithful. Messages are rendered with the model's HF chat
template (thinking off), and Qwen3.5's XML tool calls are parsed back into OpenAI `tool_calls`.

Policies are addressed by name: the base model, plus any sampler checkpoint registered at runtime via
POST /admin/models {"name": "tinker-policy-v3", "path": "tinker://..."} (the trainer does this after
each update). The Tinker key comes from .env (TINKER_API_KEY) and is never logged.

    python pirl/tinker_sampler.py --port 8300 --model Qwen/Qwen3.5-4B
"""

import argparse
import asyncio
import json
import os
import re
import sys
import time
import uuid
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

ROOT = Path(__file__).resolve().parents[1]
MAX_CONTEXT = 65_536  # Qwen3.5-4B on Tinker
MAX_GEN = 4_096       # per-call output cap, same as the L4 runs

TOOL_CALL = re.compile(r"<tool_call>(.*?)(?:</tool_call>|$)", re.S)
FUNCTION = re.compile(r"<function=([^>\s]+)>(.*?)(?:</function>|$)", re.S)
PARAM = re.compile(r"<parameter=([^>\s]+)>\n?(.*?)\n?(?:</parameter>|$)", re.S)


def _schema_types(tools):
    types = {}
    for t in tools or []:
        f = t.get("function", t)
        props = ((f.get("parameters") or {}).get("properties")) or {}
        types[f.get("name")] = {k: (v or {}).get("type") for k, v in props.items()}
    return types


def parse_tool_calls(text, tools):
    """Qwen3.5 XML tool calls -> (content, OpenAI tool_calls). Typed by the tool schema."""
    types = _schema_types(tools)
    calls = []
    for i, block in enumerate(TOOL_CALL.findall(text)):
        for name, body in FUNCTION.findall(block):
            args = {}
            for key, raw in PARAM.findall(body):
                kind = types.get(name, {}).get(key)
                value = raw
                if kind in ("integer", "number", "boolean", "object", "array"):
                    try:
                        value = json.loads(raw)
                    except ValueError:
                        value = raw
                args[key] = value
            calls.append({"id": f"call_{uuid.uuid4().hex[:12]}_{i}", "type": "function",
                          "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}})
    content = text.split("<tool_call>", 1)[0].strip() if calls else text.strip()
    return content, calls


def normalize_messages(messages):
    """OpenAI wire format -> what the HF template expects (string content, dict tool arguments)."""
    out = []
    for m in messages:
        m = dict(m)
        c = m.get("content")
        if isinstance(c, list):
            m["content"] = "".join(p.get("text", "") for p in c if isinstance(p, dict))
        elif c is None:
            m["content"] = ""
        if m.get("tool_calls"):
            calls = []
            for tc in m["tool_calls"]:
                tc = dict(tc)
                f = dict(tc.get("function") or {})
                if isinstance(f.get("arguments"), str):
                    try:
                        f["arguments"] = json.loads(f["arguments"] or "{}")
                    except ValueError:
                        pass
                tc["function"] = f
                calls.append(tc)
            m["tool_calls"] = calls
        out.append(m)
    return out


def context_error(prompt_len, max_tokens):
    msg = (f"This model's maximum context length is {MAX_CONTEXT} tokens. However, you requested "
           f"{prompt_len + max_tokens} tokens ({prompt_len} in the messages, {max_tokens} in the completion). "
           "Please reduce the length of the messages or completion.")
    return JSONResponse({"error": {"message": msg, "type": "invalid_request_error",
                                   "code": "context_length_exceeded"}}, status_code=400)


def build_app(base_model):
    load_dotenv(ROOT / ".env")
    if not os.environ.get("TINKER_API_KEY"):
        sys.exit("TINKER_API_KEY missing from environment/.env")
    import tinker
    from tinker import types
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base_model)
    stop_ids = [tok.convert_tokens_to_ids("<|im_end|>")]
    service = tinker.ServiceClient()
    clients = {base_model: service.create_sampling_client(base_model=base_model)}
    paths = {base_model: None}
    stats = {"requests": 0, "prompt_tokens": 0, "cache_hit_tokens": 0, "sampled_tokens": 0, "errors": 0,
             "context_rejects": 0}
    app = FastAPI()

    @app.get("/health")
    async def health():
        return {"ok": True}

    @app.get("/stats")
    async def get_stats():
        return stats

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": n, "object": "model", "owned_by": "tinker",
                                            "max_model_len": MAX_CONTEXT, "root": paths[n] or n} for n in clients]}

    @app.post("/admin/models")
    async def add_model(request: Request):
        body = await request.json()
        clients[body["name"]] = await asyncio.to_thread(service.create_sampling_client, model_path=body["path"])
        paths[body["name"]] = body["path"]
        return {"ok": True, "models": list(clients)}

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        body = await request.json()
        name = body.get("model") or base_model
        client = clients.get(name)
        if client is None:
            return JSONResponse({"error": {"message": f"model {name!r} not found", "type": "invalid_request_error"}},
                                status_code=404)
        tools = body.get("tools")
        ids = tok.apply_chat_template(normalize_messages(body.get("messages") or []), tools=tools,
                                      add_generation_prompt=True, tokenize=True, return_dict=False,
                                      enable_thinking=False)
        ids = list(ids)
        max_tokens = min(int(body.get("max_tokens") or body.get("max_completion_tokens") or MAX_GEN), MAX_GEN)
        if len(ids) + max_tokens > MAX_CONTEXT:
            stats["context_rejects"] += 1
            return context_error(len(ids), max_tokens)
        params = types.SamplingParams(
            max_tokens=max_tokens, temperature=float(body.get("temperature", 1.0) if body.get("temperature") is not None else 1.0),
            top_p=float(body.get("top_p") or 1.0), top_k=int(body.get("top_k") or -1), stop=stop_ids,
        )
        try:
            fut = client.sample(prompt=types.ModelInput.from_ints(ids), num_samples=1, sampling_params=params)
            resp = await asyncio.to_thread(fut.result)
        except Exception as exc:  # noqa: BLE001 - surface as a retryable upstream error
            stats["errors"] += 1
            return JSONResponse({"error": {"message": f"tinker sample failed: {type(exc).__name__}: {exc}"[:500],
                                           "type": "server_error"}}, status_code=503)
        seq = resp.sequences[0]
        out_ids, lps = list(seq.tokens), list(seq.logprobs or [])
        text = tok.decode([t for t in out_ids if t not in stop_ids], skip_special_tokens=False)
        content, tool_calls = parse_tool_calls(text, tools)
        stopped = str(getattr(seq, "stop_reason", "")).lower()
        finish = "length" if "length" in stopped or (len(out_ids) >= max_tokens and out_ids[-1:] != stop_ids) else (
            "tool_calls" if tool_calls else "stop")
        stats["requests"] += 1
        stats["prompt_tokens"] += len(ids)
        stats["cache_hit_tokens"] += int(getattr(resp, "prompt_cache_hit_tokens", 0) or 0)
        stats["sampled_tokens"] += len(out_ids)
        message = {"role": "assistant", "content": content or None}
        if tool_calls:
            message["tool_calls"] = tool_calls
        return JSONResponse({
            "id": f"chatcmpl-{uuid.uuid4().hex}", "object": "chat.completion", "created": int(time.time()),
            "model": name, "prompt_token_ids": ids,
            "choices": [{
                "index": 0, "message": message, "finish_reason": finish, "token_ids": out_ids,
                "logprobs": {"content": [{"token": f"token_id:{t}", "logprob": float(lp), "bytes": [],
                                          "top_logprobs": []} for t, lp in zip(out_ids, lps)]},
            }],
            "usage": {"prompt_tokens": len(ids), "completion_tokens": len(out_ids),
                      "total_tokens": len(ids) + len(out_ids),
                      "prompt_tokens_details": {"cached_tokens": int(getattr(resp, "prompt_cache_hit_tokens", 0) or 0)}},
        })

    return app


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", type=int, default=8300)
    p.add_argument("--model", default="Qwen/Qwen3.5-4B")
    a = p.parse_args()
    import uvicorn

    uvicorn.run(build_app(a.model), host="127.0.0.1", port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
