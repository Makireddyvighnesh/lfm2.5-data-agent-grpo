"""Start the token-faithful vLLM server on the shared L4.

Training mode adds runtime LoRA loading so AsyncGRPO can sync adapter-only (no NCCL, which cannot
pair two processes on one GPU). Eval mode can serve a saved adapter under a fixed name.
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pirl.common import MODEL, REVISION, tokenizer_for


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--model", default=MODEL, help="HF model id (e.g. a released RL checkpoint); uses LFM's template")
    p.add_argument("--revision", default=REVISION)
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--gpu-mem", type=float, default=0.40)
    p.add_argument("--max-model-len", type=int, default=32768)
    p.add_argument("--max-num-seqs", type=int, default=32)
    p.add_argument("--lora-rank", type=int, default=32, help="0 disables LoRA")
    p.add_argument("--max-loras", type=int, default=4)
    p.add_argument("--adapter", action="append", default=[], help="name=path, served as a model")
    p.add_argument("--prefix-caching", action="store_true")
    p.add_argument("--eager", action="store_true", help="Disable CUDA graphs")
    p.add_argument("--sleep-mode", action="store_true", help="Allow /sleep and /wake_up (GPU time-sharing)")
    p.add_argument("--quantization", help="vLLM online quantization, e.g. fp8 (W8A8, dynamic scales)")
    p.add_argument("--kv-cache-dtype", help="e.g. fp8; default keeps the model dtype")
    p.add_argument("--worker-cls", help="Optional runtime worker subclass (upstream source stays unchanged)")
    a = p.parse_args()

    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    template = out / "chat_template.jinja"
    template.write_text(tokenizer_for().chat_template)
    cmd = [
        "vllm", "serve", a.model,
        "--revision", a.revision,
        "--host", "0.0.0.0",  # sandboxes reach the capture proxy, not vLLM, but keep it reachable
        "--port", str(a.port),
        "--served-model-name", a.model,
        "--dtype", "bfloat16",
        "--max-model-len", str(a.max_model_len),
        "--max-num-seqs", str(a.max_num_seqs),
        "--gpu-memory-utilization", str(a.gpu_mem),
        "--enable-auto-tool-choice",
        "--tool-call-parser", "lfm2",
        "--chat-template", str(template),
        "--default-chat-template-kwargs", '{"enable_thinking":false,"preserve_thinking":true}',
        "--generation-config", "vllm",
        "--logprobs-mode", "processed_logprobs",
        "--return-tokens-as-token-ids",
        "--enable-prefix-caching" if a.prefix_caching else "--no-enable-prefix-caching",
    ]
    if a.sleep_mode:
        cmd.append("--enable-sleep-mode")
    if a.quantization:
        cmd += ["--quantization", a.quantization]
    if a.kv_cache_dtype:
        cmd += ["--kv-cache-dtype", a.kv_cache_dtype]
    if a.worker_cls:
        cmd += ["--worker-cls", a.worker_cls]
    if a.eager:
        cmd.append("--enforce-eager")
    if a.lora_rank:
        cmd += ["--enable-lora", "--max-lora-rank", str(a.lora_rank), "--max-loras", str(a.max_loras)]
        os.environ["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "1"
    if a.adapter:
        cmd += ["--lora-modules", *a.adapter]
    os.environ["VLLM_SERVER_DEV_MODE"] = "1"  # /pause, /resume and /server_info for the trainer
    (out / "serve_cmd.json").write_text(json.dumps(cmd, indent=1))
    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    main()
