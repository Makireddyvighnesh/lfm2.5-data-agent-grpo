"""How closely does the trainer reproduce the FP8 vLLM sampler's token logprobs? BF16 vs FP8-emulated trainer.

Uses captured training sequences sampled by FP8 vLLM with a zero-initialised LoRA (policy version 0 = base),
scores the same tokens with the trainer, and reports the sampler/trainer gap on completion tokens.
usage: python analysis/fp8_match.py runs/grpo_r1 --n 24
"""
import argparse
import gc
import gzip
import json
import math
import sys
from pathlib import Path

SMOL = Path(__file__).resolve().parents[2] / "SmolDataEnvs"
sys.path.insert(0, str(SMOL))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run")
    p.add_argument("--n", type=int, default=24)
    p.add_argument("--adapter", help="LoRA adapter dir the rollouts were sampled with (default: zero LoRA = base)")
    p.add_argument("--modes", nargs="+", default=["bf16", "vllm"], choices=["bf16", "fake", "vllm"])
    a = p.parse_args()
    import torch

    from pirl.grpo import build_model

    seqs = []
    for f in sorted(Path(a.run, "rollouts").glob("*.json")):  # evaluate.py output: trajectory calls
        r = json.loads(f.read_text())
        for c in (r.get("trajectory") or {}).get("calls", []):
            ids = list(c["prompt_token_ids"]) + list(c["completion_token_ids"])
            if len(ids) <= 16384 and sum(c["loss_mask"]):
                seqs.append({"input_ids": ids, "completion_mask": list(c["loss_mask"]),
                             "old_log_probs": [0.0] * len(c["prompt_token_ids"]) + list(c["logprobs"])})
    for f in sorted(Path(a.run, "rollouts").glob("*.json.gz")):
        d = json.load(gzip.open(f))
        if d["metrics"].get("policy_version") == 0:
            seqs += [s for s in d["training_sequences"] if len(s["input_ids"]) <= 16384]
        if len(seqs) >= a.n:
            break
    seqs = seqs[: a.n]
    print(f"{len(seqs)} sequences, {sum(sum(s['completion_mask']) for s in seqs)} completion tokens", flush=True)
    for mode in a.modes:
        fp8 = mode != "bf16"
        model = build_model(argparse.Namespace(lora_r=32, lora_alpha=64, compile=False, fp8_base=fp8, fp8_mode=mode))
        if a.adapter:
            from peft import set_peft_model_state_dict
            from safetensors.torch import load_file
            set_peft_model_state_dict(model, load_file(str(Path(a.adapter) / "adapter_model.safetensors")))
        model.eval()
        lr_all = []
        with torch.no_grad():
            for s in seqs:
                ids = torch.tensor([s["input_ids"]], device="cuda")
                pos = torch.arange(ids.shape[1], device="cuda")[None]
                mask = torch.tensor([s["completion_mask"]], device="cuda")
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = model(input_ids=ids, position_ids=pos, labels=ids.masked_fill(mask == 0, -100), fused_lm_head=True)
                logp = out.log_probs.float()[0]
                old = torch.tensor(s["old_log_probs"], device="cuda")[1:]
                m = mask[0, 1:] > 0
                lr_all.append((logp - old)[m])
        lr = torch.cat(lr_all)
        kl = (lr.exp() - 1 - lr).mean().item()
        print(json.dumps({"trainer": mode, "tokens": lr.numel(),
                          "kl_k3": round(kl, 6), "mean_abs_logratio": round(lr.abs().mean().item(), 5),
                          "ratio_mean": round(lr.exp().mean().item(), 5),
                          "frac_ratio_off_20pct": round(((lr.exp() - 1).abs() > 0.2).float().mean().item(), 5),
                          "frac_ratio_gt_2": round((lr.exp() > 2).float().mean().item(), 5)}), flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
