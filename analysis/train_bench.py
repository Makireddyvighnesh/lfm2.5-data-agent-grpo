"""Compare packed token budgets on the same smoke rollouts and initial adapter."""
import argparse
import gzip
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

DA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DA.parent / "SmolDataEnvs"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("run")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    import torch
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    from pirl.grpo import Engine, build_model, group_advantages, pack, train_step

    cfg = json.loads(Path(a.run, "run_config.json").read_text())
    args = SimpleNamespace(**cfg)
    groups = {}
    for f in sorted(Path(a.run, "rollouts").glob("*.json.gz")):
        d = json.load(gzip.open(f))
        if d["metrics"]["policy_version"] != 0:
            continue
        row = dict(d["metrics"])
        row["sequences"] = [SimpleNamespace(**s) for s in d["training_sequences"]]
        groups.setdefault(row["task"], []).append(row)
    assert len(groups) == 8 and sum(map(len, groups.values())) == 64
    samples = []
    for group in groups.values():
        group_advantages(group, args.std_floor)
        samples.extend((s, r["advantage"]) for r in group if abs(r["advantage"]) > 1e-8
                       for s in r["sequences"] if sum(s.completion_mask))
    engine = Engine(args.vllm_url)
    engine.sleep()
    results = []
    try:
        model = build_model(args)
        initial = load_file(str(Path(args.init_adapter, "adapter_model.safetensors")))
        params = [p for p in model.parameters() if p.requires_grad]
        for budget in [16384, 32768]:
            args.token_budget = budget
            # Warm the budget's largest packed shape; exclude compilation from the comparison.
            set_peft_model_state_dict(model, initial)
            optimizer = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.999), weight_decay=0., fused=True)
            train_step(model, optimizer, pack(samples, budget)[0]["items"], args)
            set_peft_model_state_dict(model, initial)
            optimizer = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.999), weight_decay=0., fused=True)
            torch.cuda.reset_peak_memory_stats()
            stats = train_step(model, optimizer, samples, args)
            stats.pop("per_rollout", None)
            stats.update(token_budget=budget, forward_tok_per_s=round(stats["forward_tokens"] / stats["train_s"]))
            results.append(stats)
            Path(a.out).write_text(json.dumps(results, indent=2) + "\n")
            print(json.dumps(stats), flush=True)
        del model, optimizer
        torch.cuda.empty_cache()
    finally:
        engine.wake()

if __name__ == "__main__":
    main()
