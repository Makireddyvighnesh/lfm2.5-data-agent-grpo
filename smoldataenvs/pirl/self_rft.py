"""Self-RFT warm start: masked fine-tuning on the model's OWN verified-correct, low-context rollouts.

No teacher data and no edited demonstrations: the corpus is filtered from `pirl/evaluate.py` outputs of
the base model on the training tasks. Kept: verifier-correct, capture-validated (ATIF match), never ran
out of context, peak context <= --max-peak; at most --per-task rollouts per task (lowest peak first).
Each model call of a kept rollout is one sample: the captured prompt ids are context and only the
sampled completion tokens (loss_mask) are trained, with plain cross-entropy at temperature 1.

A diversity gate refuses to train when the corpus is dominated by a few (small-file) tasks.
Output: <output>/adapter (PEFT LoRA, loadable by vLLM and by grpo.py --init-adapter) + corpus.json.
"""

import argparse
import glob
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@dataclass
class Sample:
    input_ids: list
    completion_mask: list
    rollout_id: str


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rollouts", nargs="+", required=True, help="evaluate.py output dirs (with rollouts/*.json)")
    p.add_argument("--output", required=True)
    p.add_argument("--max-peak", type=int, default=12_000)
    p.add_argument("--per-task", type=int, default=2)
    p.add_argument("--sizes", help="analysis/task_file_sizes.json, for the large-file gate")
    p.add_argument("--large-bytes", type=int, default=50_000)
    p.add_argument("--min-tasks", type=int, default=25)
    p.add_argument("--min-large-tasks", type=int, default=6)
    p.add_argument("--force", action="store_true", help="Train even if the diversity gate fails")
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--rollouts-per-update", type=int, default=8)
    p.add_argument("--token-budget", type=int, default=16_384)
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--no-compile", dest="compile", action="store_false")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--select-only", action="store_true", help="Write corpus.json and the gate report, no training")
    return p.parse_args()


def select(args):
    rows = []
    for d in args.rollouts:
        for f in glob.glob(str(Path(d) / "rollouts" / "*.json")):
            r = json.loads(Path(f).read_text())
            if r.get("correctness") == 1 and r.get("trainable") and r.get("atif") == "match" \
                    and not r.get("context_exhausted") and r.get("peak_context") is not None \
                    and r["peak_context"] <= args.max_peak and r.get("trajectory"):
                rows.append(r)
    by_task = {}
    for r in sorted(rows, key=lambda r: r["peak_context"]):
        by_task.setdefault(r["task"], [])
        if len(by_task[r["task"]]) < args.per_task:
            by_task[r["task"]].append(r)
    kept = [r for rs in by_task.values() for r in rs]
    sizes = {}
    if args.sizes:
        sizes = {t["name"]: t["bytes"] for t in json.loads(Path(args.sizes).read_text())}
    large = sorted(t for t in by_task if sizes.get(t, 0) >= args.large_bytes)
    report = {"eligible_rollouts": len(rows), "kept_rollouts": len(kept), "distinct_tasks": len(by_task),
              "large_file_tasks": len(large), "large_file_task_names": large,
              "peak_context_kept": sorted(r["peak_context"] for r in kept),
              "gate": {"min_tasks": args.min_tasks, "min_large_tasks": args.min_large_tasks},
              "gate_passed": len(by_task) >= args.min_tasks and len(large) >= args.min_large_tasks}
    return kept, report


def samples_of(rollout):
    out = []
    for c in rollout["trajectory"]["calls"]:
        ids = list(c["prompt_token_ids"]) + list(c["completion_token_ids"])
        mask = list(c["loss_mask"])
        if len(mask) != len(ids):
            raise ValueError(f"{rollout['rollout_id']}: loss_mask {len(mask)} != ids {len(ids)}")
        if sum(mask):
            out.append(Sample(ids, mask, rollout["rollout_id"]))
    return out


def train(args, kept):
    import torch

    from pirl.grpo import build_model, pack

    model = build_model(argparse.Namespace(lora_r=args.lora_r, lora_alpha=args.lora_alpha, compile=args.compile))
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0, fused=True)
    rng = random.Random(args.seed)
    log = []
    update = 0
    for epoch in range(args.epochs):
        order = kept[:]
        rng.shuffle(order)
        for start in range(0, len(order), args.rollouts_per_update):
            chunk = [s for r in order[start:start + args.rollouts_per_update] for s in samples_of(r)]
            n_tokens = sum(sum(s.completion_mask[1:]) for s in chunk)
            rows = pack([(s, 1.0) for s in chunk], args.token_budget)
            model.train()
            t0 = time.time()
            loss_sum = 0.0
            for row in rows:
                ids, pos, mask = [], [], []
                for s, _ in row["items"]:
                    ids += s.input_ids
                    pos += list(range(len(s.input_ids)))
                    mask += s.completion_mask
                ids = torch.tensor([ids], device="cuda")
                pos = torch.tensor([pos], device="cuda")
                m = torch.tensor([mask], device="cuda", dtype=torch.float32)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = model(input_ids=ids, position_ids=pos, labels=ids.masked_fill(m == 0, -100), fused_lm_head=True)
                loss = -(out.log_probs.float() * m[:, 1:]).sum() / n_tokens
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite SFT loss")
                loss.backward()
                loss_sum += loss.item()
                del out, loss
            grad = torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm).item()
            if not math.isfinite(grad):
                raise FloatingPointError("non-finite gradient")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            update += 1
            rec = {"update": update, "epoch": epoch, "nll_per_token": loss_sum, "tokens": n_tokens,
                   "samples": len(chunk), "grad_norm": grad, "seconds": round(time.time() - t0, 1),
                   "peak_mem_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
            log.append(rec)
            print(json.dumps(rec), flush=True)
    model.save_pretrained(Path(args.output) / "adapter")
    return log


def main():
    args = arguments()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    kept, report = select(args)
    (out / "corpus.json").write_text(json.dumps({**report, "rollout_ids": [r["rollout_id"] for r in kept],
                                                 "tasks": sorted({r["task"] for r in kept})}, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k not in ("peak_context_kept", "large_file_task_names")}))
    if args.select_only:
        return
    if not report["gate_passed"] and not args.force:
        raise SystemExit("Diversity gate failed: collect more rollouts (or pass --force)")
    log = train(args, kept)
    (out / "train_log.json").write_text(json.dumps({"args": vars(args), "updates": log}, indent=1))


if __name__ == "__main__":
    main()
