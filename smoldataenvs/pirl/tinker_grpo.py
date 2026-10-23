"""GRPO for Qwen3.5-4B + LoRA on Tinker, with rollouts in the Pi harness (local Docker via Harbor).

Same recipe as the L4 trainer (pirl/grpo.py), different compute:
    rollouts : pi in Docker -> Harbor capture proxy -> pirl/tinker_sampler.py -> Tinker SamplingClient
               (exact prompt ids, sampled ids and logprobs; one training row per rollout)
    update   : Tinker forward_backward(loss "ppo", clip 0.8/1.28) + optim_step on a rank-32 LoRA
    publish  : save_weights_for_sampler -> registered in the shim as tinker-policy-v{step}

    python pirl/tinker_grpo.py --output runs/tinker_q4b --check          # verification only, no update
    python pirl/tinker_grpo.py --output runs/tinker_q4b --steps 10       # train (resumable)
"""

import argparse
import json
import math
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pirl import reward as rewards
from pirl.common import DATA, ROOT
from pirl.grpo import group_advantages, run_rollout

BASE = "Qwen/Qwen3.5-4B"
CONTEXT_BUDGET = 65_536 - 4_096  # Qwen3.5-4B window on Tinker minus the per-call output cap


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--groups", "--prompts-per-step", dest="groups", type=int, default=8)
    p.add_argument("--num-generations", type=int, default=8)
    p.add_argument("--concurrency", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-5, help="LoRA LR = 10x the blog's full fine-tune 3e-6")
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--clip-low", type=float, default=0.8)
    p.add_argument("--clip-high", type=float, default=1.28)
    p.add_argument("--reward", choices=list(rewards.REWARDS), default="efficiency")
    p.add_argument("--check", action="store_true", help="Verify the pipeline on one prompt; no weight update")
    p.add_argument("--step-limit", type=int, default=17)
    p.add_argument("--agent-timeout", type=int, default=300)
    p.add_argument("--server", default="http://127.0.0.1:8200")
    p.add_argument("--sampler", default="http://127.0.0.1:8300")
    p.add_argument("--trials", default=str(ROOT / "runs" / "trials"))
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    a.vllm_url = a.sampler  # run_rollout's name for the inference endpoint
    return a


def to_datum(seq, advantage):
    """TrainingSequence -> Tinker RL Datum. Context positions get advantage 0 (excluded from the loss)."""
    from tinker import types

    ids, mask, old = seq.input_ids, seq.completion_mask, seq.old_log_probs
    return types.Datum(
        model_input=types.ModelInput.from_ints(ids[:-1]),
        loss_fn_inputs={
            "target_tokens": list(ids[1:]),
            "logprobs": [float(x) for x in old[1:]],
            "advantages": [float(advantage) * m for m in mask[1:]],
        },
    )


def logprob_stats(datums, outputs):
    """Sampler vs trainer logprobs on trained positions: the token-faithfulness check (ratio ~ 1)."""
    import numpy as np

    diffs, ratios = [], []
    for d, o in zip(datums, outputs):
        new = np.asarray(o["logprobs"].tolist() if hasattr(o["logprobs"], "tolist") else o["logprobs"], dtype=float)
        old = np.asarray(d.loss_fn_inputs["logprobs"].tolist() if hasattr(d.loss_fn_inputs["logprobs"], "tolist")
                         else d.loss_fn_inputs["logprobs"], dtype=float)
        adv = np.asarray(d.loss_fn_inputs["advantages"].tolist() if hasattr(d.loss_fn_inputs["advantages"], "tolist")
                         else d.loss_fn_inputs["advantages"], dtype=float)
        sel = adv != 0
        diffs.append(new[sel] - old[sel])
    if not diffs or not sum(len(x) for x in diffs):
        return {}
    d = np.concatenate(diffs)
    r = np.exp(d)
    return {"ratio_mean": float(r.mean()), "abs_logprob_diff_mean": float(np.abs(d).mean()),
            "abs_logprob_diff_max": float(np.abs(d).max()), "kl": float((r - 1 - d).mean()),
            "clip_frac": float(((r < 0.8) | (r > 1.28)).mean()), "n_tokens": int(d.size)}


def register(sampler, name, path):
    httpx.post(f"{sampler}/admin/models", json={"name": name, "path": path}, timeout=600).raise_for_status()


def main():
    a = arguments()
    load_dotenv(ROOT / ".env")
    rewards.CONTEXT_BUDGET = CONTEXT_BUDGET
    import tinker
    from tinker import types
    from smoldataenv_harbor.tasks import load_tasks

    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    tasks = load_tasks(DATA, "train")
    order = list(range(len(tasks)))
    random.Random(a.seed).shuffle(order)
    split = httpx.get(a.server + "/smoldataenv/splits", timeout=30).json()["train"]
    reward_fn = rewards.REWARDS[a.reward]
    service = tinker.ServiceClient()

    state_file = out / "state.json"
    state = json.loads(state_file.read_text()) if state_file.exists() else {"step": 0, "cursor": 0}
    if state.get("train_state"):
        trainer = service.create_training_client_from_state(state["train_state"])
        print(f"resumed at step {state['step']} from {state['train_state']}", flush=True)
    else:
        trainer = service.create_lora_training_client(base_model=BASE, rank=a.rank, seed=a.seed)
    policy = BASE if state["step"] == 0 else f"tinker-policy-v{state['step']}"
    if state.get("sampler_path"):
        register(a.sampler, policy, state["sampler_path"])
    (out / "run_config.json").write_text(json.dumps({**vars(a), "base": BASE, "context_budget": CONTEXT_BUDGET}, indent=2))

    if a.check:
        a.groups, a.steps = 1, 1
    pool = ThreadPoolExecutor(max_workers=a.concurrency)
    log = (out / "metrics.jsonl").open("a")
    while state["step"] < a.steps:
        t0 = time.time()
        cost0 = httpx.get(a.sampler + "/stats").json()
        batch = [order[(state["cursor"] + g) % len(order)] for g in range(a.groups)]
        futs = [(g, pool.submit(run_rollout, a, split, i, tasks[i], policy, reward_fn))
                for g, i in enumerate(batch) for _ in range(a.num_generations)]
        groups = [[] for _ in batch]
        for g, f in futs:
            groups[g].append(f.result())
        gen_s = time.time() - t0
        contrast = [group_advantages(gr) for gr in groups]
        rollouts = [r for gr in groups for r in gr]
        datums = [to_datum(s, r["advantage"]) for r in rollouts for s in r["sequences"] if r["advantage"] != 0.0]
        cost1 = httpx.get(a.sampler + "/stats").json()

        if a.check:
            print("\n== CHECK: per-rollout reward breakdown (efficiency = 1 + .05 calls + .05 context + .03 gen, correct only)")
            for r in rollouts:
                print(f"  correct={r.get('correct')} reward={r.get('reward')} adv={r.get('advantage', 0):+.3f} calls={r.get('tool_calls')} "
                      f"ctx={r.get('final_context')} gen={r.get('generated_tokens')} rows={len(r['sequences'])} "
                      f"ctx_exhausted={r.get('context_exhausted')} err={r.get('error') or r.get('capture_error')}")
            every = [to_datum(s, 1.0) for r in rollouts for s in r["sequences"]]  # adv=1 just to select positions
            fwd = trainer.forward(every, "importance_sampling").result()
            print("\n== CHECK: sampler vs trainer logprobs (step-0 LoRA == base):",
                  json.dumps(logprob_stats(every, fwd.loss_fn_outputs)))
            if datums:
                fb = trainer.forward_backward(datums, "ppo", {"clip_low_threshold": a.clip_low,
                                                               "clip_high_threshold": a.clip_high}).result()
                print("== CHECK: forward_backward(ppo) on", len(datums), "rows OK; metrics:", fb.metrics)
            else:
                print("== CHECK: group had no reward contrast; forward_backward skipped")
            print("\nno optimizer step taken (check mode)")
            return

        tstats = {}
        t1 = time.time()
        if datums:
            fb = trainer.forward_backward(datums, "ppo", {"clip_low_threshold": a.clip_low,
                                                           "clip_high_threshold": a.clip_high}).result()
            tstats = logprob_stats(datums, fb.loss_fn_outputs)
            tstats["tinker_metrics"] = fb.metrics
            trainer.optim_step(types.AdamParams(learning_rate=a.lr, grad_clip_norm=1.0)).result()
        state["step"] += 1
        state["cursor"] += a.groups
        name = f"tinker-policy-v{state['step']}"
        sampler_path = trainer.save_weights_for_sampler(name=f"step-{state['step']:04d}").result().path
        register(a.sampler, name, sampler_path)
        state["sampler_path"] = sampler_path
        state["train_state"] = trainer.save_state(name=f"state-{state['step']:04d}").result().path
        policy = name
        train_s = time.time() - t1

        scored = [r for r in rollouts if r["reward"] is not None]

        def mean(k, rows=scored):
            v = [float(r[k]) for r in rows if r.get(k) is not None]
            return sum(v) / len(v) if v else None

        dp = cost1["prompt_tokens"] - cost0["prompt_tokens"]
        dh = cost1["cache_hit_tokens"] - cost0["cache_hit_tokens"]
        ds = cost1["sampled_tokens"] - cost0["sampled_tokens"]
        dt = sum(len(d.loss_fn_inputs["target_tokens"].tolist() if hasattr(d.loss_fn_inputs["target_tokens"], "tolist")
                      else d.loss_fn_inputs["target_tokens"]) for d in datums)
        cost = (dp - dh) * 0.33e-6 + dh * 0.066e-6 + ds * 1.005e-6 + dt * 0.737e-6
        rec = {
            "step": state["step"], "epoch": round(state["cursor"] / len(order), 3), "rollouts": len(rollouts),
            "reward_mean": mean("reward"), "correct_rate": mean("correct"),
            "zero_adv_group_frac": 1 - sum(contrast) / len(contrast), "submit_rate": mean("submitted", rollouts),
            "context_exhausted_rate": mean("context_exhausted", rollouts), "truncated_rate": mean("truncated"),
            "tool_calls_mean": mean("tool_calls"), "generated_tokens_mean": mean("generated_tokens"),
            "final_context_mean": mean("final_context"), "infra_errors": sum(1 for r in rollouts if r.get("error")),
            "capture_errors": sum(1 for r in rollouts if r.get("capture_error")), "trained_rows": len(datums),
            "train_tokens": dt, "gen_s": round(gen_s, 1), "train_s": round(train_s, 1),
            "step_s": round(time.time() - t0, 1), "cost_usd": round(cost, 3),
            "prefill_tokens": dp, "cache_hit_tokens": dh, "sampled_tokens": ds, **tstats,
        }
        log.write(json.dumps(rec, default=str) + "\n")
        log.flush()
        with (out / "rollouts.jsonl").open("a") as f:
            for r in rollouts:
                f.write(json.dumps({k: v for k, v in r.items() if k != "sequences"} | {"step": state["step"]}) + "\n")
        state_file.write_text(json.dumps(state))
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items() if k != "tinker_metrics"}),
              flush=True)
    pool.shutdown()


if __name__ == "__main__":
    main()
