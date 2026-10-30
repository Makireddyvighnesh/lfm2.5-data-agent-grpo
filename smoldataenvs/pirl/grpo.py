"""Synchronous, GPU time-shared GRPO for LFM2.5-2.6B + LoRA inside the Pi harness on one L4.

Why not AsyncGRPO: Pi rollouts are 20-32k-token rows (pi's `read` returns up to 50 KB). Training a
32k row needs ~17 GiB, which cannot coexist with a vLLM that has enough KV cache for a GRPO group.
So each step time-shares the GPU:

    vLLM awake  -> G groups x N pi rollouts in Docker (Harbor capture: exact ids + logprobs)
    vLLM sleep  -> (level 1: weights to CPU, KV freed) -> one optimizer step on the whole GPU
    vLLM wake   -> load adapter trl-policy-v{step}  -> next step

Sampling is strictly on-policy (the adapter that generated a group is the one being updated), so the
PPO ratio should be ~1.0 apart from numerics; that is the token-faithfulness check (M6).
"""

import argparse
import json
import math
import os
import random
import shutil
import sys
import threading
import time
import uuid
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pirl.common import DATA, LORA_NAME, MAX_TOKENS, MODEL, PREFIX_CHAIN_TEMPLATE, REVISION, TEMPERATURE, TOP_K, TOP_P
from pirl import reward as rewards
from pirl.artifacts import append_jsonl, save_rollout, sync_directory, write_json
from pirl.harbor_client import harbor_client

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "out_proj", "w1", "w2", "w3", "in_proj"]


def _agent_seconds(trial_dir):
    """Agent execution wall time (Harbor result.json), excluding sandbox setup and grading; None if unavailable."""
    from datetime import datetime
    try:
        span = json.loads((trial_dir / "result.json").read_text())["agent_execution"]
        start, end = (datetime.fromisoformat(span[k].replace("Z", "+00:00")) for k in ("started_at", "finished_at"))
        return (end - start).total_seconds()
    except Exception:  # noqa: BLE001 - missing or partial result: no time penalty rather than a crash
        return None


def unknown_tool_calls(result_payload):
    """Calls to a tool name the harness did not offer in that same request. Needs the rollout RESULT payload
    (`session.result.model_dump()` or a capture's "result"): its turns carry `request_tools` and `tool_calls`;
    the training trace's turns do not. Turns offering no tools (text-only harness turns) are not counted.
    None when no turn carries a tool list, so a missing field can never read as "0 unknown calls"."""
    turns = (result_payload or {}).get("turns")
    if turns is None:
        turns = ((result_payload or {}).get("result") or {}).get("turns")
    n, seen = 0, False
    for t in turns or []:
        offered = {((x.get("function") or x).get("name")) for x in (t.get("request_tools") or [])}
        if not offered:
            continue
        seen = True
        n += sum(1 for c in (t.get("tool_calls") or []) if (c.get("name") or (c.get("function") or {}).get("name")) not in offered)
    return n if seen else None


def _text_len(content):
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):  # same rule as serve_harbor.py's cap: each text part is capped on its own
        return max((len(p.get("text") or "") for p in content if isinstance(p, dict) and p.get("type") == "text"),
                   default=0)
    return 0


def capped_tool_outputs(result_payload, cap):
    """Tool outputs the capture proxy cut to head + tail (HARBOR_TOOL_OUTPUT_CAP chars): a whole-file read or dump.
    The result's `request_messages` are recorded before the cap, so a tool message longer than `cap` was capped.
    Each output is counted once, on the turn where it is new (the trailing tool messages of that request), as the
    proxy's `tool_output_capped` event does. None when no turn carries request messages, or the cap is off."""
    turns = (result_payload or {}).get("turns")
    if turns is None:
        turns = ((result_payload or {}).get("result") or {}).get("turns")
    if not cap:
        return None
    n, seen = 0, False
    for t in turns or []:
        msgs = t.get("request_messages")
        if not msgs:
            continue
        seen = True
        i = len(msgs)
        while i > 0 and isinstance(msgs[i - 1], dict) and msgs[i - 1].get("role") == "tool":
            i -= 1
        n += sum(1 for m in msgs[i:] if _text_len(m.get("content")) > cap)
    return n if seen else None


def _atif_tool_calls(trial_dir):
    """Tool actions in Harbor's ATIF trajectory (the blog's count_tools): works for every harness, including
    mini-swe-agent's text actions. None when the file is missing or its action ids are missing/repeated."""
    try:
        traj = json.loads((trial_dir / "agent" / "trajectory.json").read_text())
        if not str(traj.get("schema_version", "")).startswith("ATIF-"):
            return None
        ids = [c.get("tool_call_id") for st in traj["steps"] if st.get("source") == "agent"
               for c in (st.get("tool_calls") or [])]
        return None if any(not i for i in ids) or len(set(ids)) != len(ids) else len(ids)
    except Exception:  # noqa: BLE001
        return None


def harness_for(args, index):
    names = [h.strip() for h in str(getattr(args, "harnesses", "pi")).split(",") if h.strip()] or ["pi"]
    return names[index % len(names)]


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--groups", "--prompts-per-step", dest="groups", type=int, default=8,
                   help="Prompts (GRPO groups) per optimizer update")
    p.add_argument("--num-generations", type=int, default=8, help="Rollouts per task (GRPO group)")
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--lr-schedule", choices=["constant", "cosine"], default="constant",
                   help="cosine: linear warmup over --warmup-steps, then cosine decay to --min-lr at --steps "
                        "(a function of the update number only, so resume needs no scheduler state)")
    p.add_argument("--warmup-steps", type=int, default=0)
    p.add_argument("--min-lr", type=float, default=0.0)
    p.add_argument("--schedule-steps", type=int, default=0,
                   help="cosine span in updates (0 = --steps); lets a run pause at --steps and resume to a later target")
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--epsilon", type=float, default=0.2)
    p.add_argument("--epsilon-high", type=float, default=0.28)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--token-budget", type=int, default=32768,
                   help="Max tokens per micro-batch (one packed forward/backward); the step accumulates "
                        "gradients over all micro-batches before a single optimizer update")
    p.add_argument("--concurrency", type=int, default=16, help="Concurrent rollouts (sandboxes)")
    p.add_argument("--sampler", choices=["sequential", "regret"], default="sequential",
                   help="sequential: fixed order; regret: Frontier-Learning priority with floor + adaptive group size")
    p.add_argument("--hard-generations", type=int, default=8,
                   help="(regret) rollouts for prompts that scored 0 last visit; others use --num-generations")
    p.add_argument("--new-frac", type=float, default=0.5, help="(regret) share of each batch drawn from unseen prompts")
    p.add_argument("--reward", choices=["correctness", "calls", "efficiency", "context", "tool", "tool_context", "tool_fast", "tool_ctx", "tool_ctx_err", "tool_ctx_read", "tool_ctx_read2"], default="calls",
                   help="A: correctness, B: calls (blog recipe), C: efficiency (calls+context+generation), "
                        "context: correctness + <=0.1 bonus for low peak context")
    p.add_argument("--harnesses", default="pi",
                   help="Comma-separated Harbor harnesses; task index i uses harnesses[i %% n] for all its rollouts "
                        "(the multi-harness blog's assignment), e.g. pi,mini-swe-agent,codex")
    p.add_argument("--std-floor", type=float, default=0.0,
                   help="Floor on the group reward std in the advantage, so tiny bonus-only differences among "
                        "all-correct rollouts are not scaled up to full-size advantages (0 = TRL behaviour)")
    p.add_argument("--is-correction", choices=["ppo", "tis"], default="ppo",
                   help="ppo: clipped ratio vs the sampler's logprobs (exact when sampler and trainer match). "
                        "tis: one on-policy update, so plain policy gradient, reweighted per token by the "
                        "truncated importance weight min(pi_train/pi_sampler, --tis-cap) to correct a sampler "
                        "that differs from the trainer (e.g. FP8 vLLM vs BF16 trainer; FlashRL)")
    p.add_argument("--tis-cap", type=float, default=2.0)
    p.add_argument("--root-weights", help="JSONL written by Data Agent/root_proxy.py: stratified importance weight "
                                          "of each rollout's first turn (keyed by sha1 of its token ids)")
    p.add_argument("--root-boost", type=float, default=1.0,
                   help="Multiply the advantage of each rollout's first-turn sequence (the opening action) by this, "
                        "so the decision that decides most outcomes is not ~3%% of the token-normalized loss")
    p.add_argument("--fp8-base", action="store_true",
                   help="Trainer computes like vLLM --quantization fp8 (FP8-rounded base weights with vLLM's fused "
                        "scale groups + per-token FP8 activations; LoRA in BF16), so sampler and trainer agree")
    p.add_argument("--fp8-mode", choices=["vllm", "fake"], default="vllm",
                   help="vllm: call vLLM's own FP8 quantizer + CUTLASS kernel; fake: quantize-dequantize emulation")
    p.add_argument("--init-adapter", help="Start a fresh run from this LoRA adapter (e.g. a self-RFT warm start)")
    p.add_argument("--data-root", help="Prepared data root with datasets/train/tasks (default: pirl DATA); "
                   "lets a run use a modified copy of the train split, e.g. with a different grader")
    p.add_argument("--task-file", help="JSON list of train task names: the fixed training pool, in this order")
    p.add_argument("--task-limit", type=int, default=0,
                   help="Train on only the first N prompts of the seeded order (0 = all 1,000)")
    p.add_argument("--train-zero-adv", action="store_true", help="Also forward rows with zero advantage (ablation)")
    p.add_argument("--save-steps", type=int, default=50)
    p.add_argument("--step-limit", type=int, default=17)
    p.add_argument("--agent-timeout", type=int, default=300)
    p.add_argument("--server", default="http://127.0.0.1:8200")
    p.add_argument("--vllm-url", default="http://127.0.0.1:8000")
    p.add_argument("--trials", default=str(DATA.parent / "runs" / "trials"))
    p.add_argument("--no-sleep", action="store_true", help="Keep vLLM awake while training (memory ablation)")
    p.add_argument("--no-compile", dest="compile", action="store_false", help="Skip torch.compile (speed ablation)")
    p.add_argument("--no-autocast", action="store_true", help="fp32 LoRA compute path (speed ablation)")
    p.add_argument("--max-infra-fail", type=float, default=0.5,
                   help="Redo the batch if more than this share of rollouts failed for infrastructure reasons")
    p.add_argument("--max-infra-retries", type=int, default=5, help="Consecutive redo limit before exiting")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dynamic-sampling", action="store_true",
                   help="DAPO dynamic sampling (needs --stream-groups): drop finished groups whose rollouts are all "
                        "correct or all wrong and submit new tasks until `groups` groups with correctness contrast "
                        "are ready; dropped rollouts go to filtered_rollouts.jsonl")
    p.add_argument("--dynamic-max-drop", type=int, default=32,
                   help="dynamic sampling: after this many dropped groups in one update, accept groups unfiltered")
    p.add_argument("--allow-change", action="append", default=[], metavar="KEY",
                   help="let a resume change this fixed config key (repeatable); logged to config_changes.jsonl")
    p.add_argument("--stop-at-pos", type=int, default=0,
                   help="streaming: submit no task position >= this (e.g. 1000 = end of epoch 1 on 1000 tasks), "
                        "train on the remaining groups, then stop (0 = off)")
    p.add_argument("--async-groups", type=int, default=0,
                   help="streaming: keep this many UNFINISHED groups generating at all times (finished groups wait "
                        "for the next update instead of holding a slot); 0 = keep groups+stream_groups in flight")
    p.add_argument("--dynamic-skip", action="store_true",
                   help="dynamic sampling: never revisit a task once a visit was all correct; skip a revisit of a "
                        "task whose last visit was all wrong with probability --skip-wrong-frac (deterministic per "
                        "task and visit); skips go to skipped_tasks.jsonl")
    p.add_argument("--dynamic-keep-tool-contrast", action="store_true",
                   help="dynamic sampling: also keep groups whose rollouts differ in unknown-tool calls (the "
                        "tool_ctx_err penalty then teaches within all-correct / all-wrong groups)")
    p.add_argument("--probe-hard", type=int, default=0,
                   help="dynamic sampling: a task whose last visit was all wrong first gets this many rollouts; the "
                        "group is completed to --num-generations only if a probe succeeds (or differs in unknown-tool "
                        "calls with --dynamic-keep-tool-contrast), otherwise dropped at probe cost (0 = off)")
    p.add_argument("--skip-wrong-frac", type=float, default=0.5,
                   help="dynamic skip: probability of skipping a revisit after an all-wrong visit")
    p.add_argument("--keep-reward-spread", type=float, default=0.0,
                   help="dynamic sampling: also keep groups whose rewards spread by at least this much (e.g. an "
                        "all-correct group ranked by its context / capped-output terms); 0 = off")
    p.add_argument("--solved-revisit-frac", type=float, default=0.0,
                   help="dynamic skip: probability of revisiting a task already solved 8/8 instead of skipping it "
                        "(deterministic per task and visit); 0 = solved tasks are never revisited")
    p.add_argument("--tool-output-cap", type=int, default=int(os.environ.get("HARBOR_TOOL_OUTPUT_CAP") or 4000),
                   help="chars above which the capture proxy caps a tool output (Harbor's HARBOR_TOOL_OUTPUT_CAP); "
                        "used to count capped outputs per rollout (reward tool_ctx_read); 0 = off")
    p.add_argument("--stream-groups", type=int, default=0,
                   help="Keep groups+K task groups in flight and train on the first `groups` that finish; "
                        "unfinished rollouts keep their (older) adapter and pause in vLLM while the trainer runs. "
                        "0 = synchronous batches")
    return p.parse_args()


# ---------------------------------------------------------------- vLLM control
class Engine:
    def __init__(self, url):
        self.url = url.rstrip("/")
        self.http = httpx.Client(timeout=1800)

    def sleep(self, mode="abort"):
        # mode="keep": running requests are preempted and resume (recomputing their prefix) after wake_up
        self.http.post(f"{self.url}/sleep", params={"level": 1, "mode": mode}).raise_for_status()
        while not self.http.get(f"{self.url}/is_sleeping").json()["is_sleeping"]:
            time.sleep(0.5)

    def wake(self):
        self.http.post(f"{self.url}/wake_up").raise_for_status()
        while self.http.get(f"{self.url}/is_sleeping").json()["is_sleeping"]:
            time.sleep(0.5)

    def load_adapter(self, name, path):
        # vLLM outlives trainer restarts: a crashed update may have left this name loaded with other
        # weights, and a load under an existing name is a no-op. Always replace it.
        self.unload_adapter(name)
        r = self.http.post(f"{self.url}/v1/load_lora_adapter", json={"lora_name": name, "lora_path": str(path)})
        if r.status_code != 200:
            raise RuntimeError(f"load_lora_adapter {name}: {r.status_code} {r.text}")

    def unload_adapter(self, name):
        self.http.post(f"{self.url}/v1/unload_lora_adapter", json={"lora_name": name})


# ---------------------------------------------------------------- rollouts
def verifier_text(trials, trial_name):
    path = Path(trials) / trial_name / "verifier" / "test-stdout.txt"
    return path.read_text() if path.exists() else ""


_ROOT = {"offset": 0, "weights": {}, "lock": threading.Lock()}


def _sha1_ids(ids):
    import hashlib

    return hashlib.sha1(json.dumps([int(t) for t in ids]).encode()).hexdigest()


def root_weight(path, model, prompt_ids, token_ids):
    """Stratified importance weight of a first turn chosen by root_proxy.py, or None when not recorded.

    Records are keyed by (policy model, prompt ids, completion ids): the same opening text (e.g. `head -5
    train.csv`) recurs across tasks and policy versions with different weights. Every call reads the
    records appended since the last call, so a cached key can never hide a newer assignment.
    """
    key = f"{model}|{_sha1_ids(prompt_ids)}|{_sha1_ids(token_ids)}"
    with _ROOT["lock"]:
        if Path(path).exists():
            with open(path) as stream:
                stream.seek(_ROOT["offset"])
                for line in stream:
                    if not line.endswith("\n"):
                        break
                    rec = json.loads(line)
                    if rec.get("prompt_sha1"):  # records from the v1 proxy lack it and are ignored
                        _ROOT["weights"][f"{rec['model']}|{rec['prompt_sha1']}|{rec['ids_sha1']}"] = rec
                    _ROOT["offset"] += len(line.encode())
        return _ROOT["weights"].get(key)


def run_rollout(args, split_path, index, task, model_name, reward_fn, context=None):
    """One pi rollout in Docker. Returns a dict; `sequences` is empty when the rollout is unusable."""
    from harbor_env.harness import HarborSession
    from trl.experimental.async_grpo.async_rollout_worker import _chain_to_sequences
    from trl.experimental.async_grpo.openenv_harness import _tool_call_counts_by_name, _turns_from_training_trace

    t0 = time.monotonic()
    harness = harness_for(args, index)
    row = {"rollout_id": uuid.uuid4().hex, "task": task["name"], "difficulty": task.get("difficulty"),
           "task_index": index, "split": "train", "model": model_name, "revision": REVISION,
           "harness": harness, "sequences": [], "reward": None, "correct": None,
           "sampling": {"temperature": TEMPERATURE, "top_p": TOP_P, "top_k": TOP_K, "max_tokens": MAX_TOKENS},
           **(context or {})}
    result_payload = trace_payload = None
    def new_session():
        return HarborSession(
            env=harbor_client(args.server, message_timeout_s=1800), owns_env=True, split=split_path,
            task_index=index, instruction=task["instruction"], harness=harness, sandbox="docker",
            llm_url=args.vllm_url, model=model_name, reward_key="correctness,reward",
            sampling={"temperature": TEMPERATURE, "top_p": TOP_P, "top_k": TOP_K},
            agent_step_limit=args.step_limit, agent_timeout_sec=args.agent_timeout,
        )

    session = new_session()
    try:
        # `result is None` means the server call itself failed (connection/capacity) and no rollout ran,
        # so a retry cannot bias the group. A rollout that ran and failed is never retried.
        for attempt in range(4):
            try:
                session.wait_for_completion()
            except TimeoutError:
                row["timed_out"] = True
            if session.result is not None or attempt == 3:
                break
            row["retries"] = attempt + 1
            session.close()
            time.sleep(5 * (attempt + 1))
            session = new_session()
        result = session.result
        if result is None:
            raise RuntimeError("harbor rollout call failed 4 times")
        result_payload = result.model_dump(mode="json")
        env_reward = session.verify([]).env_reward
        verifier = Path(args.trials) / result.trial_name / "verifier" / "test-stdout.txt"
        row.update(
            trial=result.trial_name,
            ok=result.ok, atif=result.atif, phase_timings=result.phase_timings,
            findings=result.findings, env_reward=env_reward,
            context_exhausted=any("context_budget_exhausted" in f for f in result.findings),
            submitted=("no answer at" not in verifier.read_text()) if verifier.exists() else None,
            n_roots=result.n_roots,
        )
        try:
            trace = session.fetch_training_trace()
        except ValueError as exc:  # FATAL capture finding: never train on uncorroborated tokens
            row["capture_error"] = str(exc)[:300]
            row["findings"] = [f[:400] for f in result.findings if "FATAL" in f]
            return row
        turns = _turns_from_training_trace(trace)
        trace_payload = trace.model_dump(mode="json")
        entries = trace.to_trace_entries()
        parsed_calls = sum(_tool_call_counts_by_name(entries).values())
        atif_calls = _atif_tool_calls(Path(args.trials) / result.trial_name)
        tool_calls = atif_calls if atif_calls is not None else parsed_calls
        sequences, tally = _chain_to_sequences(turns, row["rollout_id"], 0)
        if getattr(args, "root_weights", None) and turns:
            rec = root_weight(args.root_weights, model_name, turns[0].prompt_ids, turns[0].output_ids)
            row.update(root_weight=float(rec["weight"]) if rec else 1.0,
                       root_class=rec["class"] if rec else None, root_stratum=rec["stratum"] if rec else None,
                       root_p_alt=rec["p_alt"] if rec else None)
        completion_lens = [len(t.output_ids) for t in turns]
        final_context = len(turns[-1].prompt_ids) + len(turns[-1].output_ids) if turns else 0
        peak_context = max((len(t.prompt_ids) + len(t.output_ids) for t in turns), default=0)
        outcome = type("Outcome", (), {"env_reward": None if env_reward is None else float(env_reward),
                                        "tool_call_count": tool_calls, "final_context": final_context,
                                        "peak_context": peak_context,
                                        "first_prompt": len(turns[0].prompt_ids) if turns else None,
                                        "unknown_tool_calls": unknown_tool_calls(result_payload),
                                        "capped_tool_outputs": capped_tool_outputs(
                                            result_payload, getattr(args, "tool_output_cap", 0)),
                                        "agent_seconds": _agent_seconds(Path(args.trials) / result.trial_name),
                                        "generated_tokens": sum(completion_lens)})
        row.update(
            correct=None if env_reward is None else float(env_reward),
            reward=reward_fn(outcome),
            tool_calls=tool_calls, parsed_tool_calls=parsed_calls, atif_tool_calls=atif_calls,
            unknown_tool_calls=outcome.unknown_tool_calls,
            capped_tool_outputs=outcome.capped_tool_outputs,
            first_prompt=len(turns[0].prompt_ids) if turns else None,
            model_calls=len(turns),
            generated_tokens=sum(completion_lens),
            agent_seconds=outcome.agent_seconds,
            final_context=final_context,
            peak_context=peak_context,
            own_context=(peak_context - len(turns[0].prompt_ids)) if turns else None,
            peak_le_10k=float(peak_context <= 10_000),
            truncated=any(n >= MAX_TOKENS for n in completion_lens),
            forks=tally["fork"],
            sequences=sequences,
            # LFM2.5 special tokens: </think> = 124902, <|tool_call_start|> = 124905. With thinking off the
            # generation prompt already ends in "<think></think>", so a generated </think> is reasoning leakage.
            think_leak=float(any(124902 in t.output_ids for t in turns)),
            think_leak_turn_frac=(sum(124902 in t.output_ids for t in turns) / len(turns)) if turns else 0.0,
            # last turn emitted a tool-call token but the episode ended there without an answer: the call text
            # was not parsed into a tool call (or the rollout was cut while calling)
            ended_on_call_text=float(bool(turns) and 124905 in turns[-1].output_ids and env_reward is not None
                                     and not row.get("submitted", True)),
        )
        return row
    except Exception as exc:  # noqa: BLE001 - infrastructure failure: unscorable, not wrong
        row["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        return row
    finally:
        row["seconds"] = round(time.monotonic() - t0, 1)
        try:
            session.close()
        except Exception as exc:  # noqa: BLE001
            row["cleanup_error"] = f"{type(exc).__name__}: {exc}"
        save_rollout(args.output, row, task, result_payload, trace_payload)
        status = "error" if row.get("error") or row.get("capture_error") else "graded"
        print(f"ROLLOUT step={row.get('intended_step', '?')} task={row['task']} "
              f"id={row['rollout_id'][:8]} status={status} correct={row.get('correct')} "
              f"reward={row.get('reward')} tools={row.get('tool_calls')} "
              f"tokens={row.get('generated_tokens')} seconds={row['seconds']:.1f}", flush=True)


def group_advantages(group, std_floor=0.0):
    """TRL AsyncGRPO advantage: (r - mean) / (max(std, std_floor) + 1e-8) over the scorable rollouts of one group."""
    scored = [r for r in group if r["reward"] is not None and r["sequences"]]
    ids = {id(r) for r in scored}
    if len(scored) < 2:
        for r in group:
            r["advantage"] = 0.0
        return False
    vals = [r["reward"] for r in scored]
    # Stratified root sampling: self-normalized weighted baseline (all weights 1.0 otherwise = TRL behaviour)
    ws = [float(r.get("root_weight", 1.0)) for r in scored]
    wsum = sum(ws) or 1.0
    mean = sum(w * v for w, v in zip(ws, vals)) / wsum
    std = math.sqrt(sum(w * (v - mean) ** 2 for w, v in zip(ws, vals)) / wsum)
    for r in group:
        r["advantage"] = (r["reward"] - mean) / (max(std, std_floor) + 1e-8) if id(r) in ids else 0.0
    return std > 0


def grpo_stats(groups, scored, optimizer):
    """Reward / advantage distribution and optimizer state for one update (logged to metrics.jsonl)."""
    def stats(vals, name):
        if not vals:
            return {}
        m = sum(vals) / len(vals)
        return {f"{name}_mean": m, f"{name}_std": math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals)),
                f"{name}_min": min(vals), f"{name}_max": max(vals)}
    rewards = [float(r["reward"]) for r in scored]
    advs = [float(r.get("advantage", 0.0)) for r in scored]
    nonzero = [a for a in advs if a != 0.0]
    group_std, hist = [], {}
    for gr in groups:
        vals = [float(r["reward"]) for r in gr if r["reward"] is not None]
        if len(vals) > 1:
            m = sum(vals) / len(vals)
            group_std.append(math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals)))
        k = sum(1 for r in gr if r.get("correct") == 1)
        hist[str(k)] = hist.get(str(k), 0) + 1
    out = {**stats(rewards, "reward"), **stats(advs, "advantage"),
           "advantage_abs_mean": (sum(abs(a) for a in nonzero) / len(nonzero)) if nonzero else 0.0,
           "advantage_pos_frac": (sum(a > 0 for a in nonzero) / len(nonzero)) if nonzero else 0.0,
           "advantage_nonzero_frac": (len(nonzero) / len(advs)) if advs else 0.0,
           "group_reward_std_mean": (sum(group_std) / len(group_std)) if group_std else 0.0,
           "group_correct_hist": hist,
           "lr": optimizer.param_groups[0]["lr"]}
    out.pop("reward_mean", None)  # already logged as reward_mean above
    return out


def unknown_tool_contrast(group, cap=3):
    """True when the group's scored rollouts differ in (capped) unknown-tool calls or capped tool outputs, i.e. the
    tool-error / capped-output penalty alone gives the group a learning signal."""
    vals = {(min(cap, r.get("unknown_tool_calls") or 0), min(4, r.get("capped_tool_outputs") or 0))
            for r in group if r.get("reward") is not None and r["sequences"]}
    return len(vals) > 1


def reward_contrast(group, min_spread):
    """True when the group's scored rewards spread by at least min_spread (0 = off): an all-correct group whose
    efficiency / context terms still rank its rollouts."""
    vals = [float(r["reward"]) for r in group if r.get("reward") is not None and r["sequences"]]
    return bool(min_spread) and len(vals) > 1 and max(vals) - min(vals) >= min_spread


def keep_group(group, tool_contrast=False, min_reward_spread=0.0):
    """Dynamic sampling: keep a group with correctness contrast, or (optionally) unknown-tool contrast, or
    (optionally) a reward spread of at least min_reward_spread."""
    return (correctness_contrast(group) or (tool_contrast and unknown_tool_contrast(group))
            or reward_contrast(group, min_reward_spread))


def skip_reason(visits, coin, args):
    """Dynamic skip for a revisit with earlier outcomes `visits` (c / w / m, no skips) and this visit's coin:
    "solved" if it was solved 8/8 once (unless coin < --solved-revisit-frac), "wrong" if its last visit was all
    wrong and coin < --skip-wrong-frac, else None (run it)."""
    if "c" in visits and coin >= getattr(args, "solved_revisit_frac", 0.0):
        return "solved"
    if visits[-1] == "w" and coin < getattr(args, "skip_wrong_frac", 0.5):
        return "wrong"
    return None


def correctness_contrast(group):
    """True when the group's rollouts differ in correctness (not only in efficiency bonus)."""
    vals = {r.get("correct") for r in group if r.get("reward") is not None and r["sequences"]}
    return len(vals) > 1


# ---------------------------------------------------------------- training
def build_model(args):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM
    from trl.trainer.utils import add_fused_lm_head

    from pirl.lfm2_packing import enable_packed_convolutions

    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, dtype=torch.bfloat16, attn_implementation="kernels-community/flash-attn3"
    ).cuda()
    add_fused_lm_head(model, temperature=TEMPERATURE)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.0, target_modules=LORA_TARGETS, task_type="CAUSAL_LM"))
    if getattr(args, "fp8_base", False):
        from pirl.fp8_emulation import emulate_vllm_fp8, use_vllm_fp8_kernels

        if getattr(args, "fp8_mode", "vllm") == "vllm":
            print(f"fp8 base: vLLM kernels on {use_vllm_fp8_kernels(model)} linears", flush=True)
        else:
            print(f"fp8 base emulation on {emulate_vllm_fp8(model)} linears", flush=True)
    enable_packed_convolutions(model)
    if args.compile:
        # Fuses PEFT's scale/add, casts and norms per layer: 1,516 -> 2,265 tok/s at 16k tokens on the L4.
        for layer in model.get_base_model().model.layers:
            layer.compile()
    return model


def pack(samples, budget):
    """First-fit-decreasing packing of (sequence, advantage) samples into rows of <= budget tokens."""
    rows = []
    for s in sorted(samples, key=lambda s: len(s[0].input_ids), reverse=True):
        n = len(s[0].input_ids)
        if n > budget:
            raise ValueError(f"Training sequence has {n} tokens, exceeding the {budget}-token budget")
        for row in rows:
            if row["tokens"] + n <= budget:
                row["items"].append(s)
                row["tokens"] += n
                break
        else:
            rows.append({"items": [s], "tokens": n})
    return rows


def lr_at(args, step):
    """Learning rate for update number `step` (0-based)."""
    if getattr(args, "lr_schedule", "constant") != "cosine":
        return args.lr
    warm = max(0, getattr(args, "warmup_steps", 0))
    if step < warm:
        return args.lr * (step + 1) / warm
    span = max(1, (getattr(args, "schedule_steps", 0) or args.steps) - warm)
    frac = min(1.0, (step - warm) / span)
    lo = getattr(args, "min_lr", 0.0)
    return lo + 0.5 * (args.lr - lo) * (1 + math.cos(math.pi * frac))


def train_step(model, optimizer, samples, args):
    """One clipped-ratio policy-gradient step, token-normalized over the whole step (TRL AsyncGRPO loss)."""
    import torch

    stats = {"ratio_sum": 0.0, "kl_sum": 0.0, "entropy_sum": 0.0, "clip_low": 0.0, "clip_high": 0.0, "count": 0.0}
    per_rollout = {}
    loss_sum = 0.0
    if not samples:
        return {"skipped": True}
    n_tokens = sum(sum(seq.completion_mask[1:]) for seq, _ in samples)
    if n_tokens <= 0:
        return {"skipped": True, "skip_reason": "no supervised completion tokens"}
    rows = pack(samples, args.token_budget)
    model.train()
    t0 = time.time()
    for row in rows:
        ids, pos, mask, old, adv = [], [], [], [], []
        for seq, a in row["items"]:
            ids += seq.input_ids
            pos += list(range(len(seq.input_ids)))
            mask += seq.completion_mask
            old += seq.old_log_probs
            adv += [a] * len(seq.input_ids)
        dev = "cuda"
        ids = torch.tensor([ids], device=dev)
        pos = torch.tensor([pos], device=dev)
        mask = torch.tensor([mask], device=dev, dtype=torch.float32)
        old = torch.tensor([old], device=dev, dtype=torch.float32)
        adv = torch.tensor([adv], device=dev, dtype=torch.float32)
        # bf16 autocast keeps the fp32 LoRA master weights but runs their matmuls in bf16: PEFT otherwise
        # upcasts every activation to fp32 for the adapter path (+27% tokens/s measured at 16k tokens).
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=not args.no_autocast):
            out = model(input_ids=ids, position_ids=pos, labels=ids.masked_fill(mask == 0, -100), fused_lm_head=True)
        logp, entropy = out.log_probs.float(), out.entropy
        mask, old, adv = mask[:, 1:], old[:, 1:], adv[:, 1:]
        log_ratio = (logp - old) * mask
        ratio = torch.exp(log_ratio)
        clipped = torch.clamp(ratio, 1 - args.epsilon, 1 + args.epsilon_high)
        if getattr(args, "is_correction", "ppo") == "tis":
            # Gradient of exp(logp - logp.detach()) is grad logp; the weight corrects sampler mismatch.
            weight = torch.clamp(ratio.detach(), max=args.tis_cap)
            per_token = -weight * adv * torch.exp(logp * mask - (logp * mask).detach())
            with torch.no_grad():
                stats["tis_capped"] = stats.get("tis_capped", 0.0) + (ratio.detach() > args.tis_cap)[mask > 0].sum().item()
        else:
            per_token = -torch.min(ratio * adv, clipped * adv)
        loss = (per_token * mask).sum() / n_tokens
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite policy loss; refusing to update the adapter")
        loss_sum += loss.detach().item()
        loss.backward()
        with torch.no_grad():
            m = mask > 0
            stats["ratio_sum"] += ratio[m].sum().item()
            stats["kl_sum"] += ((ratio - 1) - log_ratio)[m].sum().item()
            stats["entropy_sum"] += entropy[m].float().sum().item()
            stats["clip_low"] += ((ratio < 1 - args.epsilon) & (adv < 0))[m].sum().item()
            stats["clip_high"] += ((ratio > 1 + args.epsilon_high) & (adv > 0))[m].sum().item()
            stats["count"] += m.sum().item()
            offset = 0
            for seq, _ in row["items"]:
                start, end = max(0, offset - 1), offset + len(seq.input_ids) - 1
                selected = m[:, start:end]
                rid = str(seq.rollout_id)
                s = per_rollout.setdefault(rid, {"tokens": 0, "ratio_sum": 0.0, "kl_sum": 0.0,
                                                 "clip_sum": 0.0, "loss_sum": 0.0})
                s["tokens"] += selected.sum().item()
                s["ratio_sum"] += ratio[:, start:end][selected].sum().item()
                kl_values = (ratio - 1) - log_ratio
                s["kl_sum"] += kl_values[:, start:end][selected].sum().item()
                clipped_tokens = ((ratio < 1 - args.epsilon) & (adv < 0)) | ((ratio > 1 + args.epsilon_high) & (adv > 0))
                s["clip_sum"] += clipped_tokens[:, start:end][selected].sum().item()
                s["loss_sum"] += per_token[:, start:end][selected].sum().item()
                offset += len(seq.input_ids)
        del out, logp, entropy, loss, per_token, ratio
    params = [p for p in model.parameters() if p.requires_grad]
    grad_norm = torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm).item()
    if not math.isfinite(grad_norm):
        optimizer.zero_grad(set_to_none=True)
        raise FloatingPointError("Non-finite gradient norm; refusing to update the adapter")
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    c = max(stats["count"], 1.0)
    return {
        "ratio_mean": stats["ratio_sum"] / c, "kl": stats["kl_sum"] / c, "entropy": stats["entropy_sum"] / c,
        "loss": loss_sum,
        "per_rollout": {rid: {"trained_completion_tokens": int(s["tokens"]),
                              "ratio_mean": s["ratio_sum"] / max(1, s["tokens"]),
                              "kl": s["kl_sum"] / max(1, s["tokens"]),
                              "clip_frac": s["clip_sum"] / max(1, s["tokens"]),
                              "loss_mean": s["loss_sum"] / max(1, s["tokens"])}
                        for rid, s in per_rollout.items()},
        "clip_low_frac": stats["clip_low"] / c, "clip_high_frac": stats["clip_high"] / c,
        "clip_frac": (stats["clip_low"] + stats["clip_high"]) / c, "grad_norm": grad_norm,
        "tis_capped_frac": stats.get("tis_capped", 0.0) / c,
        "trained_rows": len(rows), "grad_accum_microbatches": len(rows), "trained_seqs": len(samples), "trained_completion_tokens": n_tokens,
        "forward_tokens": sum(r["tokens"] for r in rows), "train_s": round(time.time() - t0, 1),
        "peak_mem_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
    }


def save_adapter(model, path):
    tmp = path.with_name("." + path.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    model.save_pretrained(tmp)
    shutil.rmtree(path, ignore_errors=True)
    tmp.rename(path)


def recover_state(dest):
    """Restore the last published generation if a process died between directory renames."""
    old = dest.with_name("." + dest.name + ".old")
    if not dest.exists() and old.exists():
        old.rename(dest)
        sync_directory(dest.parent)
    if dest.exists() and not (dest / "state.json").is_file():
        raise RuntimeError(f"Incomplete checkpoint at {dest}; refusing to restart from zero")
    return dest.exists()


def save_state(dest, step, model, optimizer, cursor, sampler=None, extra=None):
    """Publish complete adapter/optimizer/RNG state; interrupted swaps recover their .old generation."""
    import torch

    recover_state(dest)
    tmp = dest.with_name("." + dest.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    model.save_pretrained(tmp / "adapter")
    torch.save(optimizer.state_dict(), tmp / "optimizer.pt")
    torch.save({"python": random.getstate(), "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []}, tmp / "rng.pt")
    write_json(tmp / "state.json", {"step": step, "task_cursor": cursor, "saved_at": time.time(),
                                   "sampler": sampler.state_dict() if sampler else None, **(extra or {})})
    for file in tmp.rglob("*"):
        if file.is_file():
            with file.open("rb") as stream:
                os.fsync(stream.fileno())
    sync_directory(tmp)
    old = dest.with_name("." + dest.name + ".old")
    shutil.rmtree(old, ignore_errors=True)
    if dest.exists():
        dest.rename(old)
        sync_directory(dest.parent)
    tmp.rename(dest)
    sync_directory(dest.parent)
    shutil.rmtree(old, ignore_errors=True)


def trim_jsonl(path, max_step):
    """Drop log lines from updates after the restored state (they were never saved)."""
    if not path.exists():
        return
    text = path.read_text()
    lines = text.splitlines()
    keep = []
    for i, line in enumerate(lines):
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if i == len(lines) - 1 and not text.endswith("\n"):
                break  # interrupted append; earlier complete records remain authoritative
            raise
        if row["step"] <= max_step:
            keep.append(line)
    tmp = path.with_name(f".{path.name}.trim.tmp")
    with tmp.open("w") as stream:
        stream.write("".join(line + "\n" for line in keep))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)
    sync_directory(path.parent)


def infra_failed(rollout):
    """The rollout never reached the model for infrastructure reasons (server/sandbox/network down)."""
    if rollout.get("error"):
        return True
    return any("no_turns" in f for f in rollout.get("findings") or [])


# ---------------------------------------------------------------- main loop
def main():
    args = arguments()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "STOP").exists():
        print("STOP marker is present; leaving this run stopped", flush=True)
        return
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import torch
    from smoldataenv_harbor.tasks import load_tasks

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    data_root = Path(args.data_root) if getattr(args, "data_root", None) else DATA
    tasks = load_tasks(data_root, "train")
    order = list(range(len(tasks)))
    random.Random(args.seed).shuffle(order)  # fixed task order; the cursor makes resume replay-free
    if getattr(args, "task_file", None):
        name_to_idx = {t["name"]: i for i, t in enumerate(tasks)}
        order = [name_to_idx[n] for n in json.loads(Path(args.task_file).read_text())]
    if args.task_limit:
        order = order[: args.task_limit]  # same subset for every ablation variant
    split_path = (str(data_root.resolve() / "datasets" / "train") if getattr(args, "data_root", None)
                  else httpx.get(args.server + "/smoldataenv/splits", timeout=30).json()["train"])
    reward_fn = rewards.REWARDS[args.reward]
    engine = Engine(args.vllm_url)

    model = build_model(args)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0, fused=True)
    step, cursor = 0, 0
    latest = out / "state_latest"  # saved after every update: resume loses at most the update in flight
    state = {}
    if recover_state(latest):
        state = json.loads((latest / "state.json").read_text())
        step, cursor = state["step"], state["task_cursor"]
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file

        set_peft_model_state_dict(model, load_file(str(latest / "adapter" / "adapter_model.safetensors")))
        optimizer.load_state_dict(torch.load(latest / "optimizer.pt"))
        if (latest / "rng.pt").exists():
            rng = torch.load(latest / "rng.pt", map_location="cpu", weights_only=False)
            random.setstate(rng["python"])
            torch.set_rng_state(rng["torch"])
            if rng["cuda"]:
                torch.cuda.set_rng_state_all(rng["cuda"])
        trim_jsonl(out / "metrics.jsonl", step)
        trim_jsonl(out / "rollouts.jsonl", step)
        print(f"resumed at step {step}, task cursor {cursor}", flush=True)
    elif (out / "metrics.jsonl").exists() and (out / "metrics.jsonl").stat().st_size:
        raise RuntimeError("Run has metrics but no recoverable state; refusing to restart from zero")
    elif getattr(args, "init_adapter", None):
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file

        result = set_peft_model_state_dict(model, load_file(str(Path(args.init_adapter) / "adapter_model.safetensors")))
        if getattr(result, "unexpected_keys", None):
            raise RuntimeError(f"--init-adapter has unexpected keys: {result.unexpected_keys[:5]}")
        print(f"initialized LoRA from {args.init_adapter} (fresh optimizer)", flush=True)
    config = {**vars(args), "model": MODEL, "revision": REVISION, "lora_targets": LORA_TARGETS,
              "task_order_seed": args.seed, "prefix_chain_template": PREFIX_CHAIN_TEMPLATE,
              "sampling": {"temperature": TEMPERATURE, "top_p": TOP_P, "top_k": TOP_K, "max_tokens": MAX_TOKENS},
              "artifact_schema_version": 1}
    if state and (out / "run_config.json").exists():
        previous = json.loads((out / "run_config.json").read_text())
        fixed = ("model", "revision", "lora_r", "lora_alpha", "groups", "num_generations", "sampler",
                 "hard_generations", "new_frac", "reward", "seed", "prefix_chain_template", "sampling",
                 "lr", "epsilon", "epsilon_high", "step_limit", "agent_timeout", "task_limit", "task_file",
                 "std_floor", "init_adapter", "is_correction", "tis_cap", "root_boost")
        changed = [key for key in fixed if key in previous and previous[key] != config.get(key)]
        allowed = [key for key in changed if key in (getattr(args, "allow_change", None) or [])]
        for key in allowed:
            append_jsonl(out / "config_changes.jsonl", {"step": state["step"], "key": key, "old": previous[key],
                                                        "new": config.get(key), "time": time.time()})
            print(f"config change allowed at step {state['step']}: {key} {previous[key]} -> {config.get(key)}", flush=True)
        changed = [key for key in changed if key not in allowed]
        if changed:
            raise RuntimeError(f"Resume would change the experiment configuration: {changed}")
    write_json(out / "run_config.json", config)
    adapters = out / "adapters"
    adapters.mkdir(exist_ok=True)
    if engine.http.get(f"{engine.url}/is_sleeping").json()["is_sleeping"]:
        engine.wake()
    current = f"{LORA_NAME}-v{step}"
    save_adapter(model, adapters / current)
    engine.load_adapter(current, adapters / current)
    pool = ThreadPoolExecutor(max_workers=args.concurrency)
    sampler = None
    if args.sampler == "regret":
        from pirl.curriculum import PromptSampler

        sampler = PromptSampler(order, base_n=args.num_generations, hard_n=args.hard_generations,
                                new_frac=args.new_frac, seed=args.seed)
        log_path = out / "rollouts.jsonl"
        if state.get("sampler") is not None:
            sampler.load_state_dict(state["sampler"])
        elif log_path.exists():  # legacy checkpoints: statistics only; new ones also save RNG state
            name_to_idx = {t["name"]: i for i, t in enumerate(tasks)}
            sampler.rebuild([json.loads(x) for x in log_path.read_text().splitlines() if x], name_to_idx)
        print(f"regret sampler: {sampler.summary()}", flush=True)
    print(f"batch: {args.groups} prompts, {args.num_generations} base rollouts/prompt; "
          f"target={args.steps} updates; sampler={args.sampler}. Prompt visits include revisits.", flush=True)
    if not step:
        save_state(latest, 0, model, optimizer, 0, sampler)

    infra_streak = 0
    pending_batch = None
    attempt = 0
    stream = getattr(args, "stream_groups", 0)
    if stream and sampler:
        raise SystemExit("--stream-groups needs --sampler sequential")
    dynamic = getattr(args, "dynamic_sampling", False)
    if dynamic and not stream:
        raise SystemExit("--dynamic-sampling needs --stream-groups")
    run_target = getattr(args, "async_groups", 0)
    if run_target and not stream:
        raise SystemExit("--async-groups needs --stream-groups")
    if stream and args.concurrency < (run_target or args.groups + stream) * args.num_generations:
        raise SystemExit("--stream-groups: --concurrency must cover (async_groups or groups + stream_groups) x "
                         "num_generations")
    skip_on = dynamic and getattr(args, "dynamic_skip", False)
    task_hist = {}  # task index -> visit outcomes in position order: c (all correct), w (all wrong), m (mixed), s (skipped)

    def outcome(gr):
        vals = {r.get("correct") for r in gr if r.get("reward") is not None and r["sequences"]}
        return None if not vals else ("m" if len(vals) > 1 else ("c" if vals == {1.0} or vals == {1} else "w"))

    if skip_on:  # rebuild visit history from the logs (rollouts without "pos" predate this flag: first pass)
        visits = {}
        for name in ("rollouts.jsonl", "filtered_rollouts.jsonl"):
            if (out / name).exists():
                for line in (out / name).open():
                    r = json.loads(line)
                    v = visits.setdefault(r["batch_id"], {"pos": r.get("pos", -1), "idx": r["task_index"], "rows": []})
                    v["rows"].append({"reward": r.get("reward"), "correct": r.get("correct"), "sequences": True})
        events = [(v["pos"], v["idx"], outcome(v["rows"])) for v in visits.values()]
        if (out / "skipped_tasks.jsonl").exists():
            events += [(e["pos"], e["idx"], "s") for e in map(json.loads, (out / "skipped_tasks.jsonl").open())]
        for _, idx, o in sorted(events, key=lambda e: e[0]):
            if o:
                task_hist.setdefault(idx, []).append(o)
        print(f"dynamic skip: history for {len(task_hist)} tasks", flush=True)
    inflight = []  # streaming: {"pos", "idx", "futures", "model"} in submission order
    requeue = list(state.get("stream_pending") or [])  # positions in flight at the last save: resubmit first
    next_pos = state.get("stream_next", cursor)
    last_pos = args.steps * args.groups  # never submit past the last update's tasks
    if dynamic:
        last_pos = float("inf")  # dropped groups are replaced, so positions run past steps x groups (order wraps)
    if stream and getattr(args, "stop_at_pos", 0):
        last_pos = args.stop_at_pos
        requeue = [pos for pos in requeue if pos < last_pos]

    probe_n = getattr(args, "probe_hard", 0) if dynamic else 0
    probe_stats = {"started": 0, "extended": 0, "dropped": 0}

    def submit_group(pos):
        idx = order[pos % len(order)]
        ctx = {"run_id": out.name, "intended_step": step + 1, "batch_id": uuid.uuid4().hex, "batch_attempt": 1,
               "policy_version": step, "template": "prefix-chain" if PREFIX_CHAIN_TEMPLATE else "native", "pos": pos}
        visits = [o for o in task_hist.get(idx, []) if o != "s"]
        probe = bool(probe_n) and probe_n < args.num_generations and bool(visits) and visits[-1] == "w"
        n = probe_n if probe else args.num_generations
        futs = [pool.submit(run_rollout, args, split_path, idx, tasks[idx], current, reward_fn, ctx) for _ in range(n)]
        inflight.append({"pos": pos, "idx": idx, "futures": futs, "model": current, "probe": probe, "ctx": ctx})
        probe_stats["started"] += probe

    def resolve_probes(dropped):
        """Finished probe groups: complete them to num_generations if they show signal, else drop them."""
        for g in [g for g in inflight if g.get("probe") and all(f.done() for f in g["futures"])]:
            gr = [f.result() for f in g["futures"]]
            signal = (any(r.get("correct") == 1 for r in gr)
                      or (getattr(args, "dynamic_keep_tool_contrast", False) and unknown_tool_contrast(gr))
                      or sum(map(infra_failed, gr)) * 2 >= len(gr))  # outages go on to the infra check
            if signal:
                ctx = dict(g["ctx"], intended_step=step + 1, policy_version=step)  # same batch_id: one group
                g["futures"] += [pool.submit(run_rollout, args, split_path, g["idx"], tasks[g["idx"]], current,
                                             reward_fn, ctx) for _ in range(args.num_generations - len(gr))]
                g["probe"], g["model"] = False, current
                probe_stats["extended"] += 1
            else:
                inflight.remove(g)
                dropped.append((g, gr))
                probe_stats["dropped"] += 1
                if outcome(gr):
                    task_hist.setdefault(g["idx"], []).append(outcome(gr))

    skipped_now = []

    def want_more():
        if not (requeue or next_pos < last_pos):
            return False
        if run_target:
            return sum(not all(f.done() for f in g["futures"]) for g in inflight) < run_target
        return len(inflight) < args.groups + stream

    def fill():
        nonlocal next_pos
        while want_more():
            if requeue:
                submit_group(requeue.pop(0))
                continue
            for _ in range(len(order)):  # never loop forever when every task would be skipped
                if next_pos >= last_pos:
                    break
                idx = order[next_pos % len(order)]
                h = task_hist.get(idx, [])
                visits = [o for o in h if o != "s"]
                if not skip_on or not visits:
                    break
                coin = random.Random(f"{args.seed}-{idx}-{len(h)}").random()  # resume-safe coin per revisit
                reason = skip_reason(visits, coin, args)
                if not reason:
                    break
                append_jsonl(out / "skipped_tasks.jsonl", {"pos": next_pos, "idx": idx, "step": step + 1,
                                                           "last": "".join(h[-2:]), "reason": reason})
                task_hist[idx].append("s")
                skipped_now.append(next_pos)
                next_pos += 1
            if next_pos >= last_pos:
                break
            submit_group(next_pos)
            next_pos += 1

    def stream_extra():
        return {"stream_next": next_pos, "stream_pending": [g["pos"] for g in inflight] + requeue} if stream else None

    while step < args.steps:
        if (out / "STOP").exists():
            print(f"STOP requested; stopping at saved step {step}", flush=True)
            break
        t_step = time.time()
        # 1. generate G groups x N rollouts with the current adapter
        if stream:
            # keep groups + K groups (or async_groups unfinished groups) in flight; train on the first `groups`
            # that are complete
            skipped_now.clear()
            fill()
            dropped = []
            for k in probe_stats:
                probe_stats[k] = 0
            while True:
                if probe_n:
                    resolve_probes(dropped)
                if dynamic and len(dropped) < args.dynamic_max_drop:
                    for g in [g for g in inflight if all(f.done() for f in g["futures"]) and not g.get("probe")]:
                        gr = [f.result() for f in g["futures"]]
                        # mostly-infra groups stay, so the outage check below still sees them
                        if (not keep_group(gr, getattr(args, "dynamic_keep_tool_contrast", False),
                                           getattr(args, "keep_reward_spread", 0.0))
                                and sum(map(infra_failed, gr)) * 2 < len(gr)):
                            inflight.remove(g)
                            dropped.append((g, gr))
                            if outcome(gr):
                                task_hist.setdefault(g["idx"], []).append(outcome(gr))
                if dynamic or run_target:
                    fill()
                ready = [g for g in inflight if all(f.done() for f in g["futures"]) and not g.get("probe")]
                if len(ready) >= min(args.groups, sum(not g.get("probe") for g in inflight) or len(inflight)):
                    break
                if (out / "STOP").exists():
                    break
                wait([f for g in inflight for f in g["futures"] if not f.done()], timeout=30,
                     return_when=FIRST_COMPLETED)
            for g, gr in dropped:
                for r in gr:
                    append_jsonl(out / "filtered_rollouts.jsonl",
                                 {k: v for k, v in r.items() if k != "sequences"} | {"step": step + 1, "pos": g["pos"]})
            if (out / "STOP").exists():
                print(f"STOP requested; stopping at saved step {step} ({len(inflight)} groups in flight dropped)",
                      flush=True)
                break
            chosen = [g for g in inflight if all(f.done() for f in g["futures"]) and not g.get("probe")][: args.groups]
            if not chosen and not inflight and not requeue and next_pos >= last_pos:
                print(f"all task positions < {last_pos} are done; stopping at saved step {step}", flush=True)
                break
            for g in chosen:
                inflight.remove(g)
            batch = [g["idx"] for g in chosen]
            sizes = [len(g["futures"]) for g in chosen]
            groups = [[f.result() for f in g["futures"]] for g in chosen]
            gen_s = time.time() - t_step
            rollouts_all = [r for gr in groups for r in gr]
            failed = sum(infra_failed(r) for r in rollouts_all)
            if failed > args.max_infra_fail * len(rollouts_all):
                infra_streak += 1
                with (out / "infra_events.jsonl").open("a") as f:
                    f.write(json.dumps({"step": step + 1, "failed": failed, "rollouts": len(rollouts_all),
                                        "streak": infra_streak, "time": time.time()}) + "\n")
                print(f"infra outage: {failed}/{len(rollouts_all)} rollouts failed; resubmitting those groups "
                      f"(streak {infra_streak})", flush=True)
                if infra_streak >= args.max_infra_retries:
                    raise SystemExit(3)
                requeue = [g["pos"] for g in chosen] + requeue
                time.sleep(60)
                continue
        else:
            if pending_batch is None:
                if sampler:
                    batch = sampler.next_batch(args.groups, step + 1)
                    sizes = [sampler.n_rollouts(idx) for idx in batch]
                else:
                    batch = [order[(cursor + g) % len(order)] for g in range(args.groups)]
                    sizes = [args.num_generations] * len(batch)
                pending_batch = (batch, sizes, uuid.uuid4().hex)
            batch, sizes, batch_id = pending_batch
            attempt += 1
            context = {"run_id": out.name, "intended_step": step + 1, "batch_id": batch_id,
                       "batch_attempt": attempt, "policy_version": step,
                       "template": "prefix-chain" if PREFIX_CHAIN_TEMPLATE else "native"}
            # Largest groups first, so the slowest work starts earliest.
            futures = [
                (g, pool.submit(run_rollout, args, split_path, idx, tasks[idx], current, reward_fn, context))
                for g, idx in sorted(enumerate(batch), key=lambda x: -sizes[x[0]]) for _ in range(sizes[g])
            ]
            groups = [[] for _ in batch]
            for g, fut in futures:
                groups[g].append(fut.result())
            gen_s = time.time() - t_step
            rollouts_all = [r for gr in groups for r in gr]
            if (out / "STOP").exists():
                print(f"STOP requested; archived {len(rollouts_all)} attempts without applying update {step + 1}", flush=True)
                break
            failed = sum(infra_failed(r) for r in rollouts_all)
            if failed > args.max_infra_fail * len(rollouts_all):
                # Infrastructure outage, not policy behaviour: redo the same prompts instead of spending an
                # update on (mostly) empty groups.
                infra_streak += 1
                event = {"step": step + 1, "failed": failed, "rollouts": len(rollouts_all), "streak": infra_streak,
                         "errors": sorted({(r.get("error") or (r.get("findings") or [""])[0])[:120]
                                           for r in rollouts_all if infra_failed(r)})[:5], "time": time.time()}
                with (out / "infra_events.jsonl").open("a") as f:
                    f.write(json.dumps(event) + "\n")
                print(f"infra outage: {failed}/{len(rollouts_all)} rollouts failed; retrying batch "
                      f"(streak {infra_streak})", flush=True)
                if infra_streak >= args.max_infra_retries:
                    raise SystemExit(3)  # the supervisor restarts us; state_latest holds the last good update
                time.sleep(60)
                continue
        infra_streak = 0
        if stream:
            for idx, gr in zip(batch, groups):
                if outcome(gr):
                    task_hist.setdefault(idx, []).append(outcome(gr))
        if sampler:
            prev = {idx: sampler.stats.get(idx) for idx in batch}
            for idx, gr in zip(batch, groups):
                scored_g = [r for r in gr if r["reward"] is not None]
                sampler.update(idx, sum(r.get("correct") == 1 for r in scored_g), len(scored_g), step + 1)
            revived = sum(1 for idx in batch if prev[idx] and prev[idx]["s"] == 0 and sampler.stats.get(idx, {}).get("s", 0) > 0)
        contrast = [group_advantages(gr, getattr(args, "std_floor", 0.0)) for gr in groups]
        correct_contrast = [correctness_contrast(gr) for gr in groups]
        rollouts = [r for gr in groups for r in gr]
        boost = getattr(args, "root_boost", 1.0)
        samples = [(seq, r["advantage"] * float(r.get("root_weight", 1.0)) * (boost if i == 0 else 1.0))
                   for r in rollouts for i, seq in enumerate(r["sequences"])
                   if args.train_zero_adv or r["advantage"] != 0.0]

        # 2. hand the GPU to the trainer
        t_sync = time.time()
        if not args.no_sleep:
            engine.sleep(mode="keep" if inflight else "abort")
        torch.cuda.reset_peak_memory_stats()
        for g in optimizer.param_groups:
            g["lr"] = lr_at(args, step)
        tstats = train_step(model, optimizer, samples, args)
        per_rollout_stats = tstats.pop("per_rollout", {})
        for r in rollouts:
            r["optimizer_metrics"] = per_rollout_stats.get(r["rollout_id"], {"trained_completion_tokens": 0})
            r["trained"] = r["optimizer_metrics"]["trained_completion_tokens"] > 0
        torch.cuda.empty_cache()
        step += 1
        cursor += args.groups
        # 3. publish the new policy and hand the GPU back
        new = f"{LORA_NAME}-v{step}"
        save_adapter(model, adapters / new)
        if not args.no_sleep:
            engine.wake()
        engine.load_adapter(new, adapters / new)
        current = new
        in_use = {g["model"] for g in inflight}  # in-flight rollouts keep sampling from their own adapter
        for old in sorted(adapters.iterdir()):
            if old.name.startswith(f"{LORA_NAME}-v") and old.name not in in_use | {current}:
                engine.unload_adapter(old.name)
                shutil.rmtree(old, ignore_errors=True)
        if step % args.save_steps == 0 or step == args.steps:
            save_state(out / f"checkpoint-{step}", step, model, optimizer, cursor, sampler, stream_extra())
        sync_s = time.time() - t_sync - tstats.get("train_s", 0)

        # 4. log everything the blog's collapse post-mortem says to watch
        scored = [r for r in rollouts if r["reward"] is not None]

        def mean(key, rows=scored):
            vals = [float(r[key]) for r in rows if r.get(key) is not None]
            return sum(vals) / len(vals) if vals else None

        rec = {
            "step": step, "prompt_passes": round(cursor / len(order), 3), "tasks": [tasks[i]["name"] for i in batch],
            "cursor": cursor, "rollouts": len(rollouts), "group_sizes": sizes,
            **({"revived_prompts": revived, **{f"pool_{k}": v for k, v in sampler.summary().items()}} if sampler else {}),
            "reward_mean": mean("reward"), "correct_rate": mean("correct"),
            "zero_adv_group_frac": 1 - sum(contrast) / len(contrast),
            "correctness_contrast_frac": sum(correct_contrast) / len(correct_contrast),
            "bonus_only_contrast_frac": sum(c and not cc for c, cc in zip(contrast, correct_contrast)) / len(contrast),
            "peak_context_mean": mean("peak_context"),
            **{f"{k}_{h}": v for h in sorted({r.get("harness") for r in rollouts} - {None})
               for k, v in (("correct_rate", mean("correct", [r for r in scored if r.get("harness") == h])),
                            ("peak_context_mean", mean("peak_context", [r for r in scored if r.get("harness") == h])),
                            ("rollouts", sum(1 for r in rollouts if r.get("harness") == h)))},
            "root_alt_frac": (sum(r.get("root_stratum") == "alt" for r in rollouts) / len(rollouts)) if rollouts else None,
            "root_unmatched": sum(1 for r in rollouts if r.get("root_weight") is not None and r.get("root_stratum") is None
                                  and r.get("sequences")),
            "root_weight_mass_max_dev": max((abs(sum(float(r.get("root_weight", 1.0)) for r in gr) - len(gr))
                                             for gr in groups), default=0.0),
            "root_pool_p_alt_mean": (lambda v: sum(v) / len(v) if v else None)(
                [r["root_p_alt"] for r in rollouts if r.get("root_p_alt") is not None]),
            "root_alt_correct": sum(1 for r in rollouts if r.get("root_stratum") == "alt" and r.get("correct") == 1),
            "peak_context_le_10k": mean("peak_le_10k"),
            "submit_rate": mean("submitted", rollouts), "truncated_rate": mean("truncated"),
            "context_exhausted_rate": mean("context_exhausted", rollouts),
            "tool_calls_mean": mean("tool_calls"), "model_calls_mean": mean("model_calls"),
            "unknown_tool_calls_mean": mean("unknown_tool_calls", rollouts),
            **{f"unknown_tool_calls_{h}": mean("unknown_tool_calls", [r for r in rollouts if r.get("harness") == h])
               for h in sorted({r.get("harness") for r in rollouts} - {None})},
            "capped_tool_outputs_mean": mean("capped_tool_outputs", rollouts),
            **{f"capped_tool_outputs_{h}": mean("capped_tool_outputs", [r for r in rollouts if r.get("harness") == h])
               for h in sorted({r.get("harness") for r in rollouts} - {None})},
            "own_context_mean": mean("own_context"),
            "generated_tokens_mean": mean("generated_tokens"), "final_context_mean": mean("final_context"),
            "unscorable": len(rollouts) - len(scored),
            "capture_errors": sum(1 for r in rollouts if r.get("capture_error")),
            "infra_errors": sum(1 for r in rollouts if r.get("error")),
            "cleanup_errors": sum(1 for r in rollouts if r.get("cleanup_error")),
            "retries": sum(r.get("retries", 0) for r in rollouts),
            "forks": sum(r.get("forks", 0) for r in rollouts),
            "rollout_s_mean": mean("seconds", rollouts), "gen_s": round(gen_s, 1),
            "policy_lag_mean": (sum(step - 1 - r.get("policy_version", step - 1) for r in rollouts) / len(rollouts))
                               if rollouts else None,
            "groups_in_flight": len(inflight),
            **({"dyn_dropped_groups": len(dropped),
                "dyn_dropped_all_correct": sum(all(r.get("correct") == 1 for r in gr if r.get("reward") is not None)
                                               for _, gr in dropped),
                "dyn_next_pos": next_pos, "dyn_skipped": len(skipped_now),
                **({f"probe_{k}": v for k, v in probe_stats.items()} if probe_n else {}),
                "groups_running": sum(not all(f.done() for f in g["futures"]) for g in inflight)} if dynamic else {}),
            "sync_s": round(sync_s, 1), "step_s": round(time.time() - t_step, 1), **tstats,
            "process_fds": len(os.listdir("/proc/self/fd")),
            "process_threads": threading.active_count(),
            "openenv_loop_threads": sum(t.name == "openenv-sync-client-loop" for t in threading.enumerate()),
            "completed_at": time.time(),
            **grpo_stats(groups, scored, optimizer),
            "think_leak_rate": mean("think_leak", rollouts),
            "think_leak_turn_frac": mean("think_leak_turn_frac", rollouts),
            "ended_on_call_text_rate": mean("ended_on_call_text", rollouts),
        }
        append_jsonl(out / "metrics.jsonl", rec)
        for r in rollouts:
            append_jsonl(out / "rollouts.jsonl", {k: v for k, v in r.items() if k != "sequences"} | {"step": step})
        save_state(latest, step, model, optimizer, cursor, sampler, stream_extra())
        for r in rollouts:
            append_jsonl(out / "rollout_metrics.jsonl", {"event": "optimizer_update", "step": step,
                         "rollout_id": r["rollout_id"], "advantage": r["advantage"],
                         "trained": r["trained"], **r["optimizer_metrics"]})
        write_json(out / "results.json", {"run_id": out.name, "completed_steps": step,
                   "target_steps": args.steps, "status": "complete" if step >= args.steps else "running",
                   "latest_update": rec})
        def display(key, digits=4):
            value = rec.get(key)
            return f"{value:.{digits}f}" if isinstance(value, (float, int)) else "-"
        print(f"UPDATE {step}/{args.steps} correct={display('correct_rate', 3)} reward={display('reward_mean', 3)} "
              f"zero_adv={display('zero_adv_group_frac', 3)} ctx={display('context_exhausted_rate', 3)} "
              f"loss={display('loss')} ratio={display('ratio_mean', 6)} clip={display('clip_frac')} "
              f"grad={display('grad_norm')} seconds={rec['step_s']} fds={rec['process_fds']} "
              f"client_loops={rec['openenv_loop_threads']}", flush=True)
        pending_batch = None
        attempt = 0
    # Streaming leaves groups in flight at exit; their positions are in the saved state (stream_pending) and are
    # resubmitted on resume, so do not wait up to agent_timeout for rollouts nobody will train on.
    abandoned = len(inflight) if stream else 0
    if abandoned:
        print(f"abandoning {abandoned} in-flight groups (positions saved for resubmission)", flush=True)
        if not args.no_sleep:
            try:
                engine.sleep(mode="abort")  # abort their vLLM requests so the agents end quickly
            except Exception as exc:  # noqa: BLE001
                print(f"vLLM abort-sleep failed: {exc}", flush=True)
        pool.shutdown(wait=False, cancel_futures=True)
    else:
        pool.shutdown()
    engine.http.close()
    if step < args.steps:
        write_json(out / "results.json", {"run_id": out.name, "completed_steps": step,
                   "target_steps": args.steps, "status": "stopped"})
    if abandoned:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)  # worker threads still block in Harbor calls; interpreter exit would join them


if __name__ == "__main__":
    main()
