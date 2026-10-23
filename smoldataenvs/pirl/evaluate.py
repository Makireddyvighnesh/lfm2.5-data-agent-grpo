"""Pass@1 of a served policy inside one harness (default Pi) on fixed SmolDataEnvs tasks.

Every rollout is written to <output>/rollouts/<task>.json; rerunning the same command retries only
missing or ungraded tasks. The summary goes to <output>/summary.json (and --results-json if given).
"""

import argparse
import hashlib
import json
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pirl.common import DATA, MAX_TOKENS, MODEL, PREFIX_CHAIN_TEMPLATE, ROOT, TEMPERATURE, TOP_K, TOP_P
from pirl.artifacts import write_json as atomic_json
from pirl.harbor_client import harbor_client


def write_json(path, value):
    atomic_json(path, value)


def pi_session_stats(trial_dir):
    """Tool calls and per-call output lengths from pi's own session log (pi writes no ATIF)."""
    sessions = sorted((trial_dir / "agent" / "pi" / "sessions").glob("*.jsonl"))
    if not sessions:
        return None
    calls, outputs, stop_reasons = 0, [], []
    for raw in sessions[-1].read_text().splitlines():
        try:
            record = json.loads(raw)
        except ValueError:
            continue
        message = record.get("message") or {}
        if record.get("type") != "message" or message.get("role") != "assistant":
            continue
        content = message.get("content") or []
        if isinstance(content, list):
            calls += sum(1 for c in content if c.get("type") == "toolCall")
        outputs.append(int((message.get("usage") or {}).get("output") or 0))
        stop_reasons.append(message.get("rawStopReason") or message.get("stopReason"))
    return {"tool_calls": calls, "outputs": outputs, "stop_reasons": stop_reasons}


def eval_tier_session_class():
    """HarborSession that requests an *eval* rollout: for engines without token ids (e.g. Tinker's
    OpenAI endpoint), where the proxy refuses a training sampling policy."""
    from harbor_env.harness import HarborSession

    class EvalSession(HarborSession):
        def wait_for_completion(self, timeout_s=None):
            try:
                self.result = self._env.run_rollout(
                    split=self._split, task_index=self._task_index, harness=self._harness,
                    sandbox=self._sandbox, reward_key=self._reward_key, llm_url=self._llm_url,
                    model=self._model, agent_step_limit=self._agent_step_limit,
                    agent_timeout_sec=self._agent_timeout_sec, purpose="eval",
                    eval_sampling={"temperature": TEMPERATURE, "top_p": TOP_P},
                )
            except Exception:  # noqa: BLE001 - same contract as HarborSession: failure -> no result
                self.result = None
                return 1
            return 0 if self.result.ok else 1

    return EvalSession


def episode(args, split_path, index, task):
    from harbor_env.harness import HarborSession

    if args.eval_tier:
        HarborSession = eval_tier_session_class()  # noqa: N806
    session = HarborSession(
        env=harbor_client(args.server, message_timeout_s=1800),
        owns_env=True,
        split=split_path,
        task_index=index,
        instruction=task["instruction"],
        harness=args.harness,
        sandbox="docker",
        llm_url=args.vllm_url,
        model=args.model,
        reward_key="correctness,reward",
        sampling=None if args.eval_tier else {"temperature": args.temperature, "top_p": TOP_P, "top_k": TOP_K},
        agent_step_limit=args.step_limit,
        agent_timeout_sec=args.agent_timeout,
    )
    try:
        session.wait_for_completion()
        result = session.result
        if result is None:
            raise RuntimeError("harbor rollout call failed")
        capture_file = f"captures/{uuid.uuid4().hex}.json.gz"
        atomic_json(Path(args.output) / capture_file, {"schema_version": 1,
                    "task": {k: task[k] for k in ("name", "instruction", "difficulty")},
                    "result": result.model_dump(mode="json")}, compressed=True)
        correctness = session.verify([]).env_reward
        if correctness not in (0, 1):
            raise RuntimeError(f"No verifier grade: {result.error}")
        trial = Path(args.trials) / result.trial_name
        # prime-agent (a pi fork) writes the same session log; other harnesses write ATIF.
        stats = pi_session_stats(trial) if args.harness in ("pi", "prime-agent") else None
        verifier_out = (trial / "verifier" / "test-stdout.txt")
        verifier_text = verifier_out.read_text() if verifier_out.exists() else ""
        row = {
            "correctness": float(correctness),
            "atif": result.atif,
            "n_turns": result.n_turns,
            "n_roots": result.n_roots,
            "n_trainable_tokens": result.n_trainable_tokens,
            "budget_stops": result.budget_stop_count,
            "context_exhausted": any("context_budget_exhausted" in f for f in result.findings),
            "submitted": "no answer at" not in verifier_text,
            "trial": result.trial_name,
            "capture_file": capture_file,
        }
        try:
            trace = session.fetch_training_trace()
            completions = [len(t.completion_token_ids) for t in trace.turns]
            row["trajectory"] = trajectory(trace)
            peak = max((len(t.prompt_token_ids) + len(t.completion_token_ids) for t in trace.turns), default=0)
            row.update(
                trainable=True,
                peak_context=peak,
                peak_le_10k=float(peak <= 10_000),
                peak_le_15k=float(peak <= 15_000),
                generated_tokens=sum(completions),
                prompt_tokens_last=len(trace.turns[-1].prompt_token_ids) if trace.turns else 0,
                model_calls=len(trace.turns),
                truncated=any(n >= MAX_TOKENS for n in completions),
            )
        except ValueError as exc:
            row.update(trainable=False, trace_error=str(exc)[:300])
        if stats is None and (trial / "agent" / "trajectory.json").exists():
            from smoldataenv_harbor.environment import count_tools

            row["tool_calls"] = count_tools(trial / "agent" / "trajectory.json")
        if stats is not None:
            row["tool_calls"] = stats["tool_calls"]
            row.setdefault("truncated", any(n >= MAX_TOKENS for n in stats["outputs"]))
            row["truncated"] = row["truncated"] or "length" in stats["stop_reasons"]
        return row
    finally:
        session.close()


def trajectory(trace):
    """Conversation + per-model-call sampled tokens, from OpenEnv's validated capture."""
    entries = trace.to_trace_entries()
    calls = []
    for entry, turn in zip(entries, trace.turns):
        msg = ((entry.get("response") or {}).get("choices") or [{}])[0].get("message") or {}
        calls.append({
            "prompt_tokens": len(turn.prompt_token_ids),
            "prompt_token_ids": list(turn.prompt_token_ids),
            "content": msg.get("content"),
            "tool_calls": msg.get("tool_calls"),
            "completion_token_ids": list(turn.completion_token_ids),
            "logprobs": list(turn.per_token_logps),
            "loss_mask": list(turn.loss_mask),
        })
    last = entries[-1] if entries else {}
    final = ((last.get("response") or {}).get("choices") or [{}])[0].get("message") or {}
    messages = list((last.get("request") or {}).get("messages") or [])
    if final:
        messages.append({"role": "assistant", "content": final.get("content"), "tool_calls": final.get("tool_calls")})
    return {"messages": messages, "calls": calls}


def prompt_id(instruction):
    """Stable unique id per prompt (task instruction), shared by all rollouts of that prompt."""
    return hashlib.sha1(instruction.strip().encode()).hexdigest()[:16]


def summarize(rows, expected):
    graded = [r for r in rows if r.get("correctness") in (0, 1)]

    def mean(key):
        vals = [float(r[key]) for r in graded if r.get(key) is not None]
        return sum(vals) / len(vals) if vals else None

    return {
        "expected": expected,
        "graded": len(graded),
        "coverage": len(graded) / expected if expected else 0,
        "pass_at_1": mean("correctness"),
        "submit_rate": mean("submitted"),
        "truncated_rate": mean("truncated"),
        "context_exhausted_rate": mean("context_exhausted"),
        "trainable_rate": mean("trainable"),
        "atif_match_rate": (sum(r.get("atif") == "match" for r in graded) / len(graded)) if graded else None,
        "mean_tool_calls": mean("tool_calls"),
        "mean_model_calls": mean("model_calls"),
        "mean_roots": mean("n_roots"),
        "mean_generated_tokens": mean("generated_tokens"),
        "mean_seconds": mean("seconds"),
        "mean_peak_context": mean("peak_context"),
        "peak_le_10k_rate": mean("peak_le_10k"),
        "peak_le_15k_rate": mean("peak_le_15k"),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=MODEL, help="Served model (or LoRA adapter) name")
    p.add_argument("--label", required=True, help="Checkpoint identity, e.g. base or step-50")
    p.add_argument("--split", default="test", choices=["train", "test", "eval"])
    p.add_argument("--split-path", help="Served dataset path to use instead of the server's default for --split")
    p.add_argument("--data", default=str(DATA),
                   help="Prepared data root holding datasets/<split>/tasks (eval: the Data Agent project)")
    p.add_argument("--harness", default="pi")
    p.add_argument("--tasks", type=int, default=50, help="First N tasks of the sorted (filtered) split")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--step-limit", type=int, default=17)
    p.add_argument("--agent-timeout", type=int, default=300)
    p.add_argument("--server", default="http://127.0.0.1:8200")
    p.add_argument("--vllm-url", default="http://127.0.0.1:8000")
    p.add_argument("--trials", default=str(ROOT / "runs" / "trials"))
    p.add_argument("--output", required=True)
    p.add_argument("--results-json", help="Also copy the summary here")
    p.add_argument("--difficulty", help="Comma-separated filter, e.g. medium,hard")
    p.add_argument("--task-file", help="JSON list of task names to evaluate (in this order); overrides --tasks/--start")
    p.add_argument("--rollouts-per-task", type=int, default=1)
    p.add_argument("--temperature", type=float, default=TEMPERATURE,
                   help="Sampling temperature; keep the default to match training (token-faithful)")
    p.add_argument("--eval-tier", action="store_true",
                   help="Engine returns no token ids (e.g. Tinker OpenAI endpoint): request eval rollouts")
    args = p.parse_args()

    from smoldataenv_harbor.tasks import load_tasks

    tasks = load_tasks(args.data, args.split)
    wanted = set(args.difficulty.split(",")) if args.difficulty else None
    pool = [(i, t) for i, t in enumerate(tasks) if wanted is None or t["difficulty"] in wanted]
    if args.task_file:
        wanted_names = json.loads(Path(args.task_file).read_text())
        by_name = {t["name"]: (i, t) for i, t in pool}
        missing = [n for n in wanted_names if n not in by_name]
        if missing:
            raise SystemExit(f"--task-file names not in split {args.split}: {missing[:5]}")
        selected = [by_name[n] for n in wanted_names]
    else:
        selected = pool[args.start : args.start + args.tasks]
    work = [(i, t, k) for i, t in selected for k in range(args.rollouts_per_task)]
    # The server reports only its own train/test; other splits must be in its OPENENV_DATASETS.
    split_path = args.split_path or httpx.get(args.server + "/smoldataenv/splits", timeout=30).json().get(args.split)
    split_path = split_path or str(Path(args.data).resolve() / "datasets" / args.split)
    out = Path(args.output)
    identity = {
        "label": args.label, "model": args.model, "harness": args.harness, "split": args.split,
        "task_names": [t["name"] for _, t in selected], "temperature": args.temperature,
        "top_p": TOP_P, "top_k": TOP_K, "max_tokens": MAX_TOKENS, "step_limit": args.step_limit,
        "agent_timeout": args.agent_timeout,
        "instruction_hash": hashlib.sha256("".join(t["instruction"] for _, t in selected).encode()).hexdigest(),
        "difficulty_filter": args.difficulty, "rollouts_per_task": args.rollouts_per_task,
        "template": "prefix-chain" if PREFIX_CHAIN_TEMPLATE else "native",
    }
    ident_path = out / "identity.json"
    if ident_path.exists() and json.loads(ident_path.read_text()) != json.loads(json.dumps(identity)):
        raise SystemExit("Output already belongs to a different evaluation")
    write_json(ident_path, identity)

    def one(item):
        index, task, k = item
        path = out / "rollouts" / f"{task['name']}--r{k}.json"
        if path.exists():
            prev = json.loads(path.read_text())
            if prev.get("correctness") in (0, 1):
                return prev
        start = time.monotonic()
        pid = prompt_id(task["instruction"])
        row = {"prompt_id": pid, "rollout_id": f"{pid}-r{k}", "task": task["name"],
               "difficulty": task["difficulty"], "split": args.split, "label": args.label, "model": args.model,
               "harness": args.harness, "instruction": task["instruction"]}
        try:
            row.update(episode(args, split_path, index, task))
        except Exception as exc:  # noqa: BLE001 - persist ungraded rollouts for retry
            row.update(correctness=None, error=type(exc).__name__, detail=str(exc)[:500])
        row["seconds"] = round(time.monotonic() - start, 1)
        write_json(path, row)
        print(f"{task['name']:40s} {row.get('correctness')} calls={row.get('tool_calls')} "
              f"tok={row.get('generated_tokens')} {row['seconds']}s {row.get('error', '')}", flush=True)
        return row

    t0 = time.time()
    rows = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        for fut in as_completed([pool.submit(one, item) for item in work]):
            rows.append(fut.result())
            write_json(out / "progress.json", summarize(rows, len(work)))
    # One line per rollout, stable order; every rollout of a prompt shares its prompt_id.
    with (out / "rollouts.jsonl").open("w") as f:
        for r in sorted(rows, key=lambda r: r["rollout_id"]):
            f.write(json.dumps(r, default=str) + "\n")
    summary = {**identity, **summarize(rows, len(work)), "wall_s": round(time.time() - t0, 1)}
    summary.pop("task_names")
    summary["difficulty"] = {
        d: summarize([r for r in rows if r["difficulty"] == d],
                     args.rollouts_per_task * sum(t["difficulty"] == d for _, t in selected))
        for d in sorted({t["difficulty"] for _, t in selected})
    }
    write_json(out / "summary.json", summary)
    if args.results_json:
        write_json(Path(args.results_json), summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
