# Handoff to Codex (2026-10-11): 1,000-task training with 4–5 min updates

You now own the execution of this project. Read `PLAN.md` (full log), `PROJECT_BRIEF.md` and `results/phase2_summary.md` first.

## User requirements (non-negotiable)
- Model **LiquidAI/LFM2.5-2.6B** + LoRA, **Pi harness**, learns **by itself**: RL only, no prompt hints, no teacher data.
- **FP8 on both sides.** vLLM `--quantization fp8`; trainer `--fp8-base --fp8-mode vllm` (vLLM's own kernels in the trainer); `--is-correction tis`.
- **Context under about 15k:** vLLM `max_model_len 16384`, 4,096 output cap. **Thinking OFF.**
- **Reward = the blog's** (`--reward tool`): correctness × (1 + 0.1·15/(15+tool_calls)); wrong/no answer = 0. Std floor 0.02.
- **Next:** train on **all 1,000 train tasks** (medium+hard; `../SmolDataEnvs/prepared/datasets/train`), starting from
  the best adapter. Arm A's final adapter is the natural choice: `runs/p2_A_tool/state_latest/adapter`; the r2 start point is `checkpoints/r2-final`.
- **Speed:** **4–5 min per update, generation + training** (now about 7 min: generation 5.3 + training 1.6).
- **Long jobs must run in tmux** (the user closes the laptop). Session `dataagent-p2` exists; add windows or create a new session.
- Keep OpenEnv/Harbor/TRL source unmodified (runtime patches only, as in `serve_harbor.py`). Back up files before edits.
  Never use `pgrep -f`/`pkill -f` patterns (they self-match): get PIDs from listening ports (`ss -ltnp`).

## Current stack
- `restart_vllm.sh` (CHAIN=1 SEQS=64 [KV=fp8]) → FP8 vLLM :8000, 16k, sleep mode, LoRA r32, 0.62 GPU.
- `serve_harbor.py` → Harbor + capture proxy :8201 (train/test/eval splits). Restart it to pick up new patches.
- `train_grpo.sh` (TASKS, CHAIN, CONC, TIMEOUT env knobs), `run_p2_arm.sh` (arm template), `eval_checkpoint.sh`
  (CHAIN=… ./eval_checkpoint.sh <adapter> <name> [val118|test250]), `analysis/plot_metrics.py` (run with
  `/home/ubuntu/STT/LLMPerfLab/.conda-env/bin/python`), `view_rollouts.py`, `analysis/behaviour.py`.
- Trainer: `../SmolDataEnvs/pirl/grpo.py` (+ reward.py, fp8_emulation.py, evaluate.py). Tests:
  `cd ../SmolDataEnvs && .venv/bin/python -m pytest tests -q` (54 pass).

## Speed analysis done (arm A, 704 trials at 64 concurrent rollouts)
- Per rollout: env setup mean **46 s** (median 37, max 272), agent 92 s (model generation 65 s = **7.8 s per model call**,
  tools 19 s), teardown mean **25 s** (max 207). Generation per update = the slowest rollout, so the Docker tails dominate.
- vLLM: 0 preemptions, 94.6% prefix-cache hit. Decode benchmark (`analysis/decode_bench.py`, 64 concurrent ~4.7k-token
  prompts, 256 out tokens): **1,709 output tok/s** with BF16 KV (warm cache).

## In-progress steps you inherit (verify state first)
1. **Data prefetch** (`prefetch_data.py`, tmux window `prefetch` in `dataagent-p2`): downloads all 329 datasets
   (train+test+eval, about 10 GB) to `../data_cache/<BUCKET_PREFIX>/` with a `.complete` marker. Check it finished; rerun if not.
2. **`serve_harbor.py install_fast_setup()`** (written, **not yet tested**):
   - (a) points each task at its prebuilt base image (skips the per-rollout `docker compose build`);
   - (b) bind-mounts the cached data read-only at `/home/user/input` (the task's pull_bucket healthcheck then skips its download).
   - `install_fast_teardown()` (SIGKILL before `compose down`) is already active.
   - **Test:** restart Harbor, run about 64 rollouts at 64 concurrency, compare setup/teardown from `trials/*/result.json` phase timestamps.
3. **FP8 KV cache** (`KV=fp8 ./restart_vllm.sh`) was about to be benchmarked. The last command was interrupted, so
   **vLLM may be down or half-restarted.** Check `curl localhost:8000/health` and `runs/vllm.mode`.
   - Measure decode speed (decode_bench.py) **and** sampler/trainer KL. FP8 KV adds mismatch, since the trainer's attention is BF16.
   - Adopt it only if KL stays small (in-training KL is about 0.006 now; TIS cap 2).
4. Other levers to consider:
   - training-side micro-batch `--token-budget 32768`;
   - capping the slowest rollouts;
   - limiting concurrent Docker operations;
   - `network_mode: host` (security tradeoff).
   - Keep the agent timeout ≥ 240 s unless the data show negligible loss (r2: 240 s lost 0.6% of correct rollouts, 180 s lost 7.9%).

## Deliverables
1. Smoke-test each speed change; report measured gains.
2. Launch the 1,000-task GRPO run in tmux: blog reward, 8 tasks × 8 rollouts per update, chain template, FP8, 16k. Use the
   regret or sequential sampler (your call; justify). Pick a sensible number of updates and checkpoint cadence, and
   evaluate on val118 periodically (consider 4 rollouts/task for tighter error bars).
3. Confirm the first 2–3 updates hit about 4–5 min, then report back: what changed, the numbers, how to monitor, and the ETA.
   Update `PLAN.md`. Do not block for hours; leave the run going in tmux.
