# Data Agent RL on one L4: project brief (2026-10-10)

## Goal (from the user)
- Train **LiquidAI/LFM2.5-2.6B** with RL so it **learns by itself**. That means no prompt hints, and the model must discover the best tools on its own.
- **Keep context under 10–15k tokens** per rollout.
- **Beat the multi-harness RL blog** (HF + Liquid, 2026-10-01). Its LFM2.5-2.6B results:
  - Base: 42.2% pass@1 averaged over OpenCode, Claude Code, Codex and Mini-SWE-Agent (OpenCode 33.6%, Mini-SWE 62.1%).
  - After 1,000 steps: 52.3% (OpenCode-only) and 54.2% (multi-harness).
  - Setup: 2×H100, about 128k context, 600 s agent timeout.
- Order: first **300 train tasks (medium+hard)**, then scale to **1,000 (medium+hard)**. Test every idea.

## Hardware and stack
- 1× NVIDIA L4 (23 GB, SM89), 16 CPUs, 60 GB RAM, local Docker sandboxes (1 CPU / 4 GB each). Python venv at `../SmolDataEnvs/.venv`.
- Code: `../SmolDataEnvs/pirl/`.
  - `grpo.py`: synchronous, GPU time-shared GRPO. vLLM sleeps while the LoRA trainer runs. It has a regret prompt sampler, adaptive group size, a `--reward` choice of correctness, calls or efficiency (efficiency includes a context term), and resumes via `run_resumable.sh`.
  - `evaluate.py`: pass@1 per harness.
  - `serve.py`: vLLM launcher with token ids and logprobs.
  - `reward.py`, `curriculum.py`.
  - Read `../SmolDataEnvs/NOTES.md` for the hard-won infrastructure fixes.
- This folder adds:
  - `serve_harbor.py`: Harbor + capture proxy on :8201 serving the eval and test splits, plus a prime-agent seam.
  - `prime_agent_harness.py`.
  - `prepare_eval.py`: the 144-task eval split with the grader fix.
  - `sandbox_image/`: pi + Prime Agent + OpenCode preinstalled.
- Data:
  - Train pool `../SmolDataEnvs/prepared/datasets/train`: 1,000 medium+hard tasks, the same as the blog's training set.
  - Test `../SmolDataEnvs/prepared/datasets/test`: 250 tasks, the same as the blog's test set.
  - Eval split `prepared/datasets/eval`: 144 tasks, no task overlap with train or test.
- Model serving: vLLM 0.25.1, LoRA r=32/α=64 on all linear layers. Trainer LoRA uses fp32 master weights with bf16 autocast. FP8 (`--quantization fp8`) works for serving.
- Trainer memory: peaks at 11 GiB for a 16k-token row and 17 GiB for a 32k row on the whole GPU. A 32k step takes about 6–7 min for 8 prompts × 4 rollouts.

## Measured baselines (base model, no training; pass@1, one rollout per task)

| Run | Harness | Context | T | Timeout | Score |
|---|---|---|---|---|---|
| BF16, test medium+hard (217 tasks) | Pi | 32k | 1.0 | 300 s | 0.258 |
| BF16, eval split (144) | Pi | 32k | 1.0 | 300 s | 0.306 (medium+hard 0.266) |
| FP8, test (250) | Pi | 32k | 1.0 | 300 s | **0.252** (medium+hard 0.221; FP8 vs BF16 McNemar p=0.24) |
| FP8, test (250) | OpenCode | 32k | 1.0 | 300 s | 0.040 |
| FP8, test (250) | Prime Agent | 32k | 1.0 | 300 s | 0.120 |
| FP8, test, partial (57 tasks) | OpenCode | 128k | 0.8 | 600 s | 0.105 |
| FP8, test (in progress) | Pi | 64k | 1.0 | 300 s | ~0.28 at 150/250 tasks |

## Diagnosis
- **Pi tools:** `read` (up to 2,000 lines / 50 KB), `bash` (fresh shell per call), `edit`, `write`.
  - 243 of 250 rollouts start with `read` on the whole CSV. The single tool result is about 14k tokens at the median.
  - pi's `read` description says "when you need the full file, continue with offset until complete".
- **Pi at 32k:** 40% of rollouts run out of context, and every one of those scores 0. Rollouts that don't run out score 0.42.
  - The 50 wrong answers include 21 cases of answering a number for a "Which X" question that needs a label.
- **Efficient behaviour exists but is rare:** only 25 of 754 Pi rollouts (3.3%) avoided a full-file read. Of those, `read` with a `limit` solved 4/6 at a mean peak of 8.6k tokens.
- **Verified solutions need little context:** the 4,677 trajectories in `FineEnvs/data-agent-sft` (strong agents, bash tool) use a median of 1,062 LFM tokens for the whole conversation. 96% are under 4k and 100% under 16k.
- **Prime Agent:** the model keeps calling a nonexistent top-level `bash` tool (Prime exposes only `ipython`). It solves when it switches to `ipython`.
- **Earlier L4 GRPO runs at 32k** (25–35 steps, 8×4):
  - Context-exhaustion rate stayed flat at about 0.4–0.5. Train correct-rate moved from 0.18 to 0.24–0.29.
  - No held-out evaluation of a trained checkpoint exists yet.
  - At step 1, 75% of groups had zero reward variance.

## Open design questions
1. How to enforce "context under 10–15k" while keeping a learning signal:
   - a hard engine limit (vLLM `max_model_len` of about 16k), versus
   - 32k with a context penalty or bonus on correct answers, versus
   - a curriculum that tightens the limit.
2. Exploration: the efficient strategy appears in about 3% of rollouts. Options are group size, temperature, oversampling large-file tasks, and the regret sampler.
3. Self-generated warm start: rejection-sampling fine-tuning on the model's own correct, low-context rollouts. Do we use teacher SFT data at all? The user said "learn by itself", so the default is no.
4. Template: native (higher base score, about 5 training rows per rollout) versus prefix-chain (1 row per rollout, lower base score).
5. Evaluation protocol and what "beat the blog" means here. We train and evaluate in Pi; the blog never used Pi.
