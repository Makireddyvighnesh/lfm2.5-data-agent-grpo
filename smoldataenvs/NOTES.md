# Pi-harness GRPO on one L4: working notes

Model `LiquidAI/LFM2.5-2.6B` (BF16, rev `654f946`), LoRA r=32/α=64 on all linear layers
(q,k,v,out_proj, w1,w2,w3, conv in_proj). Harness: **pi 1.0.0** via OpenEnv Harbor capture, local Docker
sandboxes. Data: the tutorial's pinned 1,000 medium+hard train tasks / 250 held-out test tasks
(`FineEnvs/SmolDataEnvs-harbor-{train,test}`, zero overlap).

## Layout

| Path | What |
|---|---|
| `pirl/common.py` | model, sampling constants, non-thinking chat template (+ prefix-chain fix) |
| `pirl/serve.py` | vLLM launcher (token ids, processed logprobs, runtime LoRA, sleep mode) |
| `pirl/harbor_server.py` | Harbor task API + capture proxy for local Docker (3 local fixes, below) |
| `pirl/grpo.py` | synchronous, GPU-time-shared GRPO trainer |
| `pirl/evaluate.py` | pass@1 in a harness on fixed test tasks |
| `pirl/reward.py`, `tests/` | reward (correctness + ≤0.1 efficiency bonus on correct only), unit tests |
| `pirl/svc.sh` | start/stop vLLM and Harbor by PID file |
| `sandbox_image/` | task base image + Node 22 + pi preinstalled |

Run order: `pirl/svc.sh start harbor`, `pirl/svc.sh start vllm --output runs/vllm_train --prefix-caching --sleep-mode --gpu-mem 0.62`,
then `python pirl/grpo.py --output runs/<name> ...`.

## Plan corrections (HANDOFF.md vs reality)

- `python3.12` is not on PATH → `uv venv --python 3.12`; `uv pip` instead of `pip`.
- Model changed per user goal: LFM2.5-2.6B instead of Qwen3.5-2B.
- Held-out split is `FineEnvs/SmolDataEnvs-harbor-test` (250 tasks: 33 easy, 118 medium, 99 hard);
  the train split is exactly 1,000 medium+hard (600 hard, 400 medium).
- `max_concurrent_envs` is the env var `MAX_CONCURRENT_ENVS` of the Harbor server (set to 16).
- Single-GPU AsyncGRPO: supported in principle (adapter-only LoRA sync over the filesystem, no NCCL),
  but **not usable here** — see "Why synchronous" below.

## Local fixes needed (none edit OpenEnv/Harbor/TRL source; all live in `pirl/harbor_server.py`)

1. **Docker is a snap with a private `/tmp`.** Harbor writes compose files to `/tmp`; compose cannot
   see them ("no such file or directory"). Fix: `TMPDIR=runs/tmp`.
2. **Capture proxy address.** OpenEnv `direct` exposure advertises `127.0.0.1`, which inside a
   container is the container. Fix: advertise the docker0 gateway (172.17.0.1).
3. **OpenEnv bug: pi rollouts that hit the context limit are FATAL.** pi writes no ATIF; OpenEnv
   rebuilds one from pi's session log but drops the message text, so `reconcile()` cannot recognise
   the proxy's zero-token "Step budget exhausted; stopping." reply → `turn_mismatch` FATAL → TRL's
   harness worker raises `CaptureContractError` and stops. Fix: replacement reader that keeps the text.
   (Worth upstreaming.)
4. **pi install skipped** when the image already has pi 1.0.0 (see speed table).

## Why synchronous (time-shared GPU) instead of AsyncGRPO

- pi's `read` returns up to 50 KB; most rollouts are 20–32k-token rows.
- Trainer peak (LoRA, grad ckpt, fused LM head): 11 GiB at 16k tokens, OOM at 24k/32k next to a
  vLLM holding 45% of the GPU. A vLLM small enough to coexist has KV for only ~5 concurrent rollouts.
- So each step: vLLM generates (62% of GPU, 508k-token KV) → `/sleep` level 1 (5.8 s, leaves 0.6 GiB)
  → trainer step on the whole GPU (peak 17 GiB at 32k) → save adapter → `/wake_up` (0.7 s) → load
  `trl-policy-v{N}`. Strictly on-policy; sleep+wake+adapter swap costs ~3.5 s/step.
- AsyncGRPO also hardcodes FA3 (`kernels-community/flash-attn3`) — that hub kernel *does* run on SM89.

## Correctness checks

- Packed LFM2 training: with the PyTorch conv fallback, packed vs separate logits differ by **19.5**
  (convolutions leak across sequences). With `kernels-community/mamba-ssm` causal_conv1d + `seq_idx`
  hook: **0.0**. `pirl/lfm2_packing.py` asserts this at startup.
- Token faithfulness (smoke, 2 steps): importance ratio **0.9998–0.9999**, KL 2e-4, clip frac ≤3e-4.
- Zero-init LoRA on vLLM samples the same tokens as base; logprobs differ slightly (punica numerics),
  which is why even step 0 samples through the adapter path.

## Ablations

### Speed (what worked)

| Change | Before | After |
|---|---|---|
| pi preinstalled in sandbox image (skip nvm+npm per rollout) | 48.7 s / rollout | 25.3 s (agent_setup 22 s → ~0) |
| vLLM prefix caching (99% of tokens were re-prefill) | 233 s wall / 5 rollouts (114 s mean) | 126 s (73 s mean); 65% hit rate |
| prefix-chain template (one training row per rollout) | 5 rows / rollout, each repeating context | 1.02 rows; eval 30% faster (84 → 61 s) |
| vLLM KV at 0.62 GPU (possible only with time-sharing) | 9.9k blocks (159k tok), preemptions | 31.8k blocks (508k tok) |
| bf16 autocast on LoRA path (fp32 master weights kept) | 1,190 tok/s train | 1,516 tok/s (+27%) |
| bf16 LoRA weights (rejected: Adam updates ≪ bf16 ulp) | — | 1,730 tok/s (+45%) |
| skip zero-advantage rows in the train step | (measured in pilot) | |

### Template (what didn't work as hoped)

LFM's non-thinking prompt ends `<think></think>`, but the native template renders past assistant
turns *without* it, so every turn's prompt diverges from the previous prompt+completion → OpenEnv
cannot chain turns (`per_turn_capture_only`) → ~N× training tokens. Rendering history with
`<think></think>` fixes chaining (1.02 rows/rollout) but **lowers base pass@1 on the same 50 test
tasks from 0.40 to 0.26** (8 tasks native-only correct vs 1 chain-only; binomial p≈0.04).
Training uses the chain template (native would cost ~3.6× trainer compute); the trained adapters
are evaluated under **both** templates, and gains are judged against the native base (0.40).

### Baselines (pi, base model, first 50 test tasks, T=1.0, 4,096 cap, 32k context)

| Template | pass@1 | submit | ctx exhausted | tool calls | gen tokens | s/rollout |
|---|---|---|---|---|---|---|
| native | 0.40 | 0.54 | 0.38 | 4.2 | 889 | 84 |
| prefix-chain | 0.26 | 0.50 | 0.34 | 3.9 | 705 | 61 |

Context exhaustion (38%) is the dominant failure: two pi `read`s of a CSV fill the 32k window.
Kept as a fixed environment limit in training and eval, like the 4,096 output cap.

## Main run config (runs/main200)

- **Batch per optimizer update:** 8 prompts × 4 rollouts = 32 rollouts (user choice; 1 prompt/update
  was too noisy — the first 1×4 step had all 4 rollouts wrong, i.e. no signal).
- **Gradient accumulation:** the update's sequences are packed into ≤32k-token micro-batches; each
  micro-batch does forward+backward, loss normalized by the update's total completion tokens, then one
  AdamW step. `grad_accum_microbatches` in metrics.jsonl = micro-batches per update.
- **Epoch:** 1,000 train prompts / 8 = 125 updates. 200 updates = 1.6 epochs (2nd epoch revisits
  prompts in the same fixed order; resume continues from the saved cursor, never replays).
- 16 rollouts in flight (sandboxes); LR 2e-5 constant; ε 0.2/0.28; grad clip 1.0; reward = correctness
  + ≤0.1 efficiency bonus on correct answers; zero-advantage groups skipped in the update.
- Aborted earlier configs kept for reference: `runs/main200_aborted_g8` (1×8), `runs/aborted_1x4` (1×4).

### Two more failures found when scaling to 16 concurrent rollouts

5. **Harbor session capacity.** `MAX_CONCURRENT_ENVS=16` with 16 rollouts in flight: finished sessions
   free their slot asynchronously, so new connections were refused → 19/32 rollouts per update failed.
   The real error was masked by an OpenEnv bug: `MCPToolClient._connect_async` does `await self.close()`
   on failure, but `close()` is sync in sync mode → `TypeError: object NoneType can't be used in 'await'
   expression`. Fix: 64 slots (sandboxes are still bounded by the trainer's 16 workers) + retry only
   when no rollout ran (`result is None`), so groups are not biased.
6. **Agent timeout silently 1,500 s.** `HarborSession.wait_for_completion(timeout_s=X)` sends X to the
   server *as the agent timeout*. Passing `agent_timeout + 900` let a stuck bash loop hold an update for
   >11 min while vLLM idled. Over 190 rollouts p99 = 284 s and none >180 s was correct → agent timeout
   300 s in training and evaluation (base re-evaluated under the same setting).

### Docker networks (found at 32 concurrent sandboxes)

7. Each sandbox gets its own compose network; Docker's default pools allow ~28 → 4/32 rollouts per
   update failed (`all predefined address pools have been fully subnetted`, seen as FATAL `no_turns`).
   Fix: `/var/snap/docker/current/config/daemon.json` `default-address-pools: 10.200.0.0/16 size 24`
   (256 networks; backup `daemon.json.bak-20261003`), `sudo snap restart docker`.
8. After that, sandboxes on 10.200.x bridges cannot reach 172.17.0.1 (Docker bridge isolation), so the
   capture proxy is now advertised at the host's primary IP (`ip route get 1.1.1.1` src).

## Resumable training

- `pirl/grpo.py` writes `<run>/state_latest/` (adapter + optimizer + step + task cursor) atomically
  after every update, plus `checkpoint-N/` every `--save-steps`. On start it resumes from
  `state_latest`, trims log lines past the saved update, and replaces any same-named adapter in vLLM.
- If >50% of an update's rollouts fail for infrastructure reasons, the same batch is redone (logged in
  `infra_events.jsonl`); 5 consecutive outages → exit so the supervisor can restart services.
- `pirl/run_resumable.sh <run> <args>` supervises: health-checks/restarts Harbor and vLLM, waits for a
  free GPU, relaunches the trainer until `--steps` is reached. Verified: SIGKILL after update 1 →
  "resumed at step 1, task cursor 1" → update 2 on the next task → finished.
- The 07:12 run is `runs/main200` (supervisor pid in `runs/main200/supervisor.pid`). Earlier attempts:
  `runs/aborted_8x4_*` (capacity, timeout, c16, killed at update 6 by an unexplained external kill).

## Tinker (Qwen/Qwen3.5-4B) with the Pi harness

- **OpenAI-compatible endpoint** (`.../oai/api/v1`): base model id accepted; tool calls parsed;
  thinking off via `reasoning_effort: false` (this model accepts only true/false); **logprobs yes,
  token ids no** → eval only. Prompt cache across calls works (256-token blocks).
- `pirl/tinker_shim.py` (port 8300): OpenAI shim that injects `reasoning_effort: false` and keeps the
  key server-side. Harbor must request *eval* rollouts for it (`pirl/evaluate.py --eval-tier`);
  a training sampling policy is refused ("training sampling requires an endpoint with token capture").
- Pi on Tinker, 3 held-out tasks: **3/3 correct**, 3–6 tool calls, ~35 s/rollout; 73% of prompt tokens
  billed as cached (97k/134k). Cost ≈ $0.02.
- **Native SamplingClient**: returns sampled token ids + per-token logprobs; `prompt_cache_hit_tokens`
  = 256 on a 340-token exact prefix extension. Qwen3.5 emits XML tool calls
  (`<tool_call><function=bash><parameter=command>…`). Training on Tinker needs a SamplingClient-backed
  shim (render with the HF template, parse XML tool calls, return vLLM-style `prompt_token_ids`,
  `token_ids`, logprobs) and a prefix-extension check of the Qwen3.5 template, as was needed for LFM.

## Base-model evaluation, held-out medium+hard (2026-10-03)

`pirl/evaluate.py --split test --difficulty medium,hard --tasks 217 --concurrency 32`, native template,
vLLM 0.85 GPU (846k-token KV), prefix caching, agent timeout 300 s, T=1.0, 4,096 output cap, 32k context.
Output: `runs/eval_base_heldout_mh/rollouts.jsonl` (one line per rollout; `prompt_id` = sha1(instruction)[:16],
`rollout_id` = `<prompt_id>-r<k>`; full messages + per-call sampled token ids and logprobs),
summary `results/baseline_pi_heldout_medium_hard.json`.

| | n | pass@1 | submit | ctx exhausted |
|---|---|---|---|---|
| all | 217 | 0.258 | 0.369 | 0.378 |
| medium | 118 | 0.322 | 0.449 | 0.305 |
| hard | 99 | 0.182 | 0.273 | 0.465 |

4.7 tool calls, 4.9 model calls, 1,009 generated tokens per rollout; 0 errors, 100% ATIF match; 22 min wall.
Native template → 4.9 training rows per rollout (no prefix chaining), as expected.
