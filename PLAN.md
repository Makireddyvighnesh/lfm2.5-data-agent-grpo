# Data Agent RL plan and log

Goal: LFM2.5-2.6B learns, by itself, to solve SmolDataEnvs tasks in the Pi harness with context
under about 15k tokens, and beats the multi-harness RL blog. There are no prompt hints and no teacher
data. Plan agreed between Claude and Codex (gpt-6.1-sol, xhigh); the full Codex plan is in this session's log.

## Current protocol (phase 3; earlier experiments are logged below)
- **Serving:** vLLM, **FP8** (W8A8, online), native template, `max_model_len 16384`, 4,096 output cap.
  This gives a 12,288-token prompt window, so peak context is about 13k.
- **Limits:** T=1.0, top_p=1, 17 model calls; 240 s training and 300 s evaluation agent timeout. Thinking OFF. The benchmark prompts are unchanged.
- **Trainer:** frozen FP8 base using vLLM's own kernels (`--fp8-base --fp8-mode vllm`), LoRA r=32/α=64 with fp32 master weights and BF16 compute. Sampler/trainer mismatch is corrected with truncated importance sampling (`--is-correction tis`, cap 2).
- **Reward (`--reward tool`):** wrong/no answer = 0; correct = 1 + 0.1·15/(15+tool_calls), the blog recipe. Advantage std floor 0.02.
- **Training pool:** `manifests/train1000.json`: all 1,000 medium+hard train tasks (400 medium, 600 hard), seeded sequential order. Pure GRPO, no teacher data or prompt changes. Start from arm A's final adapter, preserved at `checkpoints/p3-init-A-tool`.
- **Validation:** `manifests/val118.json`, the medium+hard eval tasks that share no notebook with train or test.
- **Final comparison:** the 250 test tasks. Comparators: the base model and the blog's released checkpoints, both in Pi at 16k.

## Earlier plan (historical)
1. [running] FP8 16k baselines: val118, the 8×100 train collection (base pass@8 per task), and test250.
2. Self-RFT (`pirl/self_rft.py`): only the model's own correct rollouts with peak ≤12k, at most 2 per task, behind a diversity gate. Evaluate on val118.
3. GRPO from the RFT adapter: 8 tasks × 8 rollouts, lr 1e-5, regret sampler.
   - Screen for 25 updates with a val118 evaluation, then continue in steps of 25.
   - Stop rules: >75% zero-advantage groups for 20 updates; no validation gain across two evaluations; capture errors or non-finite values.
4. Pure-GRPO control from base (same settings) for attribution.
5. If held-out gains appear, scale the pool to 300, then 1,000. Lr 5e-6 for the 1,000 stage.

## Code changes (all opt-in flags; tests pass)
- `SmolDataEnvs/pirl/grpo.py`: `--reward context`, `--std-floor`, `--is-correction tis`, `--init-adapter`, `--task-file`. Peak context is now taken over all turns, and correctness contrast is logged separately from bonus-only contrast.
- `SmolDataEnvs/pirl/reward.py`: the `context` reward.
- `SmolDataEnvs/pirl/evaluate.py`: `--task-file` and peak-context statistics.
- `SmolDataEnvs/pirl/serve.py`: `--model`, `--quantization`.
- `SmolDataEnvs/pirl/self_rft.py`: new.
- Backups are in `SmolDataEnvs/runs/*.bak-20261010`.

## Results log
| When | Run | Result |
|---|---|---|
| 10-10 | Pi FP8 test250, 32k / 64k context | 0.252 / 0.224. At 64k, 62% of rollouts time out, so more context did not help. |
| 10-10 | **Base, FP8 16k, val118** | **0.102** (12/118). 86% ran out of context; 2.2 tool calls per rollout. A single whole-file `read` fills the 12k prompt window. |
| 10-10 | Base, FP8 16k, train100 collection (stopped at 300/800) | 2% correct. The first call is `read` in 94% of rollouts and `read(limit)` in 5%. A 182 MB-file task was solved twice via `read(limit)→bash→bash→write`. |
| 10-10 | Decision | Self-RFT skipped for now: the corpus would be about 25 rollouts from a few tasks, which fails the diversity gate. Started with pure GRPO (`grpo_r1`). Self-distillation can be revisited later from GRPO-discovered successes. The test250 base baseline is deferred to the final comparison. |
| 10-10 | `grpo_r1` (pure GRPO, regret sampler), stopped at update 12 | **No learning.** 0–12% correct and 75–100% zero-advantage groups. Whole-file `read` stayed at 90–100% as first call. All 19 correct rollouts came from 2 tasks. `read(limit)` appeared only on 2 famous-dataset tasks (sign_mnist, winemag), which the model recognises by filename. The control is kept in `runs/grpo_r1`. |
| 10-10 | First-turn probe (`analysis/first_turn_probe.py`; 100 tasks × 32 candidates) | At T=1.0: full read 94.7%, bash 3.1%, read(limit) 2.2%; 28/100 tasks have ≥1 alternative. T=1.3: 7.1% alternatives (61/100 tasks). T=1.5: 5.7% degenerate no-tool turns, so rejected. |
| 10-10 | **Fix: stratified root sampling** (`root_proxy.py`; Codex's rank-1 option) | Per (policy, task): a pool of 128 first turns at T=1.0. Each group of 8 gets up to 2 "alternative" (bash/read(limit)) candidates. Stratified importance weights `W = p·G/n` go to the trainer (`--root-weights`), with a weighted baseline and the first-call advantage ×8 (`--root-boost`). No prompt or tool changes; capture stays exact. Smoke update: zero-advantage share 0.0 (vs about 0.8), 25% correct, and a whole-file-read-fails task was solved via `read(limit)`. |
| 10-10 | `grpo_r2` launched | r1 protocol + root sampling + sequential (uniform) task order, from the base model, 25 updates. Track: `root_alt_frac`, `root_alt_correct`, zero-advantage share, out-of-context rate, and natural T=1 first actions (probe on the adapter). |
| 10-10 | `grpo_r2` updates 1–4 | Correct 7.8/3.1/4.7/6.2%. Alternative-start correct per update 0→2→3→4. In-training sampler/trainer KL grew 0.0046→0.027, while BF16-sampler runs stayed flat at about 0.0007. |
| 10-10 | Codex rollout audit (576 rollouts, updates 1–9) | **Real learning.** Natural (pre-selection) pool alternative rate on update-5–9 tasks: 1.25% (base probe) → 23%. Alternative-start correct 23%→43% and out-of-context 28%→5%. No reward hacking; successes do compact real analysis (median peak about 4.2k). Remaining failures: number written for "Which X" (19/42 submitted alternative failures), reverting to a whole-file read, huge string outputs, 10 unparsed tool calls, poor error recovery. **Bug:** root-weight lookup keyed by completion hash only; 10/576 wrong weights. |
| 10-10 | Fixes, then `grpo_r2` resumed at update 12 | Weight records keyed by (policy, prompt sha1, completion sha1), all new records read every call, per-group weight mass logged (`root_weight_mass_max_dev`). Adaptive alternative slots `k_alt = max(2, round(8·p_alt))`, capped at 7. Checkpoints every 5 updates. New log `runs/root_assign_r2b.jsonl`. |
| 10-10 | **HELD-OUT: r2 update 12 on val118** (FP8, 16k, no proxy) | **0.314** (37/118) vs base **0.102**. Medium 6→23/68, hard 6→14/50. Paired: 30 tasks solved only by r2-u12, 5 only by base, **McNemar p = 2.2e-5**. Out-of-context 86%→19%; submitted 13%→68%; mean peak context 5.1k. Natural first call on val118 (32 samples/task, T=1): whole-file read **91.7% → 1.6%**, bash 6.6% → 92.5%. Adapter saved at `checkpoints/r2-u12`. |
| 10-10 | **FP8 on both sides** (user request) | `pirl/fp8_emulation.py`, `--fp8-base --fp8-mode vllm`: the trainer's frozen base linears run vLLM's own `scaled_fp8_quant` (per-token) + `cutlass_scaled_mm` with vLLM's fused scale groups (qkv, w13). LoRA stays BF16 on the unquantized input; the BF16 base weights are freed. Measured with the v4 adapter: KL vs FP8 sampler **0.0003** (batch-1 sampling) vs 0.0045 with a BF16 trainer. A quantize-dequantize emulation did *not* help (0.0062). Residual: vLLM FP8 is batch-variant (KL 0.002–0.003 under 32 concurrent or n=64 sampling), handled by TIS (cap 2). Update-4 KL spike = a few rare tokens. `grpo_r2` resumed from update 4 with the FP8 trainer. |

## Phase 2 plan: 300 tasks, target 5–7 min per update (user requirement)
Measured on r2 updates 12–19: generation 5–7 min (64 rollouts, 32 concurrent; median rollout 119 s) and
training 2–12 min. Training is 0.3–1.8M tokens at a constant 2.5k tok/s; under the native template each of
about 9 model calls trains as its own sequence, repeating the context.

| Change | Expected |
|---|---|
| **Prefix-chain template** (`CHAIN=1`, `restart_vllm.sh`): one sequence per rollout | training about 5× fewer tokens: 12 → about 2–2.5 min |
| **64 concurrent rollouts** (`CONC=64`, vLLM `SEQS=64`; KV 633k tokens vs about 64×5k needed) | generation about 4.5 min |
| **Agent timeout 240 s** in training (`TIMEOUT=240`) | loses 1/164 correct (0.6%) on r2 data |
| Root boost 1 (under chain the boost would hit the whole trajectory; pools are already about 99% efficient openings) | n/a |

Gate before committing to the chain template: evaluate r2-u25 on val118 under native (the protocol) **and** under
chain (`CHAIN=1 ./eval_checkpoint.sh ... r2-u25-chain`). Adopt chain only if it does not lose materially.
Evaluation stays at thinking OFF (user decision) and a 300 s timeout.

### Generation-phase profile (r2 updates 15+, 384 trials; mean 124 s per rollout)
| Phase | Mean | Note |
|---|---|---|
| agent execution | 76.7 s | model + tools |
| environment setup | 19.2 s | compose build/up + HF-bucket data pull (4–5 s when idle); slower under 32-way load |
| **teardown** | **19.2 s** | `sleep infinity` as PID 1 ignores SIGTERM, so every `compose down` waits the 10 s grace (kill→die median 10.2 s) |
| other gaps + verifier | 9 s | |
Fix (next Harbor restart): `serve_harbor.py install_fast_teardown()` chowns logs, then `compose kill`, then `down`. Teardown should drop to about 3–5 s.
With 64 concurrent rollouts the update's generation time = the slowest rollout, so this shortens every update.

## Phase 2 (started 2026-10-10 17:33, unattended in tmux session `dataagent-p2`)
- r2 stopped at **update 22** → `checkpoints/r2-final`.
- **Reward ablation** on `manifests/hard50.json` (50 unseen hard train tasks): 20 updates × 8 tasks × 8 rollouts per arm, all from
  r2-final with the same task order (sequential).
  - A = `tool`: the blog's reward, correctness × (1 + 0.1·15/(15+calls)).
  - B = `context`: r2's reward.
  - C = `correctness` only.
  - Common settings: std floor 0.02, root boost 1, no root proxy.
- **Speed setup:** prefix-chain template, 64 concurrent rollouts (vLLM max-num-seqs 64), 240 s training timeout, fast sandbox
  teardown (`serve_harbor.py install_fast_teardown`). Evaluation stays at 300 s, T=1, FP8, 16k, thinking off.
- Pipeline `phase2_pipeline.sh` (resumable): r2-final on val118 under chain (the start point) → arms A, B, C each followed by
  a val118 evaluation → r2-final on val118 under native (template check) → `results/phase2_summary.md`.
- Log: `runs/phase2_pipeline.log`. Watch with `tmux attach -t dataagent-p2` (window `progress`).

## Phase 3: all 1,000 tasks (2026-10-11, launched 06:59 UTC)

- Run: `runs/p3_train1000_tool`, supervisor `run_train1000.sh` in tmux `dataagent-p2:train1000`.
  Monitor `dataagent-p2:progress1000` or `python3 analysis/monitor_train1000.py`.
- 250 updates = two complete sequential passes over the seeded 1,000-task manifest. Sequential sampling
  guarantees coverage in 125 updates; regret would spend part of that budget revisiting already seen tasks.
  Start from arm A's final adapter, copied and hash-verified at `checkpoints/p3-init-A-tool`; fresh optimizer, lr 5e-6.
- 8 tasks × 8 rollouts, concurrency 64, chain template, thinking OFF, 16,384 serving limit, 4,096 output cap,
  240 s training timeout. Blog `tool` reward, std floor 0.02, FP8 sampler + vLLM-kernel FP8 trainer, TIS cap 2.
  Packed training micro-batches are 32,768 tokens; each individual rollout remains within the 16k context limit.
- Atomic `state_latest` after every update; named checkpoints every 10 updates. Val118 at updates
  50/100/125/175/225/250, four rollouts per task (472 trials), 300 s timeout, no root proxy.
- Prefetch complete: all 329 datasets (16.79 GB downloaded), including all 1,000 training tasks, have `.complete` markers.
- Runtime Harbor patches: correct the prebuilt-image Compose variable; skip per-rollout builds; mount cached inputs
  read-only; cache fixed Docker host metadata; combine log ownership commands; keep fast SIGKILL teardown; reuse
  dedicated external bridge `dataagent-p3-fast`. Containers retain separate filesystems and CPU/memory limits;
  containers in this run share the bridge rather than separate per-rollout networks. No host networking.

### Measured speed gates

| Experiment | Result | Decision |
|---|---|---|
| Arm A, historical full 20 updates | mean gen 323.9 s, train 104.9 s, total 431.3 s | comparison baseline |
| Cached/prebuilt full 64-rollout smoke, BF16 KV, 16k packing | gen 347.1 s, train 78.6 s, total 435.1 s; KL 0.00642; no capture/infra errors | alone does not meet speed target |
| Same 124,731 training tokens, warm replay | 16k: 63.4 s / 8 rows; 32k: 49.5 s / 4 rows; KL 0.00642 → 0.00649; peak 13.4 GiB | adopt 32k packing (22% faster) |
| 64 real sandbox setup/teardown, metadata patch | 76.4 → 62.4 s | adopt |
| Direct Docker exec; bounded lifecycle concurrency=8; GOMAXPROCS=2 alone | no clear standalone improvement; concurrency=8 slowed teardown | direct exec and lifecycle limit disabled |
| Dedicated shared bridge, identical GOMAXPROCS=2 benchmark | 70.3 → 24.4 s; mean setup 37.1 → 12.0 s; mean teardown 21.7 → 9.6 s | adopt shared bridge; GOMAXPROCS=2 retained to match benchmark |
| Warm decode, 64 requests, 4,681 prompt tokens, 256 output | BF16 KV 1,708 tok/s; FP8 KV 1,976 tok/s; fixed worker 1,989 tok/s | adopt FP8 KV (16% faster) |
| FP8 KV mismatch replay, initial A adapter, 24 sequences / 35,760 completion tokens | KL 0.007754; mean ratio 1.0005; TIS cap fraction 0.00221 | mismatch stays small |
| FP8 KV sleep/wake | upstream failed on hybrid cache list `zero_()`; local worker flattens cache tensor view during reset and restores structure; two GPU sleep/wake/generation cycles passed (4.41 / 1.24 s) | runtime fix, upstream vLLM source untouched |

Backups: `runs/backups/codex-20261011`. OpenEnv/Harbor/TRL source unmodified; 54 regression tests passed,
and 493 installed Harbor/TRL source hashes match the pre-edit snapshot. Full-run first-update timing confirmation follows below.

### First three full-run updates confirmed (07:19 UTC)

| Update | Generation (s) | Training (s) | Total update (s) | KL | Capture/infra errors |
|---|---|---|---|---|---|
| 1 | 279.1 | 80.1 | 361.3 | 0.00887 | 0/0 |
| 2 | 283.4 | 94.1 | 379.7 | 0.00748 | 0/0 |
| 3 | 236.4 | 88.1 | 326.6 | 0.00745 | 0/0 |

State latest is update 3 / cursor 24; optimizer, adapter and RNG files verified. All 192 rollouts have
zero capture and infrastructure errors. Adapter `trl-policy-v3` is served; the supervisor continues in tmux.
Mean total update is **5.93 min**, **17.5% below** arm A's historical 7.19 min.
**The 4–5 min target has not been reached.** The remaining bottlenecks are long agent/model/tool trajectories
(including 240 s agent timeouts) and 80–94 s training. Timeout was kept at 240 s to preserve successful rollouts.
Further training/decoding optimization remains open; no unvalidated change to the learning objective was adopted.
Estimated remaining time: **26–28 h including six periodic val118 evaluations** (roughly 2026-10-12 09:00–11:00 UTC).
Held-out improvement is still unmeasured; first new validation is at update 50.

Stop safely: `touch runs/p3_train1000_tool/STOP`. Resume by removing that marker and rerunning
`KV=fp8 BUDGET=32768 ./run_train1000.sh` inside tmux. Latest state resumes without replaying saved updates.

| 10-11 | **Outage 12:55–15:25 UTC (update 56)** | The capture proxy's one-time check of trl-policy-v55 came back `logprobs` (not `tokens`) and was cached forever, so every session was rejected ("training sampling requires an endpoint with token capture"); all 4 supervisor attempts failed identically. Fix: `serve_harbor.py install_reprobe_degraded()` retries the check 3× and never caches a degraded level. Harbor restarted; the v55 check now gives `tokens`, and update 56 rollouts are graded. Val118 at u50: 0.390 (start 0.381). |
| 10-12 | **1000-task run finished** (13:59 UTC, 250 updates, mean 6.29 min/update) | val118 (4 attempts/task): start 0.373 (1/task) → u50 0.390, u100 0.424, u125 0.451, u175 0.468, **u225 0.483**, u250 0.479. u50→u225 +9.3 pts (paired z 3.9); u175→u250 +1.1 (z 0.5): plateau. Tool calls 7.5 → 5.9; peak context ~5.3k; out-of-context ~14–16% unchanged. Best checkpoint: checkpoint-225 (u250 equivalent). Next: test250 comparison vs base and blog checkpoints in Pi at 16k. |
| 10-12 | **Rollout failure review** (10 Haiku agents; `analysis/failure_review/SUMMARY.md`) | Eval u225 failures: ~13-15% context overflow (real prompt limit 12,288 = 16,384 − 4,096 reserved; Pi truncates tool output only at 50 KB), ~35-40% of wrong answers are grader/task issues (±0.05 tolerance, superset lists, unstated seed/split, unanswerable tasks where N/A is never gold), 3-5% unparsed pythonic tool-call text, 2-5% timeouts. Training fixed read→bash and number-for-"which". No reward hacking/memorisation. Tool bonus amplified by std floor 0.02 in 8/8 groups. Next levers: dynamic max_tokens, tool-output cap, grader patch, task filter, bounce unparsed calls. |
| 10-12 | **Blog verification audit** | Same pinned grader (16f906e9) and identical train1000/test250 task lists on both sides, so grader flaws hit both equally. Blog eval: T 0.8, 131k ctx, 600 s, step limit 17, BF16, 4 harnesses, 1 sample/pair, ungraded pairs excluded. Ours (val118): T 1.0, 16k (effective 12.3k), 300 s, FP8, Pi, 4 samples. 134/250 test tasks (51/122 datasets) reuse a dataset seen in training (different questions) in both setups. Blog's own 25-task pilot shows same-model baselines 24% vs 17%. Final comparison must be test250 under matched settings. |
| 10-12 | Findings log started | `FINDINGS.md`: benchmark flaws (shared with blog), reference-eval caveats, our deviations, supported claims, open checks. Append new findings there. |
| 10-13 | Fixed grader v3 | `grader_v3.py`; blind-judge validated (97.7% on disagreements; strict 99.3% on agreements). u225 pass@1 pinned 48.3 / v3 48.1 / strict 44.3; gain u50→u225 holds. `analysis/failure_review/GRADER_V3.md`. |
| 10-13 | **p4_train500 started** (tmux `dataagent-p4`; `run_train500.sh`; monitor `analysis/monitor_train500.py`) | Fresh LoRA on base; 500 tasks (`manifests/train500.json`: 1000 minus 210 never-solved, stratified, seed 20261013); 8 tasks × 4 rollouts = 32/update, one optimizer step over 32k-token micro-batches; 125 updates (2 epochs); reward = blog tool reward graded by grader_v3 (`prepared_v3/datasets/train`); std floor 0.1; FP8 + TIS; Harbor proxy: tool-output cap 4,000 chars + output-reservation shrink (`serve_harbor.py install_context_fixes`); trainer: `--data-root`, GRPO stats (reward/advantage std/min/max, group histogram, lr, think-leak, ended-on-call-text). **Base val118 with these harness fixes: 30.9% (was 10.2%)**. First 10 updates: 4.3 min/update, correct 31-69%, KL ~0.007. Harbor restarted 15:20 UTC to fix the shrink (it changed only max_tokens; vLLM uses max_completion_tokens): two infra retries, then shrink works (10 shrunk / 1 exhausted). Note: `ended_on_call_text_rate` also counts rollouts cut by timeout or budget while calling, not only unparsed call text. |

### 2026-10-13 20:08 UTC — p4_train500 stopped by the user at update 59/125
- Stopped with runs/p4_train500/STOP after update 59; state_latest = step 59, checkpoint-25 and checkpoint-50 kept.
- val118 (pinned, 4/task): u0 30.9, u25 38.6, u50 38.3 (v3.1: 29.9 / 37.1 / 37.5); u25 -> u50 flat (paired z -0.1).
- Reason for redesign: the first call is Pi `read` on the whole CSV in 87% of rollouts (Pi system prompt: "Use read to
  examine files"), then pandas re-loads the same file; NL planning is near zero in turn 1 (thinking-off prefill
  `<think></think>`, Pi "Be concise"). Rollouts with NL before >= 2 tool calls: 44.7% correct vs 30.3% without.
- vLLM (:8000) and Harbor (:8201/:8101) left running for the next run.

### 2026-10-14 06:13-06:45 UTC — thinking-ON check (base model, val118, 1 attempt/task)
- `PIRL_THINKING=1` (pirl/common.py, default off; backup runs/backups/claude-20261013/common.py.bak): native `<think>`
  generation prefix, past turns re-rendered as "<think>"+content so attempts still chain (1.12 roots/attempt).
- vLLM currently serves the thinking template (runs/vllm.mode = chain1_seqs64_kvfp8_THINKING).
- val118 pinned pass@1: thinking-on base 39.8% vs thinking-off base 30.9% (paired +8.9 pts, z 2.6, 33 tasks better /
  18 worse) and vs thinking-off u50 38.3% (+1.5, z 0.4). Plan before first action 100% (was 0%); first call still
  `read` 92% (Pi system prompt rule); output tokens/turn median 245, p90 989; 3 turns hit 4,096; peak context 8.2k;
  ran out of context 1.7% (was 9.5%). 9 tasks first failed on Harbor websocket handshake timeouts at concurrency 64;
  re-run at concurrency 9. Script: analysis/compare_thinking.py. Files: runs/val118_base_thinking_r1, results/.

### 2026-10-14 06:58 UTC — p5_train500h8 started (tmux `dataagent-p5`; `run_train500h.sh`; monitor `analysis/monitor_train500h.py`)
- User spec: 8 attempts/task, 500 tasks (300 hard + 200 medium), 2 epochs; thinking OFF; 4k cap + context bonus; plain GRPO.
- Tasks `manifests/train500h.json` (pool = 1000 minus 210 never solved; seed 20261014; 311 overlap with train500).
- Split `prepared_v31/datasets/train` (grader v3.1 sha 207f003d, no pip line). `prepared_v3` untouched (p4 reproducible).
- Reward `tool_context` (pirl/reward.py, test added; backups in runs/backups/claude-20261013/):
  correct x (1 + 0.1*15/(15+calls) + 0.1*clip((12k - peak_context)/8k, 0, 1)); verified on update 1 (0 mismatches).
- 8 tasks x 8 rollouts = 64/update, CONC 64, 125 updates, lr 1e-5, std floor 0.1, FP8 + TIS 2, save every 25.
- val118 at 25/50/75/100/125; u0 reused from p4_train500-u0 (30.9%, identical base, harness, template, grader).
- Monitor adds per-update behaviour: first call `read` rate and number-for-label rate (update 1: 100% / 67%).
- Update 1: correct 45.3%, 7.8 min/update (incl. compile); ETA ~23:20 UTC + ~1.5 h evals.
- 2026-10-14 09:11 UTC: paused by the user (GPU needed elsewhere). STOP after update 17 (state_latest = 17; update 18's
  64 rollouts archived unapplied). vLLM stopped; GPU free. Harbor (:8201, CPU) left running.
  Resume: `rm runs/p5_train500h8/STOP`, then in tmux `dataagent-p5` window `train`: `./run_train500h.sh`
  (it restarts vLLM in thinking-off mode and continues from update 17).

- 2026-10-15 07:15 UTC: p5_train500h8 STOPPED by user request after update 110/125 (update 111 in progress, discarded). Supervisor, grpo.py, vLLM killed; GPU 0 MiB. STOP file left in runs/p5_train500h8/. val118 pinned: u0 30.9, u25 39.4, u50 43.6, u75 47.7, u100 47.5. Resume: rm STOP, rerun ./run_train500h.sh in tmux dataagent-p5 (resumes from last saved step).

- 2026-10-15 07:32 UTC: LiquidAI/LFM2.5-1.2B-Instruct (rev 0f604ada) on val118, 4 rollouts/task, Pi, FP8, 16k, native template (no <think>), T 1.0 (same as all val118 evals), pinned grader: pass@1 0.0% (0/470 graded; 2 Harbor handshake failures), pass@4 0/118, submit 0.2%. Behaviour: 1 tool call (mostly full `read`) then a chat-text answer without writing answer.txt; the one submission wrote 95293 (the prompt's example). Launcher serve_lfm12b.sh; results/val118_lfm12b-instruct_r4.json.

## p6_train1000f (started 2026-10-15 08:38 UTC, tmux dataagent-p6; supervisor ./run_train1000f.sh)
- User spec: train from scratch, LoRA r16 (alpha 32), thinking OFF, 1000 tasks (manifests/train1000.json), 4 tasks x 8
  rollouts per update, 2 epochs = 500 updates, grad accumulation via token-budget micro-batches; no time constraint but
  penalise time/tokens; tool-efficiency bonus; eval val118 every 50 updates, 1 rollout/task, eval concurrency 16.
- Reward `tool_fast` (pirl/reward.py; test in tests/test_context_reward.py; 56 tests pass; backups
  runs/backups/claude-20261015/): correct x (1 + 0.1*15/(15+calls) - 0.1*clip((agent_s-20)/180) - 0.1*clip((gen_tok-300)/2700)),
  range [0.8, 1.094]; failures 0. agent_s = Harbor agent_execution span (grpo.py `_agent_seconds`). Replayed on p5
  correct rollouts: 0.85-1.09, median 1.04.
- No practical time limit: agent timeout 3600 s (hang guard; a task's default is 600 s; 0 falls back to it); turn cap 17.
- Train split prepared_v32: all 1000 train tasks with grader v3.1 (sha 207f003d), pip line removed (500 overlap with
  prepared_v31 identical). Proxy tool-output cap 4000 (as p4/p5). Eval: pinned grader (prepared/), u0 run fresh (r1).
- Stopped the user's LFM2.5-1.2B vLLM on :8000 at 08:37 with their approval. The supervisor never kills a foreign vLLM.
- Monitor: python3 analysis/monitor_train1000f.py. Stop: touch runs/p6_train1000f/STOP.
- 2026-10-15 09:19 UTC: p6 RESTARTED from scratch (first attempt archived as runs/aborted_p6_train1000f_20261015T0919*,
  4 updates). Reason: with no time limit, 5 heavy-compute rollouts (forest-cover precision, grid search) held update 4
  for 15+ min with the GPU idle. User: cap 10 min; on correct rollouts a small time bonus under 5 min, penalty over.
  New tool_fast: correct x (1 + 0.1*15/(15+calls) + 0.05*clip((300-agent_s)/300,-1,1) - 0.1*clip((gen_tok-300)/2700,0,1)),
  range [0.85, 1.144]; failures 0. Agent timeout 600 s for training and eval. p5 replay: 0.96-1.14, median 1.095.
- Update-2 review (Haiku 5.5, analysis/p6_u2/report_*.md): 3/4 tasks have unreachable golds (unstated split/seed,
  GD 1500 steps at lr 0.01, one of 30 preprocessing setups); 0048 needs A-Z prior knowledge. Real failure: tool calls
  emitted as plain text (25% of rollouts in updates 2-3).

## p7_think_bash (started 2026-10-17 14:27 UTC, tmux dataagent-p7; ./run_train_p7.sh; monitor analysis/monitor_p7.py)
- Goal: test250 70%+. p6 u500: val118 51.7% (r1), bash-only 49.2%, test250 56.0% (Pi normal tools; first-50-only run 62% discarded).
- p6 recipe + thinking ON (PIRL_THINKING=1; vllm.mode chain1_seqs64_kvfp8_THINKING) + pi bash-only (HARBOR_PI_TOOLS=bash
  -> `--tools bash`, runtime patch in serve_harbor.py) + turn cap 25 (train and eval) + pool manifests/train798.json
  (train1000 minus 202 tasks never solved in 16 p6 attempts; 102 of them had a consensus wrong answer). 400 updates = 2 epochs.
- Verified: template opens "<think>", model writes a plan then </think>; only bash tool calls.
- Stopped p6's own vLLM (pid 3891052) so the supervisor could restart it in thinking mode.
