# Data Agent RL: experiment log (what we changed, what we got, what worked)

Model: LiquidAI/LFM2.5-2.6B (rev 654f9463), FP8 sampling and training (TIS cap 2), GRPO, Pi harness, 16k context.
Eval: val118 (pinned grader) unless noted; test250 = blog test split. r1 = 1 attempt/task (+-4.5 pts on val118,
+-3 pts on test250), r4 = 4 attempts/task. Blog (Adithya, multi-harness RL) test250 pass@1: mini-swe-agent 64.8,
Codex 53.6, OpenCode 49.6, Claude Code 48.8, average 54.2.

## Runs

| Run | Changes vs previous | Best val118 | test250 | Verdict |
|---|---|---|---|---|
| base | thinking off, Pi all tools, 300 s limit | 30.9 (r4) / 37.3 (r1, 600 s, conc 16) | - | reference |
| base, thinking ON | native `<think>` (PIRL_THINKING=1) | 39.8 (r1, vs 30.9) | - | **+8.9, worked** |
| p3_train1000_tool | 1000 tasks, 250 upd x 64 rollouts, tool-call bonus, no tool-output cap | 48.3 (r4, u225) | - | learned to avoid full-file `read` (context pressure) |
| p4_train500 | 500 tasks, 8x4, grader v3.0, 4k tool-output cap | 38.6 (r4, u25) | - | cap removed pressure against `read`; stopped at u59 |
| p5_train500h8 | 300 hard + 200 medium, 8x8, grader v3.1, reward + peak-context bonus | 47.7 (r4, u75) | - | fast early gain, plateau u75; stopped u110 |
| p6_train1000f | LoRA r16, 4x8, 1000 tasks x2 ep, reward `tool_fast` (time bonus <5 min / penalty >5 min, token penalty), 10-min cap, eval r1 conc 16 | 52.5 (r1, u400); 51.7 u500 | **56.0** (u500) | number-for-label 51% -> 4%, no-answer 23% -> 10%; plateau after u200; still `read` 90% |
| p6 u500 bash-only (eval only) | pi `--tools bash` at eval | 49.2 (r1) | - | removing `read` barely hurts an untrained-for-it model |
| sd_p7 | SFT (fresh LoRA r64 on base) on 1,451 own correct p7 rollouts, 1 epoch | - | 58.0 (15 better/24 worse vs p7) | close to p7, not better |
| p7_think_bash | p6 + thinking ON + pi bash-only + turn cap 25 + pool minus 202 never-solved tasks (798) | 53.4 (r1, u250); 52.5 u350 | **61.6** (u350; +5.6 vs p6, 27 better/13 worse, z 2.2) | `read` 0%, no-answer 7.6%; val118 r1 too noisy to show the gain; stopped u352/400 |

## What worked
- Thinking ON: +8.9 at base (val118), part of p7's +5.6 on test250.
- Bash-only Pi (`HARBOR_PI_TOOLS=bash` -> `--tools bash`, runtime patch in serve_harbor.py): full-file reads 90% -> 0%.
- Rewards only on correct answers (failures exactly 0): tool-call bonus, time term (5-min pivot), token penalty ->
  faster, shorter correct answers (p6: 58 s -> 41 s, 1000 -> 657 tokens).
- Grader v3.1 for training (0.7% error vs blind judges; pinned grader 3.8%).
- Dropping never-solved tasks: removes all-zero groups (no gradient), many have bad golds.

## What did not work / limits
- No time limit (p6 first try): heavy-compute rollouts held updates 15+ min with the GPU idle -> 10-min cap.
- 4k tool-output cap without a context cost: model kept full-file `read` (p4-p6).
- More updates of the same recipe: val118 plateaus after ~u200 (p5, p6, p7).
- val118 r1 cannot resolve 2-3 point differences; use r4 or test250 for decisions.

## Failure analysis (p6 u400 val118, Haiku 5.5; analysis/p6_u400_fail/)
Wrong analysis 22%, underspecified gold 10%, no answer 10%, number-for-label 2%, wrong gold 2%, grader 1%.
Context exhausted: 1-5 attempts per eval (not a factor).

## Task solve-rate buckets (19 val118 evals; analysis/val118_task_buckets.json)
always 12, reliable >75% 24, unreliable 25-75% 36, rare <25% 22, never 24 (9 unwinnable, ~10 capability).
94/118 (80%) solved at least once -> the main gap is reliability, not capability.

## sd_p7: self-distillation (started 2026-10-19 09:19 UTC, tmux dataagent-p7:sft)
- Corpus: correct p7 TRAINING rollouts (thinking + bash-only, own sampled tokens only; analysis/p7_to_rft_corpus.py),
  top-3 by reward per task -> self_rft.py keeps 2/task by lowest peak: 1,451 rollouts, 741 tasks (510 large-file).
- SFT: pirl/self_rft.py, fresh LoRA r64/alpha 128 on base, lr 1e-5, 1 epoch, ~182 updates of 8 rollouts.
- SFT finished (182 updates, nll/token 0.29 -> ~0.22, peak 11.7 GiB); adapter runs/sd_p7/adapter.
- vLLM: restart_vllm.sh now takes LORA_RANK (default 32); served with LORA_RANK=64. vLLM needs an ABSOLUTE adapter path.
- test250 run 1 (11:30 UTC): 38.4% but INVALID: 59/250 rollouts never reached the model (sandbox/infra, 30-90 s, 0 calls).
  On the 191 valid tasks: sd_p7 96 (50.3%) vs p7 u350 110 (57.6%) -> SD looks WORSE. Rerun started 12:15 UTC.
- **test250 rerun (valid, 0 infra failures): 58.0%** (submit 87%) vs p7 u350 61.6: sd better on 15 tasks, worse on 24
  (sign test p~0.2, not significant). Verdict: SD from base reaches ~p7 level in 182 SFT steps (vs 350 RL updates)
  but does NOT beat it. Next: SD on top of p7 u350 / RL from the SD adapter.
  Run-1 symptoms: submit 62% (p7 87%), mean generated tokens 3342 (p7 2848).
- Plan: eval val118 r4 + test250 (p7 setup) vs p7 u350 (61.6 test250), then RL from this adapter.

## Next (older plan)
SFT on the model's own correct, efficient rollouts (p6 + p7), then RL with larger groups on frontier tasks,
LoRA r64, p7 setup. Track: solve rate on unreliable/rare buckets, test250.

## Ablations on abl40 (started 2026-10-19 16:56 UTC; run_ablations.sh, tmux dataagent-p7:abl2, log runs/ablations.log)
- Tasks (manifests/abl40.json): 30 hard + 10 medium train tasks p7 got 1/8 correct on its last visit (all solved at least once).
- Every arm: init from p7 u350 (LoRA r16, fresh optimizer), 10 updates x 4 tasks x 8 rollouts = one pass over the 40,
  same order; p7 setup (thinking, bash-only, 25 turns, 600 s cap, FP8 + TIS). Metric: same 40 tasks x 4 rollouts,
  before (p7 u350) vs after. Caveat: train = eval tasks, so this measures learning speed on hard tasks, not generalisation.
- Arms: base (p7 recipe, tool_fast, lr 1e-5) | lr3e5 | binary (reward = correctness only) | cliphigh (epsilon-high 0.4).

| Arm | change | abl40 r4 pass@1 | train correct (u1 -> u10) |
|---|---|---|---|
| p7 u350 (before) | - | 16.3 (26/160), 20 tasks >=1 | - |
| base | p7 recipe | 16.3 (26), 19 tasks; 10 up / 12 down | 31 -> 16% (noisy, per-batch tasks differ) |
| **lr3e5** | lr 3e-5 | **23.8 (38)**, 25 tasks; 13 up / 8 down | 34 -> 22% |
| binary | reward = correctness only | 20.6 (33), 22 tasks; 13 up / 9 down | 34 -> 12% |
| cliphigh | epsilon-high 0.4 | 13.8 (22), 17 tasks; 11 up / 13 down | 34 -> 22% |

Reading: 10 updates of the p7 recipe do nothing on hard tasks (16.3 -> 16.3). A 3x higher lr is the only clear gain
(+12 correct, ~2 SE; 25 vs 20 tasks solved at least once). Dropping the speed/tool/token shaping (binary) helps a
little (+7, within noise). Clip-higher hurts slightly (within noise). Train correct_rate per update is not comparable
across updates (different tasks per batch). Next: confirm lr3e5 (+ binary) on test250 / longer run.

## Speed profile (2026-10-20; abl_lr3e5 + p7 metrics, vLLM benchmark on the idle L4)
- One update (4x8 rollouts): ~480 s on abl40 (p7: ~370 s) = generation 80% + training 20% + sync <1%.
- Generation time = the SLOWEST rollout of the batch (median rollout 155 s vs slowest 368 s on abl40; p7 85 vs 295 s).
  GPU request slots are only ~43% occupied during generation: the tail runs a handful of sequences.
- Inside a rollout: model decoding 90%, tool execution 10%; plus ~33 s sandbox start/verify per rollout.
- vLLM (FP8, CUDA graphs, 13.5k-token prompt, 512-token decode), with a LoRA r16 adapter:
  1 seq 50 tok/s | 32 seqs 1,208 tok/s (38/seq) | 64 seqs 1,791 tok/s (28/seq). Base without LoRA: 64 tok/s at 1 seq.
  -> throughput scales almost linearly with concurrency; the GPU has headroom we do not use.
- KV per token: 8 attention layers x 8 KV heads x 64 x 2 x 1 B (fp8) = 8 KB -> 32 seqs x 16k = 4 GB.

## Speed change 1: streaming rollouts (2026-10-20; grpo.py --stream-groups K)
- groups + K task groups in flight; train on the oldest `groups` that are complete; unfinished rollouts keep their
  adapter (old adapters stay loaded until unused) and pause in vLLM during training (sleep mode="keep", verified:
  in-flight requests resume with full logprobs). Lag recorded as policy_lag_mean (0.5 = half the rollouts 1 update old).
- Test: lr3e5 arm rerun with K=2, CONC=48, token budget 49k (peak 18.5 GiB):
  10 updates in 63 min vs 86 min (1.36x faster; 218-476 s/update vs 290-716 s); abl40 r4 = 23.75% (38/160),
  identical to the synchronous lr3e5 arm -> no quality loss.
- Less than the ~2x estimate: groups still finish at their slowest member, 48 concurrent sequences decode ~25%
  slower each, and paused rollouts lose ~100 s of their 600 s cap during each training phase.
- Also: eval scripts now take EVAL_CONC (default 32, was 16); run_ablations.sh takes CONC / BUDGET.

## test250 failures vs training categories (2026-10-21, analysis/category_overlap.py)
- No task/notebook/question overlap between test250 and train. Categories do overlap: every test answer type and question type also appears in training, and 0 of the 96 failures lack solved similar training questions (5 nearest by TF-IDF).
- The weak spot is the same in train and test: ML/modelling questions (accuracy/F1/R2/AUC/"which model is best") make up 345/797 train tasks (43%) with p7 train solve 52%. On test they are 62 tasks at 40% pass, which is 37 of the 96 failures. Hard numeric: train 49% / test 31%.
- Many of these golds come from the original notebook's unseen pipeline (random split, chosen hyperparameters, "50 test iterations"), so they can't be reproduced exactly. That points to grading/pool noise, not a category the model never saw.
- Count/aggregate/percent/correlation: train 72-82%, test 63-88%.

## Harness prompt overhead (2026-10-21, first-call prompt tokens, LFM tokenizer, thinking ON)
| Harness | Tools | First prompt | Source |
|---|---|---|---|
| Pi bash-only (p7) | 1 | ~870 | captures, 720 rollouts |
| mini-swe-agent | 0 (text bash) | ~945 | tokenized trajectory; run failed at install (litellm imports missing `mcp`) |
| Pi default | 4 | ~1,600 | captures |
| OpenCode | 8 | ~6,200 | captures |
| Codex | 7 | ~6,330 | smoke runs/plen_codex, 2/2 correct, atif match |
| Prime Agent | 1 | ~8,400 | captures |
| Claude Code | 12 | ~10,770 | smoke runs/plen_claude-code, 1/2 correct, atif match, peak 13.6k / 15.4k after 4-5 turns |
Script: runs/harness_promptlen.sh. Claude Code tools include Agent/Skill/Workflow/ScheduleWakeup etc. that are unused here.

## p8 multi-harness setup (2026-10-21): pi + mini-swe-agent + codex, 24k window
Changes (backups in runs/backups/claude-20261015/*.bak-20261021):
- Sandbox image pirl/sandbox:pi-mh (sandbox_image/Dockerfile.mh), retagged savatar101/env-data-agent-train:base.
  It adds Codex 0.163.0 and mini-swe-agent 2.4.6, installed with `--with mcp` because current litellm imports
  mcp without declaring it. Rollback: `docker tag pirl/sandbox:pi-prime-opencode savatar101/env-data-agent-train:base`.
- serve_harbor.py:
  - install_mini_swe_prebaked_skip: Harbor would otherwise reinstall mini-swe-agent unpinned on every rollout.
  - install_merge_split_assistant: Codex (Responses dialect) sent one reply back as two assistant messages, so every
    turn became a new training root (6 roots / 6 turns). After the merge: 1 root, ~750 trainable tokens.
- Smoke (2 test tasks each, base): mini-swe-agent 2/2, first prompt ~1,020, 1 bash tool; codex 2/2, ~6,330.
  Both atif match. Per-rollout setup ~1 s.
- mini-swe-agent tool output arrives as user messages, so our 4,000-char tool cap does not apply. The agent truncates
  output itself.
- vLLM: restart_vllm.sh MAX_LEN (now 24576). KV pool 71,369 blocks x 16 = 1.14M tokens.
- grpo.py:
  - `--harnesses` (task index mod n).
  - Tool calls from ATIF (the blog's count_tools; falls back to the parsed count).
  - first_prompt logged; per-harness correct_rate / peak_context / rollouts metrics.
  - `--lr-schedule cosine --warmup-steps --min-lr`.
- reward.py `tool_ctx`: correct = 1 + 0.1*15/(15+calls) + 0.1 if (peak - first prompt) <= 15k; wrong = 0.
- 59 tests pass.
- Launcher run_train_mh.sh (p8_mh); baselines run_mh_baseline.sh (p7 u350 on val118 per harness at 24k).

### Multi-harness baseline: p7 u350 (pi-trained) on val118 r1, 24k window, turn cap 25 (2026-10-21)
| Harness | pass@1 | submit | out of ctx | peak ctx median / mean | generated/rollout median / p90 / max | tools | sec median |
|---|---|---|---|---|---|---|---|
| pi | 53.4 | 0.92 | 0.0 | 4.9k / 6.6k | 1.9k / 6.9k / 16.9k | 7.6 | 88 |
| mini-swe-agent | 61.0 | 0.97 | 0.8% | 6.0k / 7.7k | 2.5k / 7.6k / 15.8k | 10.2 | 121 |
| codex | 34.7 | 0.72 | 3.4% | 13.3k / 15.1k | 3.5k / 12.8k / 17.0k | 13.0 | 175 |
p7 u350 at 16k on pi was 52.5, so 24k is neutral for pi. Pi training transfers to mini-swe-agent (better than pi), but
not to Codex: 28% no answer and a ~6.3k prompt; Codex rollouts are 2x slower. Mean roots 1.06 for codex (merge fix works).
p8_mh started 2026-10-21 ~10:00 UTC (run_train_mh.sh 200).

## p8_mh result (1 epoch: 200 updates x 4 tasks x 8 = 6,400 rollouts, 798 tasks once; 2026-10-21 10:00 -> 2026-10-22 ~05:30 UTC)
- Stable: no crashes, ~5.3 min/update, 228 timeouts (3.6%), grad 0.088 -> 0.068, clip ~0.033.
- Train-batch correct (20-update windows): first pi 0.53 / mini 0.78 / codex 0.62 (all 0.64); last pi 0.71 / mini 0.72 / codex 0.77 (all 0.73). Codex still rising at the end.
- Efficiency: peak context 9.1k -> 7.4k, tool calls 9.3 -> 8.0.
- Groups: 411 mixed (51%), 322 all-correct (40%, bonus-only signal), 67 all-wrong (8%).
- val118 r1, 24k, p8 u200 vs p7 u350: pi 60.2 vs 53.4 (+6.8); mini-swe-agent 57.6 vs 61.0 (-3.4, within +-4.5 noise); codex pending.
- Next (run_p8_pipeline.sh, tmux dataagent-p7:pipe): test250 for p8u200 on pi/mini/codex + p7u350 on mini/codex at 24k,
  then epoch 2 = run_train_mh_e2.sh (p9_mh_e2: from checkpoint-200, manifests/train_e2.json = 477 tasks not solved 8/8,
  lr 2e-5 cosine, 120 updates, save every 20).

### test250 r1, 24k, turn cap 25 (2026-10-22): p8 u200 vs p7 u350, same tasks
| Harness | p7 u350 | p8 u200 | gained / lost | Blog multi-harness |
|---|---|---|---|---|
| pi | 61.6 (16k) | 58.4 | 19 / 27 (noise) | - |
| mini-swe-agent | 64.0 | 62.8 | 21 / 24 (noise) | 64.8 |
| codex | 39.6 | **53.6** | **43 / 8** (real) | 53.6 |
- Codex gain is real: submit 63% -> 88%, truncated 5.6% -> 1.2%, peak 14.8k -> 12.3k. Pi/mini flat.
- Noise check: same p7 u350 run twice on val118 flips 21/118 tasks; val118 r1 "pi +6.8" was noise.
- Pi regression: unparsed final tool calls (bad quoting in bash(command='...')) 2 -> 7 on test250.
- Verdict: multi-harness training fixed the weak harness without moving the strong ones. Blog parity on codex
  and mini-swe at 24k (blog served 128k). Epoch 2 (p9_mh_e2) started 14:41 UTC.

## p9_mh_e2 result (epoch 2: 120 updates on train_e2, from p8 checkpoint-200; 2026-10-22 14:41 -> 2026-10-23 03:33 UTC)
- Train: on the same tasks epoch 2 beat epoch 1 by +5..+11 pts per 20-update window; raw batch accuracy fell after u40 only because later tasks are harder.
- test250 r1, 24k, checkpoint-120 (gained/lost vs p8 u200):
| Harness | p7 u350 | p8 u200 | p9 u120 | vs p8 | Blog |
|---|---|---|---|---|---|
| pi | 61.6 | 58.4 | **63.2** | 23 / 11 (borderline, p~0.06) | - |
| mini-swe-agent | 64.0 | 62.8 | 63.6 | 17 / 15 (noise) | 64.8 |
| codex | 39.6 | 53.6 | 52.0 | 18 / 22 (noise) | 53.6 |
| mean of 3 | 55.1 | 58.3 | **59.6** | | |
- Pi unparsed final tool calls back to 1 (p8: 7, p7: 2); submit 92%.
- Verdict: p9 u120 = best overall checkpoint; codex gain from p8 kept, pi recovered. Gains vs p8 mostly within r1 noise.

## Confirmation runs (2026-10-23, run_p9_confirm.sh; merge with analysis/merge_r4.py)
- OpenCode (never trained on), test250 r1: p7 u350 44.4 -> p9 u120 50.8 (+6.4); blog OpenCode 49.6.
- Claude Code first try INVALID: 500/500 "Agent setup timed out after 360 s" (per-rollout npm install at conc 32).
  Fix: Claude Code 2.1.306 prebaked (Dockerfile.mh -> pirl/sandbox:pi-mh2, retagged train:base; rollback: tag pirl/sandbox:pi-mh).
  Smoke 2 tasks OK (48 s). Rerun queued: run_cc_rerun.sh (after confirm).
- pi r4 (1000 attempts each): p8 u200 61.0, p9 u120 60.9, diff -0.1 +- 2.2 -> the r1 "+4.8" was noise.
- r4 final (1000 attempts each, p8 u200 vs p9 u120): pi 61.0 vs 60.9; mini-swe-agent 64.1 vs 63.6 (-0.5 +- 2.4);
  codex 57.2 vs 54.1 (-3.1 +- 2.5, significant). Verdict: epoch 2 did not help and slightly hurt codex -> p8 u200 is the best checkpoint.
  vs blog (128k): mini-swe 64.1 vs 64.8, codex 57.2 vs 53.6.
