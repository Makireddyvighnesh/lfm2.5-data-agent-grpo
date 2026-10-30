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

## Epoch 1 on unseen harnesses (2026-10-24, test250 r1, 24k)
- Stage 2 (p8 u200): OpenCode 56.0 (Stage 1 44.4, Stage 3 50.8), Claude Code 53.2 (Stage 1 46.4, Stage 3 52.4).
  Paired vs Stage 1: OpenCode +11.6 +- 6.2, Claude Code +6.8 +- 5.5. Stage 2 is best on all five harnesses.
- Average over the reference's 4 harnesses: Stage 2 57.6 vs reference multi-harness 54.2 (reference base 42.2).

## p10_mh_r32: multi-harness from scratch (started 2026-10-26, tmux dataagent-p10, run_train_p10.sh)
- Fresh LoRA r32 / alpha 64 on base (no p7/p8 init). Harnesses pi + mini-swe-agent + codex, train1000 (all tasks, incl. the 202 p6 never solved), p8 recipe
  (24k, thinking OFF (p8: ON), prefix chain, pi bash-only, turn cap 25, 600 s, tool_ctx, lr 3e-5 cosine warmup 10 min 3e-6, FP8 + TIS 2).
- Batch: 8 tasks x 8 rollouts = 64 per update (packed 32k-token micro-batches, one optimizer step), 200 updates
  (~2 passes), cosine spans all 200 (new grpo.py --schedule-steps; backup runs/backups/claude-20261015/grpo.py.bak-20261026).
- Streaming 1 extra group, concurrency 72 (p8: 2 groups, 48; the trainer needs conc >= (groups + stream) x 8).
- DAPO dynamic sampling (new grpo.py --dynamic-sampling): finished groups with no correctness contrast (all correct or
  all wrong) are dropped (filtered_rollouts.jsonl) and replaced by new tasks; cap 32 drops per update.
  Earlier starts stopped before update 1: train798 (runs/p10_mh_r32.aborted-798), thinking ON (.aborted-think).
- Eval every 100 updates: val118 + test250, r1, 24k, each of the 3 harnesses (trainer pauses during eval).
- Progress 2026-10-27 02:40 UTC: u41/200, ~18 min/update (gen 13-16, train 4.3), 0 infra errors, peak mem 13.8 GiB.
  Kept-group correct (10-update windows): 0.43 -> 0.47 -> 0.51 -> 0.53; codex 0.25 -> 0.51, submit 0.71 -> 0.91.
  Dropped groups/update 6.8 -> 10.0; all-correct share of drops 0.9 -> 5.3 (model is solving more tasks 8/8).
- Speed diagnosis (u30-44): update ~18-20 min = generation 15.3 + training 4.4 + sync 0.04. vLLM ran only ~30-39 of 72
  rollouts: streaming kept at most groups+1 = 9 groups in flight, and finished groups waiting for the update held their
  slot, so at update time 1 group was still generating (groups_in_flight 1.0). KV use 12-17%, CPU 57% idle, rollout
  overhead outside the agent ~9 s. Trainer 2,350 tok/s with full gradient checkpointing (compute-bound; no change).
- Speed changes from u45 (2026-10-27 03:30 UTC, grpo.py backup runs/backups/claude-20261015/grpo.py.bak-20261027-dapo):
  `--async-groups 14` (keep 14 UNFINISHED groups generating; finished groups wait for the next update), conc 112,
  vLLM max-num-seqs 96 + 6 LoRA slots (restart_vllm.sh MAX_LORAS), `--dynamic-skip` (revisit skipped once after an
  all-correct visit or two all-wrong visits; skipped_tasks.jsonl; takes effect from the 2nd pass, ~u62), agent
  wall-clock cap 900 s (in-flight rollouts now pause during each ~4.5-min training phase; active cap stays ~600 s).
  New test tests/test_grpo_dynamic.py (60 tests pass).
- Restarts after the speed change: Harbor hit its 128-session cap (needs >= 2x 112 slots) -> --max-envs 256; trainer
  waited up to agent_timeout for in-flight rollouts on exit -> now aborts them (vLLM abort-sleep, os._exit, positions
  saved); pipeline waits for both Harbor ports before restarting; resume guard override --allow-change agent_timeout
  (config_changes.jsonl).
- Steady-state timing (u47-49): 8.1 / 10.8 / 9.7 min (gen 3.7-7.5, train 3.2-4.4), lag 1.0, KL 0.0084 (= before), 0 infra.
  Sampler over one cycle: training = 42% of wall time; generation 980 tok/s mean (1,180 at >=40 running = 20 tok/s/seq,
  ~30% under the 64-seq benchmark), 96% prefix-cache hits. vLLM was padding the r32 adapter to LoRA rank 64 ->
  restarted at u49 with --max-lora-rank 32 (05:26 UTC).
- Skip rule changed at u53 (user decision, 06:27 UTC; backup grpo.py.bak-20261027-skip1): a task is removed for the
  rest of the run once a visit was all correct; a revisit after an all-wrong visit is skipped with p=0.5
  (--skip-wrong-frac, coin seeded by task + visit count). Pass-1 groups so far (839): mixed 50%, all-wrong 29%,
  all-correct 21% (codex: 54 / 34 / 12) -> ~35% of pass-2 visits skipped.
- Epoch-1 stop (user decision, 07:46 UTC at u59): grpo.py --stop-at-pos 1000 (submit no position >= 1000, train on the
  remaining epoch-1 groups, stop; test in tests/test_grpo_dynamic.py, 61 pass). run_train_p10.sh EPOCH_STOP=1000 keeps
  the adapter in runs/p10_mh_r32/epoch1 and runs test250 r1 (24k, thinking OFF, 600 s) on pi / mini-swe-agent / codex
  as p10e1. Bug fixed on the way: run_test250_p7.sh hardcoded PIRL_THINKING=1 (now ${PIRL_THINKING:-1}).

### p10 epoch 1 (u63, from scratch, thinking OFF) on test250 r1, 24k, 600 s (2026-10-27 08:46-11:00 UTC)
| Harness | p10e1 | p8 u200 (thinking ON, from p7) | p7 u350 | Blog multi-harness |
|---|---|---|---|---|
| pi | 55.2 | 58.4 | 61.6 (16k) | - |
| mini-swe-agent | 63.2 | 62.8 | 64.0 | 64.8 |
| codex | 50.0 | 53.6 | 39.6 | 53.6 |
| opencode (not trained) | 44.0 | 56.0 | 44.4 | 49.6 |
| claude-code (not trained) | 47.6 | 53.2 | 46.4 | 48.8 |
- Valid: 0 infra errors; 0 rollouts without a model call except claude-code 8/250; claude-code out of context 43/250
  (its ~10.8k-token prompt in the 24k window).
- Blog 4-harness average (mini, codex, opencode, claude-code): p10e1 51.2, p8 u200 57.6, blog multi-harness 54.2, blog base 42.2.
- r1 SE ~3 pts per harness. p10e1 = 63 updates x 64 rollouts from base; p8 = 200 x 32 on top of the p7 lineage + thinking ON.
- No base thinking-OFF test250 number yet, so the epoch-1 gain over its own starting point is not measured.
- Epoch 2 started 2026-10-27 15:44 UTC (user decision): EPOCH_STOP=2000 ./run_train_p10.sh resumes from u63 (pos 1000),
  stops when all positions < 2000 are done, keeps runs/p10_mh_r32/epoch2, test250 r1 on all 5 harnesses as p10e2.
  Skips active from u64 (solved tasks removed, 50% of all-wrong revisits skipped). LR keeps the 200-update cosine.
- Harbor session leak (u64-67, 28 infra errors "at capacity: 256/256"): sessions of rollouts abandoned at trainer stops
  (u49/u53/u59, ~350 rollouts) are never freed by Harbor (up since 04:38). Fix 16:47 UTC at u67: the pipeline now
  restarts Harbor before every training start (FRESH_HARBOR=1 in run_train_p10.sh).

### p10 epoch 2 (u109) on test250 r1, 24k, thinking OFF (2026-10-28, evals done 02:50 UTC)
- Epoch 2 = u64-109 (46 updates), u70+ mean 10.9 min/update, 0 infra errors after the Harbor restart, KL 0.0087.
| Harness | p10 e1 (u63) | p10 e2 (u109) | gained / lost (paired) | p8 u200 | Blog |
|---|---|---|---|---|---|
| pi | 55.2 | 56.4 | 20 / 17 (z +0.5) | 58.4 | - |
| mini-swe-agent | 63.2 | 62.0 | 20 / 23 (z -0.5) | 62.8 | 64.8 |
| codex | 50.0 | **57.6** | 37 / 18 (**z +2.6**) | 53.6 | 53.6 |
| opencode (not trained) | 44.0 | 37.6 | 21 / 37 (**z -2.1**) | 56.0 | 49.6 |
| claude-code (not trained) | 47.6 | 47.2 | 26 / 27 (z -0.1) | 53.2 | 48.8 |
- Valid: 0 errors; claude-code 1 rollout without a model call, out of context 18.8%.
- Blog 4-harness average: e1 51.2 -> e2 51.1 (p8 57.6, blog multi-harness 54.2).
- Reading: epoch 2 helped the hardest trained harness (codex, now above p8 and the blog) and hurt OpenCode, which is
  never trained on; pi / mini / claude-code unchanged within noise.

### Why p10 is low on OpenCode / Claude Code (audit 2026-10-28, analysis/tool_audit.py + ATIF trajectories)
No pipeline bug found: same grader, Harbor config and 24k window as the p8 evals, eval ran with PIRL_THINKING=0
(matches training), 0 infra errors. The losses are model behaviour in harnesses it never trained on:
| Run | pass | context out | wrong tool name | cmd killed (timeout in s, harness wants ms) | clean |
|---|---|---|---|---|---|
| opencode p8 u200 | 56.0 | 4 (0.00) | 9 (0.33) | 95 (0.51) | 142 (0.63) |
| opencode p10 e2 | 37.6 | 1 (0.00) | 62 (0.29) | 138 (0.41) | 49 (0.41) |
| claude-code p8 u200 | 53.2 | 46 (0.04) | 19 (0.37) | 0 | 185 (0.67) |
| claude-code p10 e2 | 47.2 | 47 (0.02) | 64 (0.38) | 0 | 139 (0.67) |
(rollouts, pass rate in brackets; categories assigned in that order)
- Wrong tool names carried over from training harnesses: OpenCode `finish` (92 calls / 62 rollouts); Claude Code
  lowercase `bash` 143, ToolSearch 111, Agent 31 (85 rollouts; p8: 68 calls / 28 rollouts).
- OpenCode bash `timeout` passed in seconds (pi semantics) -> killed after N ms: 1,669 of ~2,000 timeout args in p10 e2,
  kills in 193/250 rollouts (p8: 101). Claude Code is not affected.
- Claude Code context exhaustion ~18% for every model (p7/p8/p9/p10): its ~10.8k-token prompt in the 24k eval window.
- Thinking OFF costs accuracy: p10 rollouts where the model still reasons in text (emits </think>) pass far more often
  (OpenCode 0.42 vs 0.30, Claude Code 0.52 vs 0.32, Codex 0.60 vs 0.43, Pi 0.58 vs 0.41). p8 reasoned on 99% of turns.
- Claude Code gap is fully the wrong-tool habit (clean rollouts 0.67 for both). OpenCode also has a general gap
  (clean 0.41 vs 0.63), consistent with thinking OFF.
- User decision 2026-10-28: thinking stays OFF (the blog used thinking OFF), so p10 is compared with the blog, not p7-p9.
- User decision 2026-10-28: OpenCode and Claude Code stay held out (test-only) to measure adaptation to unseen harnesses.
- Why epoch 2 is lower on OpenCode (44.0 -> 37.6; other harnesses flat or up):
  - Harness errors are NOT more frequent in e2 (timeout args 1,866 -> 1,671; rollouts with ms-kills 138 -> 138;
    `finish` calls 87 -> 94). `finish` is harmless: the answer was written before it in 61/62 rollouts.
  - The drop is give-ups: "Not Applicable" answers in OpenCode 25 -> 38 (never correct: 0/250 test golds, 0/1000 train
    golds are "Not Applicable"); 87-88% of those rollouts had commands killed by the ms timeout. Other harnesses:
    NA counts unchanged (pi 11->8, mini 6->7, codex 23->22, claude-code 18->22).
  - Lost-task answers (37): wrong label 22, no answer 9, wrong number 6.
  - Reading: after epoch 2 the model more often concludes "no applicable answer" when its OpenCode commands die,
    a failure mode the training harnesses never produce. Not a pipeline bug.
- Why Claude Code did not improve (47.6 -> 47.2), 2026-10-28:
  - Clean rollouts improved (pass 0.62 -> 0.67) but rollouts with harness-habit errors grew (46 -> 63, pass ~0.38):
    lowercase `bash` calls 78 -> 143 (Claude Code: "No such tool… call Bash instead"); ToolSearch 72 -> 111 and Agent
    are valid Claude Code tools but unhelpful detours. Net zero.
  - Context exhaustion ~19% at every epoch (p8 too): ~10.8k-token fixed prompt leaves ~13k of the 24k window. The tool
    cap DOES apply (checked: Read results arrive capped at ~4,047 chars; the 92k-char sizes were Claude Code's own
    uncapped log). The window fills by accumulation: whole-file Reads of two files per call (one rollout hit Read's
    256 KB limit and retried with allow_large=true), long assistant messages (up to 10k chars), ~15 turns.
- Claude Code at a 32k window (2026-10-28, run_cc32k.sh; only vLLM max_model_len changed):
  p10e2 47.2 (24k) -> 46.8 (32k), out of context 18.8% -> 1.6%; p10e1 47.6 -> 46.0, 17.2% -> 2.4%.
  Of the 47 e2 tasks that ran out of context at 24k: 7 correct at 32k, 17 no answer (median 16 turns, ~23k peak),
  3 Not Applicable, 3 still out. Paired 24k->32k on e2: 26 gained / 27 lost (r1 noise). The window is not the
  bottleneck: the overflowing tasks are long, non-converging rollouts. ~47% is p10's real Claude Code level.
- OpenCode epoch-2 repeat (24k, independent second run, label mh24k_opencode_p10e2rep): 41.2 (first run 37.6).
  Same adapter, 63/250 tasks flip between the two runs (r1 noise is large). Mean of two e2 runs 39.4 vs e1 44.0
  (single run). e1 vs repeat paired: 31 gained / 38 lost (z -0.8). "Not Applicable" reproduces: e1 25, e2 38 and 41.
  Reading: the e2 OpenCode drop is ~4-5 pts, about half of the first estimate; the give-up increase is real.
- Epoch 3 started 2026-10-28 07:07 UTC (user decision): EPOCH_STOP=3000 from u109 (pos 2000); 336 tasks already
  solved 8/8 are removed, so ~660 remain before the 50% all-wrong skip. Test250 on all 5 harnesses at the end (p10e3).
  Question it answers: harness overfitting (trained harnesses hold, held-out OpenCode / Claude Code fall) vs noise.
- Epoch 3 done at u150 (2026-10-28 14:55 UTC; u110-150, ~11.9 min/update, 0 infra). Eval order changed to held-out
  first (run_p10e3_eval.sh took over the pipeline end). OpenCode p10e3 38.0 (e1 44.0, e2 37.6 / 41.2):
  "Not Applicable" 25 -> 38/41 -> 43; ms-kill rollouts ~185-193 and timeout arg on ~50% of calls at every epoch.
- Claude Code p10e3 42.0 (e1 47.6, e2 47.2): lowercase `bash` calls 78 -> 143 -> 231, rollouts with it 20 -> 37 -> 51
  and their pass 0.40 -> 0.43 -> 0.27; clean rollouts 0.59 -> 0.61 -> 0.56; "Not Applicable" 18 -> 22 -> 28;
  out of context 43 -> 47 -> 37. Same pattern as OpenCode: training-harness habits grow every epoch.
- 2026-10-28 ~16:05 UTC: epoch-3 eval stopped by the user after OpenCode (38.0) and Claude Code (42.0); pi was cut
  mid-run (partial output removed), mini-swe-agent and codex not run. vLLM stopped; GPU idle. Adapters kept:
  runs/p10_mh_r32/epoch1, epoch2, epoch3 (u63 / u109 / u150).
- Why the model picks wrong tool names despite a correct tool list: correct training rollouts contain ~2,340 `bash`
  calls (pi + mini), and also calls to a nonexistent `finish` (codex 112, pi 57 in an 800-rollout sample) that still
  earned full reward, so RL reinforces them. In Claude Code the model's probability for the wrong `bash` token rises
  0.85 (e1) -> 0.85 (e2) -> 0.96 (e3) while its calls to it go 77 -> 131 -> 218. The tool list never mattered in
  training, so RL sharpened the habit until the prompt cannot override it.

## p11_mh4_r32: 4 training harnesses + tool-error penalty (started 2026-10-28, run_train_p11.sh, tmux dataagent-p11)
- User choice (option 3): add Prime Agent as a 4th TRAINING harness (offers only `ipython`) and penalise invented
  tools. OpenCode and Claude Code remain held out. Everything else = p10 (fresh LoRA r32, thinking OFF, 1000 tasks,
  8 x 8, DAPO dynamic sampling + async + skip rules, 24k, lr 3e-5 cosine/200, FP8 + TIS).
- Reward tool_ctx_err (pirl/reward.py): tool_ctx - 0.1 x unknown-tool calls (cap 0.3), on every scored rollout.
  Unknown = called name not in that request's captured `request_tools` (pirl/grpo.py unknown_tool_calls).
  Checked on p10 evals: Codex 455 unknown calls in 172/250 rollouts (read_file 212, finish 206), Pi 67 (read, finish),
  mini 0, OpenCode 41 (finish), Claude Code e3 368 (bash 222).
- Dynamic sampling keeps groups with correctness OR unknown-tool contrast (--dynamic-keep-tool-contrast).
- Prime Agent smoke (base, 8 train tasks, 24k): mechanics OK (atif match, 1 root, first prompt ~8.4k, peak <=11k,
  0 out of context); 1/8 correct; base calls `bash` 3-25 times per rollout although only `ipython` is offered.
- prime_agent_harness.py: context window from PRIME_AGENT_CONTEXT (was hardcoded 32768). 64 tests pass.
- BUG (found 2026-10-29 at u46): unknown_tool_calls() was given the training trace, whose turns have no
  request_tools, so every rollout read 0 unknown calls: the tool_ctx_err penalty and --dynamic-keep-tool-contrast
  were inert for u1-46. (The pre-launch check used capture["result"], which has the field; the unit test used a
  synthetic payload.) Fix: read the result payload; return None (not 0) when no turn carries a tool list; test added.
  The running trainer still has the old code. So far p11 = "Prime Agent as 4th harness, no penalty".
- Real unknown-tool calls in p11 training (recomputed from captures): Prime 11.1/rollout (u1-10) -> 2.2 -> 1.6 -> 1.5
  -> 1.4 (u41-50) but still >=1 in 98% of Prime rollouts; Codex flat ~0.33/rollout (25% of rollouts); Pi ~0; mini 0.
  Prime alone teaches most of the `bash`->`ipython` switch; Codex's invented read_file/finish get no pressure.
- Audit 2026-10-29 (p11 u48, epoch 1 running unchanged):
  - Not bugs: packed rows are isolated (pirl/lfm2_packing.py, Oct 3: mamba-ssm conv kernel + seq_idx hook + startup
    check; FA3 hub kernel handles attention varlen); KL ~0.0097 = one-update policy lag of async generation (lag 0.86),
    TIS covers it; no recompiles (2,390 tok/s steady); multi-root rollouts <=4%.
  - Time: update 14.3 min = generation 9.5 + training 4.8. vLLM 73-80 running, KV 37%, GPU 90-100%, 95% prefix hits;
    effective decode ~908 tok/s. Generated tokens: kept groups 43%, dropped all-wrong 48%, dropped all-correct 9%.
    Rollouts >17 calls: 9% of rollouts, 12% of tokens, 7% correct. Calls hitting 4,096 tokens: 9% of rollouts, 11% of
    tokens, 0% correct, but rollouts whose longest call is 2,048-3,071 tokens are 22% correct -> per-call cap kept.
    800 s+ rollouts (19%, 8% correct) mostly spanned 1-2 training phases (vLLM asleep, wall clock running).
  - Changes from epoch 2 (run_train_p11.sh, auto-started by run_p11_chain.sh after the epoch-1 evals):
    unknown-tool counter fix (penalty + tool-contrast keep become active), training turn cap 25 -> 17 (blog value;
    evals stay 25), probe sampling --probe-hard 4 for tasks last seen all wrong (pirl/grpo.py resolve_probes; test in
    tests/test_grpo_dynamic.py). 65 tests pass. Backups: runs/backups/claude-20261015/grpo.py.bak-20261029-probe.
- p11 stopped at u50 (2026-10-29 ~04:20 UTC, user decision) before the end of epoch 1; no eval. Data kept.

## p12_mh4_r32: fresh start with all fixes (2026-10-29, run_train_p12.sh, tmux dataagent-p12)
- = p11 design (4 training harnesses incl. Prime Agent; OpenCode/Claude Code held out; reward tool_ctx_err) with the
  fixed unknown-tool counter, tool-contrast keep, training turn cap 17 and probe sampling active from update 1.
- Epoch-1 test250 r1 (p12e1): OpenCode 37.2, Claude Code 47.2, Pi 58.8, mini-swe-agent 62.9, Codex 52.4 (Prime Agent
  pending). Failure buckets (analysis/failure_buckets.py): OpenCode 186/250 rollouts hit the seconds-as-ms bash
  timeout kill (~11 pts); Claude Code 31 context-out (~7 pts), invented tools now ~0.6 pts (p10 e2: 6.7).
  Claude Code context-out makeup: fixed prompt ~10.8k, model's own text ~7k, tool outputs (after 4k cap) ~3.7k.
- Changes from epoch 2 (2026-10-29, run_train_p12.sh when EPOCH_STOP >= 2000; auto-started in tmux dataagent-p12:e2
  after the epoch-1 evals): Pi tools bash -> bash,read (file reading under RL pressure; p7 bash-only had cut full
  reads 90% -> 0% by removal, not learning); reward tool_ctx_err -> tool_ctx_read (--allow-change reward):
  - capped-output penalty 0.05 per tool output over the 4000-char proxy cap, max 0.2, on every scored rollout
    (pirl/grpo.py capped_tool_outputs, counted from result request_messages, recorded pre-cap);
  - own-context bonus graded 0.1 at <= 4k own tokens -> 0 at 12k (was flat 0.1 under 15k, paid to ~all e1 rollouts:
    own context mean 3.5-5k);
  - groups differing only in capped outputs are kept by --dynamic-keep-tool-contrast.
  - e1 training rate of capped outputs per rollout: Pi 0.29, Prime 0.21, mini 0.14, Codex 0.07.
  - Caveat: e2 test250 Pi runs with bash,read, e1 with bash only (not directly comparable on Pi).
  67 tests pass (2 new in tests/test_reward.py).
- Epoch 2 done at u127 (2026-10-30 ~02:20 UTC). Epoch-2 test250 r1 (p12e2) vs p12e1: OpenCode 45.6 (37.2),
  Claude Code 51.6 (47.2), Pi 55.2 (58.8; e2 runs Pi with bash,read, not comparable), mini-swe-agent 63.6 (62.9),
  Codex 54.4 (52.4), Prime Agent 54.2 (47.2; 249 graded).
  - OpenCode: timeout-kill rollouts 186 -> 133 (pass 0.32 vs 0.61 without); clean-rollout pass 0.50 -> 0.65.
  - Claude Code: context-out unchanged (31 -> 30); clean-rollout pass 0.58 -> 0.66; invalid-tool cost 0.6 -> 2.8 pts.
  - Full-CSV Read calls on held-out harnesses did NOT fall (Claude Code 428 -> 429, OpenCode 285 -> 269; with a
    limit 135 -> 96, 63 -> 20). Training: Pi capped outputs/rollout 0.87 -> 0.59 over e2; own context flat ~5k.
    So the held-out gains are not explained by the read/context terms.
