# Findings log: flaws found in the benchmark, the reference setup and our setup

Purpose: material for writing up our results. Each item has a status (✔ verified by us against code/logs;
◐ reported by a review agent, spot-checked; ○ agent claim, not yet re-checked) and where the evidence lives.
Started 2026-10-12. Append new findings at the end of the relevant section, with the date.

Paths: `DA` = `MiMo_rl/Data Agent`, `SMOL` = `MiMo_rl/SmolDataEnvs`, `TUT` = `SMOL/FineEnvs/05-multi-harness-rl`,
`TRIALS` = `SMOL/runs/trials_data_agent`, `FR` = `DA/analysis/failure_review`.

---

## A. Benchmark flaws (SmolDataEnvs tasks + pinned grader) — affect Adithya's results and ours equally

Shared basis ✔: both sides use grader sha256 `16f906e9…` (pinned in `TUT/prepare.py:35`; ours in all 1,000 train /
250 test / 144 eval tasks) and identical task lists (`TUT/data/train.json`, `TUT/data/test.json` == our splits).
Grader: `tests/grader.py` tiers exact → numeric (per-task ATOL/RTOL from task.toml) → percent/fraction bridge →
list → math_verify. `tests/test.sh` runs `pip install math-verify` at grading time.

### A1. Grader false positives (wrong answers rewarded)
| Case | Gold | Accepted pred | Path | Status | Evidence |
|---|---|---|---|---|---|
| Shape after dropping rows (train 0016_714_16714217_qa_4) | `392, 9` | `398, 9` (×7) | math_verify | ✔ | `TRIALS/oe-0016_714_16714217_qa_4-*/verifier/test-stdout.txt` |
| Test accuracy (train 0055_521_55521574_qa_5) | `0.82` | `78.43`, `78.4`, `78.18` (×10) | numeric_scaled | ✔ | `TRIALS/oe-0055_521_55521574_qa_5-*` |
| ADF p-value (train 0035_712_35712790_qa_3) | `1.255263e-07` | `5.02e-08` (×4) | numeric (abs tol) | ✔ | `TRIALS/oe-0035_712_35712790_qa_3-*` |
| 2-decimal golds ≤ 1 accept ±0.05 abs (e.g. 0.90 for 0.95; 0.0 for 0.03/0.04) | — | — | numeric | ◐ inferred from 1,377 grades | `FR/report_train_u226-250.md` §5 |
| Tiny golds (3.6e-14, 9.8e-22) accept anything tiny (1.67e-20, 5.9e-30) | — | — | numeric | ◐ | `FR/report_train_u226-250.md`, `FR/report_train_u151-175.md` |
| Superset list accepted: `Singapore, 7, South Korea, 7` for `Singapore, 7` | | | math_verify | ◐ | `FR/report_train_u101-125.md` §5.2 |
| Partial list accepted: `2` for `0, 2` | | | | ◐ | `FR/report_train_u151-175.md` §6 |
Size: ~3.3% of correct rewards are clear false positives (window 151-175); ~10.5% of rewards depend on tolerance (226-250).
**Full audit (2026-10-12) ✔:** 20,143 graded answers → 362 clearly wrong accepted, 461 accepted 1-5% off, 28 unparsed, 87 correct rejected. Root causes (absolute ATOL on tiny golds, ±0.05 on 2-dp golds, list tier reads first number in each element, math_verify fallback, zero tolerance on rounded integer golds, no label normalisation). See `analysis/failure_review/GRADER_FAILURES.md`, `grader_audit.csv`, script `analysis/grader_audit.py`.

### A2. Grader false negatives (right answers rejected)
| Case | Gold | Rejected pred | Status | Evidence |
|---|---|---|---|---|
| eval 0002_335_2335686_qa_2 | `accel. y` | `acceleration_y` (the real column name; 0/13 ever) | ◐ header checked by agent | `FR/report_eval_part4.md` A1 |
| eval 0002_152_2152188_qa_1 | `157` | `157.111` (prompt says "keep decimal precision") | ◐ | `FR/report_eval_part2.md`, `report_eval_progress.md` |
| train 0011_587 | `70` | `70.37` (57/81) | ◐ | `FR/report_train_u101-125.md` §5.3 |
| Separator inconsistency | `50` list | `50,50,50` rejected while `50, 50, 50` accepted | ◐ | `FR/report_train_u101-125.md` |
| Label spelling | `Gradient Boosting` | `GradientBoosting` | ◐ | `FR/report_train_u151-175.md` |
| eval 0011_088 | task.toml says `Riesling`, grader used `Shiraz` (Shiraz verified in data) | | ○ | `FR/report_eval_part2.md` |

### A8. Fixed grader v3 (2026-10-13 ✔)
`grader_v3.py` fixes A1/A2 deterministically (no LLM, no network). Blind check by 4 Haiku judges on 632 unique cases:
where v3 and the pinned grader disagree (1,014 rollouts) the judges side with v3 97.7% vs 3.7%; on a random sample of
agreements, v3 strict (no task RTOL) matches the judges 99.3%. Re-scored pass@1 (pinned / v3 / strict): u225 48.3 /
48.1 / 44.3; u50 39.0 / 37.3 / 33.3; base val118 10.2 / 9.3 / 8.5. Training gain u50→u225 holds or grows
(+9.3 / +10.8 / +11.0). Details: `analysis/failure_review/GRADER_V3.md`.

**A8b. Re-audit on p4_train500 rollouts (2026-10-13 ✔).** 6 Haiku 5.5 judges, blind, all 823 unique graded answers
(1,604 trials: training updates 1-33 + val118 u0/u25). Error rate vs judges: pinned 3.8% (14 false accepts / 13 false
rejects of 707 eval trials), v3.0 1.4% (7 / 15 of 1,604), v3.1 0.7% (0 / 12). v3.0 bug: golds printed "1.0" were given
±0.05, so 0.9667 was accepted for accuracy 1.0; v3.1 uses ±0.005 for "X.0" golds and maps True/False → yes/no and
"Ridge" → "Ridge regression". The training reward is sound (correct 51.6% by v3.0 vs 51.7% by the judges), so the live
run keeps its v3.0 copies. Details: `analysis/verifier_p4/REPORT.md`.

### A3. Unanswerable tasks; "Not Applicable" is never gold
- Gold needs an artifact absent from the files: Sudoku CNN parameter count (gold 256, only `sudoku.csv`; train 0085_343),
  "gradient descent implementations" (0038_189), training-loss log (0030_588), "SHAP / CDF / heatmap / dist-plot analysis",
  "in the notebook", t-SNE plot (eval 0012_330), insurance question over a flower-image dataset (0042_806). ◐
  Evidence: `FR/report_train_u151-175.md` §3.3, `FR/report_train_u101-125.md` §5.4, `FR/report_train_u226-250.md` §3.A.
- The prompt allows "Not Applicable", but no gold is N/A in any reviewed slice: honest N/A always scores 0, while a
  fabricated proxy can score 1 (0038_189: invented MLP 0.8875 accepted vs gold 0.86). ◐ **Most harmful reward signal.**

### A4. Under-specified protocol (gold matches only one unstated choice)
Seed, test_size 0.2 vs 0.3, stratify, scaling, CV vs hold-out, one-hot vs label encoding, drop_first, k, encoding
(latin-1 vs utf-8). Paired success/failure rollouts differ only in that choice (eval 0012_222: seed 42 → 78.4078 ✓ vs
unseeded 79.695; eval 0016_567: test 0.3 → 66.67 ✓ vs 0.2 → 63.33). ◐ Roughly a third of eval wrong answers.
Evidence: `FR/report_eval_part{3,4}.md`, `FR/labels_eval_part*.csv`.

### A5. "Held-out" means new questions, not new data ✔
134/250 test tasks (51/122 test datasets) and 68/118 of our val118 tasks use a dataset (`BUCKET_PREFIX`) that also
appears in the 1,000 training tasks, with different questions. Exact task overlap is 0. Applies to both setups.
Check: script in this session (compare `BUCKET_PREFIX` across `SMOL/prepared/datasets/{train,test}` and `DA/prepared/datasets/eval`).

### A6. Environment-dependent grading ✔
`test.sh` installs math-verify inside the sandbox at grading time; without network the math_verify tier silently
disappears, so the same answer can grade differently across sandboxes. (Worked in our runs: math_verify hits in logs.)

### A7. Data/sandbox traps (affect any harness)
Headerless and whitespace-delimited CSVs, CR-only line endings, 3,198-column CSVs, filenames with spaces, non-UTF-8
files, pandas 3 `str` dtype, missing packages (catboost, textblob, nltk data, vaderSentiment, imbalanced-learn,
tensorflow, langid) with no network for pip. ◐ Evidence: all `FR/report_*.md`.

---

## B. Reference (Adithya / FineEnvs article) evaluation caveats

Source: `TUT/eval/evaluate.py`, `TUT/eval/README.md`, `TUT/RESULTS.md`, `TUT/train/multi_harness.py`, `TUT/jobs/serve_model.py`. ✔
- B1. Eval protocol: Harbor, 4 harnesses (OpenCode, Claude Code, Codex, Mini-SWE-Agent), 250 tasks × 1 sample per
  harness, T 0.8, top_p 1.0, step limit 17, agent timeout 600 s, vLLM BF16 at max_model_len 131,072, thinking off.
- B2. Single sample, no confidence intervals. Their own pilot measured the same base model at 24% and 17% on
  25 tasks (`RESULTS.md`); the article says 52.3% vs 54.2% is not a significant difference.
- B3. Ungraded (infrastructure-failed) pairs are excluded from pass@1 (`evaluate.py` coverage logic). Baseline
  was 421 correct of 998 graded pairs (2 ungraded, excluded); finals had 1,000/1,000 graded.
- B4. Historical article curves came from an earlier recipe (`RESULTS.md`: "These curves do not validate the new
  pilot configuration"); the published tutorial scripts are not the ones that produced 52.3/54.2%.
- B5. All benchmark flaws in section A apply to their numbers.
- B7. (2026-10-12 ✔) Same grader failures apply to the blog's numbers: identical grader file, gold answers and per-task
  ATOL/RTOL in `test.json`. Exposure of the 250 test tasks: 9 two-decimal golds ≤ 1 with ±0.05, 5 tiny golds where
  ATOL 0.001/0.05 accepts any small value, 60 integer golds with zero tolerance (≥ 3 are rounded percentages, e.g.
  0000_682 qa_1 "56", qa_2 "92", 0001_237 qa_1 "60"), 20 multi-word labels with no space normalisation, 2 list golds.
  Test is cleaner than train (fewer list/ratio golds). In our 323 graded test-split answers: 9 false positives
  (2.8%) and 1 false negative; estimated effect on the blog's pass@1: inflated by roughly 1-3 points for base and
  trained alike (small sample, base-model answers). The blog's RL reward used the train split, which is more exposed
  (792 false positives in our train logs).
- B6. Training settings for reference: T 0.8, lr 3e-6, 8 generations, max completion 16,384, BF16, LFM template
  patched so the generation prompt is `<think></think>` (`multi_harness.py:55-67`).

---

## C. Flaws and deviations on our side

| # | Item | Status | Effect | Evidence |
|---|---|---|---|---|
| C1 | **Effective prompt limit 12,288, not 16,384**: vLLM rejects prompt + 4,096 reserved output > 16,384; all rejections at exactly 12,289 input tokens. Unintended. | ✔ | 64/472 (13.6%) u225 eval rollouts end this way; ~8.8% of training rollouts | capture findings `context_budget_exhausted`; `FR/report_train_u151-175.md` §3.1 |
| C2 | **Pi truncates tool output only at 50 KB (~15k tokens)**: one wide-CSV `head` overflows on the first call; overflowing output is not stored in the trajectory and the proxy labels it "Step budget exhausted". | ◐ | 8/29 never-solved eval tasks are pure overflow | `FR/report_eval_progress.md` §5 |
| C3 | Eval temperature 1.0 (blog 0.8) | ✔ | Probably lowers our pass@1 slightly | `results/val118_*.json` |
| C4 | Agent timeout 300 s eval / 240 s train (blog 600 s) | ✔ | 2-5% of rollouts time out | `results/*.json`, `run_train1000.sh` |
| C5 | val118 (from the 144-task eval split) ≠ blog's test250 | ✔ | Our 48% is not comparable to 52-54% | — |
| C6 | Base evaluated with `native` template, trained checkpoints with `prefix-chain` | ✔ | Part of base→trained gain may be template | `results/base16k_val118.json` vs `val118_p3_*` |
| C7 | Start point for the 1000-task run (arm A, 0.373) measured with 1 sample/task vs 4 later | ✔ | Noisier baseline | `results/val118_p2_A_tool-final.json` |
| C8 | FP8 sampler + trainer (blog BF16); Pi harness (not in blog's set); 16k cap by design | ✔ | Intended deviations; state them | PLAN.md |
| C9 | **Tool bonus amplified**: `adv = (r − mean)/max(std, 0.02)` (✔ exact on 1,600 rollouts); in 8/8 groups a ~0.01 reward gap becomes |adv| up to 0.78. 22% of groups train on "fewer calls" only. No harm observed yet (no guessing). | ✔ formula / ◐ effect | Risk on hard tasks | `FR/report_train_u226-250.md` §7 |
| C10 | Pythonic tool-call text (`<|tool_call_start|>[bash(...)]`) not parsed → episode ends with 0 | ◐ | 3-5% of rollouts | all `FR/report_*.md` |
| C11 | `</think>` leaks into visible text. **Corrected 2026-10-13 ✔:** caused by the prefix-chain template, not by RL: the untrained base model leaks in 9% of rollouts with the native template (`base16k_val118`) but 80% with prefix-chain (`val118_p4_train500-u0_r4`); trained checkpoints ~90-93%. Prefix-chain renders past assistant turns as `<think></think>` + text, so the model imitates a closing tag after its reasoning. Answer files are clean. | ✔ | Context waste; a reason to revisit the chain template or strip the tag | `runs/vllm_fp8_16k_chain1_seqs64_kvfp8/chat_template.jinja` |
| C12 | Supervisor stall 2026-10-11 12:55-15:25 UTC: capture proxy cached a degraded `logprobs` probe for trl-policy-v55; fixed by `serve_harbor.py install_reprobe_degraded()` | ✔ | 2.5 h lost; no data corruption | PLAN.md 10-11 |
| C13 | Discarded sampled turns in ~36% of training rollouts; tool-count metric differs from transcript count in ~37% (metric counts the retained path only) | ○ | Possible bias toward first attempts | `FR/report_train_u151-175.md` §8, `u226-250` §5 |

---

### C14. Harness fixes measured (2026-10-13 ✔)
With the tool-output cap (4,000 chars) and output-reservation shrink in the capture proxy, the **untrained base model**
scores **30.9%** on val118 (4 samples/task, pinned grader), vs 10.2% before; context exhaustion 86% → 9.5%;
ATIF match 100%. C1/C2 were the dominant cause of the base model's low score. (`results/val118_p4_train500-u0_r4.json`)

### C15. The 4k tool-output cap removed the pressure that taught the model to avoid `read` (2026-10-14 ✔)
First call on val118: base (no cap) `read` 93% -> r2 u12 bash 95% -> p3 u50..u250 bash 100% (u225: `head -n 5` + `wc -l`
+ pandas `shape/dtypes/head(3)` in one command, 56% use pandas in the first call). Without the cap, `read` on a large
file overflowed the 12.3k prompt (base: read-first 8% correct vs bash-first 37.5%), so RL dropped it within 12 updates.
With the p4 proxy cap (4,000 chars), `read` became harmless (read-first 30-39% correct, same as bash-first) and stayed
at 90-94% through u50. The harness fix raised the starting score but removed the learning signal for safe inspection.
Script: inline in session; data runs/val118_*/rollouts.jsonl.

### C16. Largest learnable error: the value instead of the label for "which X" questions (2026-10-14 ✔)
val118, label-gold attempts answered with a bare number: p3 u50 42.6% -> u100 29.5% -> u175 12.3% -> u225 12.1%
(label-gold correct 38.5% -> 60.5%: most of p3's u50->u225 gain). p4 (4k cap): u0 60.2%, u25 48.4%, u50 51.0%;
thinking-on base 40.0%. At p4 u50 these are 75 attempts (16% of all) on 30 tasks, often 4/4 in a group (zero
advantage, so no learning signal); in 83% of them the correct label was printed in a tool output AND written in the
model's own text before it wrote the number. Example: "Which state has the highest total homicides?" gold California,
answer 98994. Failure buckets at p4 u50 (all attempts): correct 38.3, number-for-label 15.9, different number 15.0,
no answer (stopped/bad call 7.4, step limit 4.9, timeout 3.8, context 3.6), close number (<=5%) 5.1, other label 4.2.

## D. What the results do support (for the write-up)

- On val118 (4 samples/task, T 1.0, 16k): 37.3% (start) → 48.3% at update 225; u50→u225 +9.3 pts, paired z = 3.9. ✔
- Behaviour learned without hints: whole-file `read` (93% of base first calls) → `bash` + pandas (≈100%); numeric
  answers to "which X" 51% → 16%; tool calls 7.4 → 5.9; peak context ≈5.3k. ✔ (`FR/report_eval_progress.md`)
- No reward hacking or epoch-2 memorisation found. ◐
- Plateau: u225 ≈ u250. ✔

## E. Open checks before publishing
1. test250 head-to-head in Pi under matched settings (T 0.8, 600 s): base, u225, FineEnvs/LFM2.5-2.6B-{opencode,multiharness}-RL; optionally a 131k-context run.
2. Re-grade affected rollouts with a patched grader and report both raw and corrected pass@1.
3. Re-verify ○ items (0011_088 gold mismatch; C13).
