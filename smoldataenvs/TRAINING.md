# Training and rollout analysis

The previous `runs/l4_regret200` run is stopped at saved update 35. Its checkpoints,
rollout summaries and Pi session files remain available. Its exact training token
captures were not archived and cannot be recovered from the text alone.

## Launch the fresh pilot

Run on the VM, with GPU and Docker access, from the repository root:

```bash
source .venv/bin/activate
export PYTHONPATH="$PWD/OpenEnv/envs${PYTHONPATH:+:$PYTHONPATH}"
python -u -m pirl.run_pilot --output runs/l4_fixed50 --extend-to 200
```

Use a terminal multiplexer to keep the command attached and its metrics visible.
The output directory must be new. The runner restarts this repository's Harbor
and vLLM services, evaluates the base model on 50 held-out medium/hard tasks,
then trains a fresh LoRA adapter for 50 updates. It retains the existing recipe:
LFM2.5-2.6B, eight prompts/update, four base rollouts/prompt, the regret sampler,
eight rollouts for previously unsuccessful prompts, and the efficiency reward.
The fixed task pool contains 400 medium and 600 hard tasks.

After 50 updates, it validates every committed capture, optimizer metrics,
checkpoint, client-loop cleanup and file-descriptor growth. Any unexpected
trainer restart stops this pilot. It evaluates step 50 on exactly the same
held-out tasks and sampling/template settings as the fresh baseline.

Extension to **200 total updates** requires all health checks to pass, both
evaluations to have complete validated token captures, and observed held-out
pass@1 to improve. This small evaluation is a pilot gate, not proof of
statistical significance. A failed gate saves its reasons in `health.json` and
does not extend training. `--extend-to 50` always ends after the pilot evaluation.

## Saved artifacts

| File | Contents |
| --- | --- |
| `rollouts/<rollout_id>.json.gz` | Full Harbor result, conversation/tool records, exact prompt/completion IDs, logprobs, masks, and derived training sequences |
| `rollout_metrics.jsonl` | Append-only completion and optimizer events, including failed and uncommitted attempts |
| `rollouts.jsonl` | Rollouts belonging to committed updates, with rewards, advantages and individual optimizer metrics |
| `metrics.jsonl` | Aggregate metrics per update, loss/ratio/clipping/gradients, timing, memory, FDs and remaining client loops |
| `results.json` | Latest training status and update metrics |
| `state_latest/` | Adapter, optimizer, task cursor, sampler statistics and Python/PyTorch RNG states |
| `checkpoint-N/` | Checkpoints every ten pilot updates and at completion |
| `eval_base/`, `eval_step50/`, `eval_step200/` | Evaluation rollouts, full compressed captures and summary results |
| `health.json`, `pipeline.json` | Gate findings and pipeline completion/failure status |

Raw captures are saved before optimization and remain even when an update is
discarded or a process crashes. Resume trims only the committed logs past the
restored checkpoint; it preserves the attempt log and captures. A checkpoint
swap interrupted between directory renames recovers its `.old` generation.

Example analysis:

```python
import gzip, json
import pandas as pd
rows = pd.read_json("runs/l4_fixed50/rollouts.jsonl", lines=True)
with gzip.open("runs/l4_fixed50/" + rows.iloc[0].capture_file, "rt") as f:
    capture = json.load(f)
```

Terminal progress prints one `ROLLOUT` line per attempt and an `UPDATE` line with
accuracy, reward, loss, ratio, clipping, gradient norm, duration and resources.
For a second terminal: `python -m pirl.watch runs/l4_fixed50`.

To stop training and its supervisor while preserving saved checkpoints:

```bash
python3 pirl/stop_run.py runs/l4_fixed50
```

A direct trainer also checks its `STOP` marker between rollout collection and
optimization and between updates. Keep the pilot pipeline terminal attached so
it exits when the supervisor stops. `pipeline.pid` identifies that launcher.
