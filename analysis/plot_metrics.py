"""Track and plot every training run and held-out evaluation of the Data Agent project.

Writes (re-run any time; the tmux window `plots` in session dataagent-p2 refreshes them every 10 min):
  results/metrics_long.csv   run, phase, update, metric, value  - every numeric per-update metric of every run
  results/evals.csv          one row per held-out evaluation (val118 / test250)
  results/plots/phase1_training.png   r1 (plain GRPO) vs r2 (root sampling) on train100
  results/plots/phase2_ablation.png   reward ablation arms A/B/C on hard50
  results/plots/heldout_val118.png    pass@1 of every evaluated checkpoint
usage: ../SmolDataEnvs/.venv/bin/python analysis/plot_metrics.py
"""
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DA = Path(__file__).resolve().parents[1]
OUT = DA / "results" / "plots"
# Reference palette (dataviz skill), first three slots: validated all-pairs for CVD in light mode.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
PHASES = {
    "phase1": [("grpo_r1", "r1: plain GRPO"), ("grpo_r2", "r2: + root sampling")],
    "phase2": [("p2_A_tool", "A: blog tool reward"), ("p2_B_context", "B: context reward"),
               ("p2_C_correctness", "C: correctness only")],
}
PANELS = [  # (metric, title, scale, y-limits)
    ("correct_rate", "Training correctness (%)", 100, (0, 100)),
    ("context_exhausted_rate", "Ran out of context (%)", 100, (0, 100)),
    ("correctness_contrast_frac", "Groups with a learning signal (%)", 100, (0, 100)),
    ("tool_calls_mean", "Tool calls per rollout", 1, None),
    ("peak_context_mean", "Peak context per rollout (tokens)", 1, None),
    ("step_s", "Minutes per update", 1 / 60, None),
]


def load(run):
    p = DA / "runs" / run / "metrics.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def write_tracking():
    rows = []
    for phase, runs in PHASES.items():
        for run, _ in runs:
            for m in load(run):
                for k, v in m.items():
                    if isinstance(v, (int, float)) and not isinstance(v, bool) and k != "step":
                        rows.append((run, phase, m["step"], k, v))
    with open(DA / "results" / "metrics_long.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "phase", "update", "metric", "value"])
        w.writerows(rows)
    evals = []
    for p in sorted((DA / "results").glob("*.json")):
        if not (p.name.startswith("val118_") or p.name.startswith("test250_") or p.name == "base16k_val118.json"):
            continue
        s = json.loads(p.read_text())
        split = "test250" if p.name.startswith("test250") else "val118"
        name = p.stem.split("_", 1)[1] if "_" in p.stem and p.name != "base16k_val118.json" else "base"
        evals.append({"split": split, "model": name, "pass_at_1": round(s["pass_at_1"], 4),
                      "correct": round(s["pass_at_1"] * s["graded"]), "graded": s["graded"],
                      "out_of_context": round(s["context_exhausted_rate"], 4), "tool_calls": round(s["mean_tool_calls"] or 0, 2),
                      "peak_context": round(s.get("mean_peak_context") or 0), "seconds": round(s["mean_seconds"] or 0),
                      "mtime": p.stat().st_mtime})
    evals.sort(key=lambda e: e["mtime"])
    with open(DA / "results" / "evals.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[k for k in evals[0] if k != "mtime"] if evals else ["split"])
        w.writeheader()
        for e in evals:
            w.writerow({k: v for k, v in e.items() if k != "mtime"})
    return len(rows), evals


def rolling(vals, k=3):
    out = []
    for i in range(len(vals)):
        window = [v for v in vals[max(0, i - k + 1):i + 1] if v is not None]
        out.append(sum(window) / len(window) if window else None)
    return out


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_phase(phase, title, fname):
    runs = [(run, label, load(run)) for run, label in PHASES[phase]]
    runs = [r for r in runs if r[2]]
    if not runs:
        return False
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.6), facecolor=SURFACE)
    for ax, (metric, ptitle, scale, ylim) in zip(axes.flat, PANELS):
        style(ax)
        ax.set_title(ptitle, fontsize=10, color=INK, loc="left")
        for i, (run, label, rows) in enumerate(runs):
            xs = [m["step"] for m in rows]
            ys = [None if m.get(metric) is None else m[metric] * scale for m in rows]
            color = SERIES[i]
            ax.plot(xs, [y if y is not None else float("nan") for y in ys], color=color, linewidth=1, alpha=0.25)
            sm = rolling(ys)
            ax.plot(xs, [y if y is not None else float("nan") for y in sm], color=color, linewidth=2, label=label)
            if sm and sm[-1] is not None:  # direct end label (relief for the low-contrast slot), text in ink
                ax.plot([xs[-1]], [sm[-1]], "o", color=color, markersize=5, markeredgecolor=SURFACE, markeredgewidth=1.5)
                ax.annotate(f"{sm[-1]:.0f}" if sm[-1] >= 10 else f"{sm[-1]:.1f}", (xs[-1], sm[-1]),
                            xytext=(5, 0), textcoords="offset points", fontsize=8, color=INK2, va="center")
        if ylim:
            ax.set_ylim(*ylim)
        ax.set_xlabel("training update", fontsize=8, color=INK2)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=len(labels), frameon=False, fontsize=9, labelcolor=INK)
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, color=INK)
    fig.text(0.01, 0.935, "Thick line: 3-update rolling mean; faint line: per update. One update = 8 tasks x 8 rollouts.",
             fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT / fname, dpi=130, facecolor=SURFACE)
    plt.close(fig)
    return True


def plot_heldout(evals):
    rows = [e for e in evals if e["split"] == "val118"]
    if not rows:
        return False
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(rows) + 1.3), facecolor=SURFACE)
    style(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    pretty = {"base": "untrained base (native)", "r2u12": "r2 update 12 (native)", "r2-final-chain": "r2 update 22 (chain)",
              "r2-final-native": "r2 update 22 (native)", "p2_A_tool-final": "phase 2 A: blog tool reward",
              "p2_B_context-final": "phase 2 B: context reward", "p2_C_correctness-final": "phase 2 C: correctness only"}
    names = [pretty.get(e["model"], e["model"]) for e in rows][::-1]
    vals = [e["pass_at_1"] * 100 for e in rows][::-1]
    ax.barh(names, vals, color=SERIES[0], height=0.6)
    for y, (v, e) in enumerate(zip(vals, rows[::-1])):
        ax.annotate(f"{v:.1f}%  ({e['correct']}/{e['graded']})", (v, y), xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=8, color=INK2)
    ax.set_xlim(0, max(60, max(vals) + 15))
    ax.set_xlabel("pass@1 on val118 (118 held-out medium+hard tasks, FP8, 16k context)", fontsize=8, color=INK2)
    ax.tick_params(axis="y", labelsize=8, colors=INK)
    ax.set_title("Held-out accuracy by checkpoint (oldest at top)", fontsize=11, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "heldout_val118.png", dpi=130, facecolor=SURFACE)
    plt.close(fig)
    return True


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    n, evals = write_tracking()
    made = [plot_phase("phase1", "Phase 1: learning on 100 train tasks (16k context, FP8)", "phase1_training.png"),
            plot_phase("phase2", "Phase 2: reward ablation on 50 unseen hard tasks", "phase2_ablation.png"),
            plot_heldout(evals)]
    print(f"metrics rows: {n}; evaluations: {len(evals)}; plots written: {sum(made)} -> {OUT}")


if __name__ == "__main__":
    main()
