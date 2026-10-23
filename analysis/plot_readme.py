"""README figures: RL training curves (p8 epoch 1 + p9 epoch 2) and test250 results. Writes docs/plots/*.png."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DA = Path(__file__).resolve().parents[1]
OUT = DA / "docs/plots"; OUT.mkdir(parents=True, exist_ok=True)
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
H = {"pi": "#2a78d6", "mini-swe-agent": "#eb6834", "codex": "#1baf7a"}  # categorical slots 1-3, validated
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left"})

def load(run):
    return [json.loads(l) for l in open(DA / f"runs/{run}/metrics.jsonl")]
p8, p9 = load("p8_mh"), load("p9_mh_e2")
X = p8 + p9; n8 = len(p8); steps = list(range(1, len(X) + 1))

def roll(vals, w=20):
    out = []
    for i in range(len(vals)):
        win = [v for v in vals[max(0, i - w + 1): i + 1] if v is not None]
        out.append(sum(win) / len(win) if win else None)
    return out

def harness_rate(key):  # rollout-weighted rolling mean per harness
    res = {}
    for h in H:
        num = [(x.get(f"{key}_{h}") or 0) * x.get(f"rollouts_{h}", 0) for x in X]; den = [x.get(f"rollouts_{h}", 0) for x in X]
        r = []
        for i in range(len(X)):
            a, b = sum(num[max(0, i - 19): i + 1]), sum(den[max(0, i - 19): i + 1]); r.append(a / b if b else None)
        res[h] = r
    return res

def epoch_marks(ax):
    ax.axvline(n8 + 0.5, color=INK2, lw=1, ls=(0, (4, 3)))
    ax.text(n8 / 2, 1.0, "epoch 1 (p8)", transform=ax.get_xaxis_transform(), ha="center", va="bottom", color=INK2, fontsize=9)
    ax.text(n8 + len(p9) / 2, 1.0, "epoch 2 (p9)", transform=ax.get_xaxis_transform(), ha="center", va="bottom", color=INK2, fontsize=9)

def label_ends(ax, series, fmt):
    for h, ys in series.items():
        i = max(j for j, y in enumerate(ys) if y is not None)
        ax.annotate(f"{h} {fmt(ys[i])}", (steps[i], ys[i]), xytext=(6, 0), textcoords="offset points", va="center", color=INK, fontsize=9)

# 1. training accuracy per harness
fig, ax = plt.subplots(figsize=(9, 4.2))
acc = harness_rate("correct_rate")
for h, ys in acc.items(): ax.plot(steps, [y * 100 if y is not None else None for y in ys], color=H[h], lw=2, label=h)
label_ends(ax, {h: [y * 100 if y is not None else None for y in ys] for h, ys in acc.items()}, lambda v: f"{v:.0f}%")
epoch_marks(ax); ax.set_xlim(1, len(X) + 45); ax.set_ylim(30, 100)
ax.set_xlabel("update"); ax.set_ylabel("correct rollouts (%)")
ax.set_title("Training accuracy by harness (20-update rolling mean)", pad=18)
ax.legend(loc="lower left", frameon=False, ncol=3)
fig.text(0.01, 0.005, "Epoch 2 trains only on tasks not solved 8/8 in epoch 1, so its batches are harder.", color=INK2, fontsize=8.5)
fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(OUT / "train_accuracy.png", dpi=150); plt.close(fig)

# 2. small multiples: reward, peak context, tool calls, grad norm (one measure per panel, shared x)
fig, axs = plt.subplots(2, 2, figsize=(10, 6.4), sharex=True)
panels = [("reward_mean", "Mean reward", None), ("peak_context_mean", "Mean peak context (tokens)", None),
          ("tool_calls_mean", "Mean tool calls per rollout", None), ("grad_norm", "Gradient norm", None)]
for ax, (k, title, _) in zip(axs.flat, panels):
    raw = [x.get(k) for x in X]
    ax.plot(steps, raw, color="#86b6ef", lw=1, alpha=0.8, label="per update")
    ax.plot(steps, roll(raw), color=H["pi"], lw=2, label="20-update mean")
    ax.axvline(n8 + 0.5, color=INK2, lw=1, ls=(0, (4, 3))); ax.set_title(title)
axs[0, 0].legend(frameon=False, loc="lower right", fontsize=8.5)
for ax in axs[1]: ax.set_xlabel("update")
fig.suptitle("RL training signals (dashed line = start of epoch 2)", x=0.01, ha="left", fontweight="bold")
fig.tight_layout(); fig.savefig(OUT / "train_signals.png", dpi=150); plt.close(fig)

# 3. test250 results: one panel per harness, models as bars, blog as a reference line
R = {"Pi": (61.6, 61.0, 60.9, None), "mini-swe-agent": (64.0, 64.1, 63.6, 64.8), "Codex": (39.6, 57.2, 54.1, 53.6),
     "OpenCode": (44.4, None, 50.8, 49.6), "Claude Code": (None, None, 52.4, 48.8)}
models = ["p7 u350", "p8 u200", "p9 u120"]; M = ["#86b6ef", "#2a78d6", "#1c5cab"]  # sequential blue: later = darker
fig, axs = plt.subplots(1, 5, figsize=(12, 3.8), sharey=True)
for ax, (hn, vals) in zip(axs, R.items()):
    for i, (m, v) in enumerate(zip(models, vals[:3])):
        if v is None:
            ax.text(i, 2, "not run", ha="center", va="bottom", color=INK2, fontsize=8); continue
        ax.bar(i, v, width=0.72, color=M[i], edgecolor=SURF, linewidth=2)
        ax.text(i, v + 1, f"{v:.1f}", ha="center", va="bottom", fontsize=8.5, color=INK)
    if vals[3] is not None:
        ax.axhline(vals[3], color=INK, lw=1.2, ls=(0, (4, 3)))
        ax.text(-0.45, 73, f"- - blog {vals[3]}", ha="left", va="top", fontsize=8.5, color=INK2)
    ax.set_title(hn + ("" if hn in ("Pi", "mini-swe-agent", "Codex") else " (unseen)"), fontsize=10)
    ax.set_xticks(range(3), ["p7", "p8", "p9"]); ax.set_ylim(0, 75); ax.set_xlim(-0.6, 2.6); ax.grid(axis="x", visible=False); ax.set_axisbelow(True)
axs[0].set_ylabel("test250 pass@1 (%)")
fig.suptitle("test250 pass@1 by harness  (p8/p9 on trained harnesses: 4 attempts/task; others: 1)", x=0.01, ha="left", fontweight="bold")
fig.tight_layout(); fig.savefig(OUT / "test250_results.png", dpi=150); plt.close(fig)
print("wrote", sorted(p.name for p in OUT.glob("*.png")))
