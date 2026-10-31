"""README plots for p12: training stability, training behaviour, test250 by epoch -> plots/*.png.

Usage: python analysis/plot_p12.py   (reads runs/p12_mh4_r32/metrics.jsonl and results/test250_mh24k_*_p12e*_r1.json)
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUN, OUT = "runs/p12_mh4_r32", "plots"
EPOCH_ENDS = [71, 127, 183]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, RAW = "#2a78d6", "#9ec5f4"
HARNESS_COLORS = {"pi": "#2a78d6", "mini-swe-agent": "#eb6834", "codex": "#1baf7a", "prime-agent": "#eda100"}
EPOCH_COLORS = ["#9ec5f4", "#5598e7", "#1c5cab"]  # sequential blue: e1 -> e3
WIN = 10

plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                     "axes.titlesize": 11, "axes.titleweight": "bold", "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6, "lines.linewidth": 2})


def load():
    rows = {}
    for line in open(f"{RUN}/metrics.jsonl"):
        r = json.loads(line)
        if "step" in r:
            rows[r["step"]] = r  # last write wins after a resume
    return [rows[s] for s in sorted(rows)]


def rolling(xs, ys, ws=None, win=WIN):
    """Trailing mean over the last `win` steps, weighted by ws (e.g. rollout counts); skips missing values."""
    out = []
    for i in range(len(xs)):
        num = den = 0.0
        for j in range(max(0, i - win + 1), i + 1):
            if ys[j] is None:
                continue
            w = 1.0 if ws is None else (ws[j] or 0)
            num += w * ys[j]; den += w
        out.append(num / den if den else None)
    return out


def epochs(ax):
    ax.set_xlim(0, EPOCH_ENDS[-1] + 3)
    for e in EPOCH_ENDS[:-1]:
        ax.axvline(e + 0.5, color=INK2, lw=0.8, ls=(0, (3, 3)))


def epoch_labels(ax):
    top = ax.get_ylim()[1]
    for a, b, name in [(1, 71, "e1"), (72, 127, "e2"), (128, 183, "e3")]:
        ax.text((a + b) / 2, top, name, ha="center", va="bottom", color=INK2, fontsize=8)


def panel(ax, rows, key, title, fmt=None, raw=True, ylim=None):
    xs = [r["step"] for r in rows]; ys = [r.get(key) for r in rows]
    if raw:
        ax.plot(xs, [y if y is not None else float("nan") for y in ys], color=RAW, lw=1)
    ax.plot(xs, [y if y is not None else float("nan") for y in rolling(xs, ys)], color=BLUE)
    ax.set_title(title, loc="left", pad=14); epochs(ax)
    if ylim: ax.set_ylim(*ylim)
    if fmt: ax.yaxis.set_major_formatter(fmt)
    epoch_labels(ax)


def stability(rows):
    fig, axs = plt.subplots(2, 3, figsize=(13, 6.2), constrained_layout=True)
    panel(axs[0, 0], rows, "grad_norm", "Gradient norm", ylim=(0, None))
    panel(axs[0, 1], rows, "kl", "KL to sampler policy", ylim=(0, None))
    panel(axs[0, 2], rows, "entropy", "Token entropy", ylim=(0, None))
    panel(axs[1, 0], rows, "ratio_mean", "Importance ratio (mean)", ylim=(0.99, 1.01))
    panel(axs[1, 1], rows, "clip_frac", "Outside PPO clip range (diagnostic)", ylim=(0, None))
    panel(axs[1, 2], rows, "lr", "Learning rate", raw=False, ylim=(0, None))
    for ax in axs[1]: ax.set_xlabel("update")
    fig.suptitle("p12 training stability (light: per update, dark: 10-update mean)", x=0.01, ha="left", fontsize=12)
    fig.savefig(f"{OUT}/p12_stability.png", dpi=130); plt.close(fig)


def behaviour(rows):
    fig, axs = plt.subplots(2, 3, figsize=(13, 6.2), constrained_layout=True)
    xs = [r["step"] for r in rows]
    ax = axs[0, 0]
    for h, c in HARNESS_COLORS.items():
        ys = rolling(xs, [r.get(f"correct_rate_{h}") for r in rows], [r.get(f"rollouts_{h}") for r in rows], win=15)
        ax.plot(xs, [y if y is not None else float("nan") for y in ys], color=c, label=h)
    ax.set_title("Training accuracy by harness (15-update mean)", loc="left", pad=14); ax.set_ylim(0, 1); epochs(ax); epoch_labels(ax)
    ax.legend(loc="lower right", frameon=False, fontsize=8, ncol=2)
    panel(axs[0, 1], rows, "reward_mean", "Mean reward")
    panel(axs[0, 2], rows, "unknown_tool_calls_mean", "Unknown-tool calls per rollout", ylim=(0, None))
    panel(axs[1, 0], rows, "capped_tool_outputs_pi", "Pi: capped outputs per rollout (from e2)", ylim=(0, None))
    panel(axs[1, 1], rows, "own_context_mean", "Own context, tokens (from e2)", ylim=(0, None))
    panel(axs[1, 2], rows, "tool_calls_mean", "Tool calls per rollout", ylim=(0, None))
    for ax in axs[1]: ax.set_xlabel("update")
    fig.suptitle("p12 training behaviour (dashed lines: epoch boundaries)", x=0.01, ha="left", fontsize=12)
    fig.savefig(f"{OUT}/p12_behaviour.png", dpi=130); plt.close(fig)


def test250():
    hs = ["opencode", "claude-code", "pi", "mini-swe-agent", "codex", "prime-agent"]
    names = ["OpenCode*", "Claude Code*", "Pi", "mini-swe-agent", "Codex", "Prime Agent"]
    blog = {"opencode": 49.6, "claude-code": 48.8, "mini-swe-agent": 64.8, "codex": 53.6}
    fig, ax = plt.subplots(figsize=(11, 4.4), constrained_layout=True)
    w = 0.26
    for k, ep in enumerate(["p12e1", "p12e2", "p12e3"]):
        vals = [100 * json.load(open(f"results/test250_mh24k_{h}_{ep}_r1.json"))["pass_at_1"] for h in hs]
        xs = [i + (k - 1) * (w + 0.02) for i in range(len(hs))]
        ax.bar(xs, vals, w, color=EPOCH_COLORS[k], label=f"epoch {k + 1}", edgecolor=SURFACE, linewidth=1)
        if k == 2:
            for x, v in zip(xs, vals): ax.text(x, v + 0.8, f"{v:.1f}", ha="center", fontsize=8, color=INK)
    for i, h in enumerate(hs):
        if h in blog:
            ax.plot([i - 0.45, i + 0.45], [blog[h]] * 2, color=INK, lw=1.5, ls=(0, (4, 2)))
    ax.plot([], [], color=INK, lw=1.5, ls=(0, (4, 2)), label="blog (128k context)")
    ax.set_xticks(range(len(hs)), names); ax.set_ylim(35, 70); ax.set_ylabel("pass@1 (%)")
    ax.grid(axis="x", visible=False); ax.legend(frameon=False, ncol=4, loc="upper left")
    ax.set_title("test250 pass@1 by epoch (axis starts at 35%; 1 attempt per task, about ±3 pts; * = never trained on)", loc="left")
    fig.savefig(f"{OUT}/p12_test250.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    rows = load()
    stability(rows); behaviour(rows); test250()
    print("wrote", sorted(os.listdir(OUT)))
