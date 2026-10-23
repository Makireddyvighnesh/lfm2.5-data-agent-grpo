"""Phase-2 reward ablation summary -> results/phase2_summary.md (also printed).

Held-out: val118 pass@1 (FP8, 16k, T=1, no proxy) for the start point and each arm's final adapter.
Training: per-arm correctness / calls / timing over the 20 updates on manifests/hard50.json.
"""
import json
from pathlib import Path

DA = Path(__file__).resolve().parents[1]
ARMS = [("A", "p2_A_tool", "tool (blog: correct x (1 + 0.1*15/(15+calls)))"),
        ("B", "p2_B_context", "context (r2: correct x (1 + 0.1*ctx bonus))"),
        ("C", "p2_C_correctness", "correctness only")]


def val(name):
    p = DA / "results" / f"val118_{name}.json"
    if not p.exists():
        return None
    s = json.loads(p.read_text())
    rows = [json.loads(l) for l in open(DA / "runs" / f"val118_{name}" / "rollouts.jsonl")]
    by = {d: [r for r in rows if r["difficulty"] == d] for d in ("medium", "hard")}
    return {"pass": s["pass_at_1"], "n": s["graded"], "correct": round(s["pass_at_1"] * s["graded"]),
            "medium": f"{sum(r['correctness'] for r in by['medium']):.0f}/{len(by['medium'])}",
            "hard": f"{sum(r['correctness'] for r in by['hard']):.0f}/{len(by['hard'])}",
            "ctx_out": s["context_exhausted_rate"], "calls": s["mean_tool_calls"], "peak": s.get("mean_peak_context"),
            "secs": s["mean_seconds"]}


def train(run):
    p = DA / "runs" / run / "metrics.jsonl"
    if not p.exists():
        return None
    L = [json.loads(l) for l in open(p)]
    if not L:
        return None
    m = lambda rows, k: sum(float(r.get(k) or 0) for r in rows) / max(1, len(rows))
    first, last = L[:5], L[-5:]
    return {"updates": L[-1]["step"], "correct_first5": m(first, "correct_rate"), "correct_last5": m(last, "correct_rate"),
            "calls_first5": m(first, "tool_calls_mean"), "calls_last5": m(last, "tool_calls_mean"),
            "ctx_last5": m(last, "context_exhausted_rate"), "min_per_update": m(L, "step_s") / 60,
            "gen_min": m(L, "gen_s") / 60, "train_min": m(L, "train_s") / 60, "kl": m(L, "kl")}


def main():
    out = ["# Phase 2: reward ablation on 50 unseen hard tasks", "",
           "All arms start from r2-final (update 22) and use the same task order, prefix-chain template, FP8, 16k context, "
           "64 concurrent rollouts, 240 s training timeout and 20 updates of 8 tasks x 8 rollouts. Evaluation: val118, "
           "300 s, T=1, no proxy.", "",
           "## Held-out (val118, 118 medium+hard tasks)", "",
           "| model | pass@1 | medium | hard | out-of-ctx | tool calls | peak ctx | s/rollout |", "|---|---|---|---|---|---|---|---|"]
    rows = [("base (untrained), native", "base16k_val118"), ("r2 update 12, native", "r2u12"),
            ("r2-final (update 22), native", "r2-final-native"), ("r2-final (update 22), chain = start point", "r2-final-chain")]
    rows += [(f"arm {a}: {desc}", f"{run}-final") for a, run, desc in ARMS]
    for label, name in rows:
        v = val(name) if name != "base16k_val118" else None
        if name == "base16k_val118":
            s = json.loads((DA / "results" / "base16k_val118.json").read_text())
            out.append(f"| {label} | {s['pass_at_1']:.3f} ({round(s['pass_at_1'] * s['graded'])}/{s['graded']}) | 6/68 | 6/50 | "
                       f"{s['context_exhausted_rate']:.2f} | {s['mean_tool_calls']:.1f} | n/a | {s['mean_seconds']:.0f} |")
        elif v:
            out.append(f"| {label} | {v['pass']:.3f} ({v['correct']}/{v['n']}) | {v['medium']} | {v['hard']} | {v['ctx_out']:.2f} | "
                       f"{v['calls']:.1f} | {v['peak'] or 0:.0f} | {v['secs']:.0f} |")
        else:
            out.append(f"| {label} | (not run) | | | | | | |")
    out += ["", "## Training (hard50)", "",
            "| arm | updates | correct first5 → last5 | tool calls first5 → last5 | out-of-ctx last5 | min/update (gen + train) | KL |",
            "|---|---|---|---|---|---|---|"]
    for a, run, desc in ARMS:
        t = train(run)
        if t:
            out.append(f"| {a}: {desc} | {t['updates']} | {t['correct_first5']:.3f} → {t['correct_last5']:.3f} | "
                       f"{t['calls_first5']:.1f} → {t['calls_last5']:.1f} | {t['ctx_last5']:.3f} | "
                       f"{t['min_per_update']:.1f} ({t['gen_min']:.1f} + {t['train_min']:.1f}) | {t['kl']:.4f} |")
        else:
            out.append(f"| {a}: {desc} | (not run) | | | | | |")
    text = "\n".join(out) + "\n"
    (DA / "results" / "phase2_summary.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
