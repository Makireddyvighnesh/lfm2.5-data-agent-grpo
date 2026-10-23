"""Interactive terminal viewer for Pi rollouts (evaluation or training), grouped by prompt.

    .venv/bin/python pirl/view_rollouts.py runs/eval_base_heldout_mh/rollouts.jsonl
    .venv/bin/python pirl/view_rollouts.py runs/abl_correctness/rollouts.jsonl --filter wrong
    .venv/bin/python pirl/view_rollouts.py runs/ablate_eval/base/rollouts.jsonl --task 0000_340 --full

Keys (type, then Enter):
    n / Enter   next rollout of this prompt        p   previous rollout
    N / ]       next prompt                        P / [   previous prompt
    g <k>       go to prompt number k (1-based)    / <text>  next prompt whose task/question contains text
    f <kind>    filter: all|correct|wrong|noanswer|ctx|timeout   (applies to prompts and rollouts)
    t           toggle full tool outputs (default: first 25 lines / 2,500 chars)
    s           toggle the system prompt            q   quit

Sources: an eval row carries its trajectory; a training row only names its `trial`, so the transcript is
read from runs/trials/<trial>/agent/pi/sessions/*.jsonl (preferred for both: it has full tool outputs).
"""

import argparse
import json
import sys
import tomllib
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRIALS = ROOT / "runs" / "trials"
TASKS = ROOT / "prepared" / "datasets"

try:
    from rich.console import Console
    from rich.markup import escape
    from rich.panel import Panel
    from rich.rule import Rule
    from rich.syntax import Syntax

    console = Console(highlight=False)
except ImportError:  # plain fallback
    console = None


# ---------------------------------------------------------------- loading
def task_info(name):
    for split in ("test", "train"):
        d = TASKS / split / "tasks" / name
        if d.exists():
            spec = tomllib.loads((d / "task.toml").read_text())
            return {"split": split, "instruction": (d / "instruction.md").read_text(),
                    "difficulty": spec["metadata"].get("difficulty_tier"),
                    "gold": spec["metadata"].get("gold_answer"), "reward_mode": spec["verifier"]["env"].get("REWARD_MODE")}
    return {"split": "?", "instruction": "", "gold": None, "reward_mode": None, "difficulty": None}


def question(instruction):
    if "Question:" in instruction:
        return instruction.split("Question:", 1)[1].split("\n\n", 1)[0].strip()
    return instruction.strip()[:300]


def pi_session(trial):
    sessions = sorted((TRIALS / str(trial) / "agent" / "pi" / "sessions").glob("*.jsonl"))
    if not sessions:
        return None
    msgs = []
    for line in sessions[-1].read_text().splitlines():
        try:
            e = json.loads(line)
        except ValueError:
            continue
        m = e.get("message")
        if e.get("type") == "message" and m:
            msgs.append(m)
    return msgs


def from_pi(msgs):
    """Normalize pi session messages into [{role, text, tool_calls, name, error, usage}]."""
    out = []
    for m in msgs:
        content = m.get("content")
        items = content if isinstance(content, list) else [{"type": "text", "text": content or ""}]
        text = "".join(x.get("text", "") for x in items if isinstance(x, dict) and x.get("type") == "text")
        calls = [{"name": x.get("name"), "args": x.get("arguments")} for x in items
                 if isinstance(x, dict) and x.get("type") == "toolCall"]
        role = {"toolResult": "tool"}.get(m.get("role"), m.get("role"))
        if role == "system":
            secs = m.get("sections") or {}
            text = "\n\n".join(str(v) for v in secs.values()) if isinstance(secs, dict) else text
        out.append({"role": role, "text": text, "tool_calls": calls, "name": m.get("toolName"),
                    "error": m.get("isError"), "usage": m.get("usage")})
    return out


def from_trajectory(traj):
    out = []
    for m in traj.get("messages") or []:
        content = m.get("content")
        text = content if isinstance(content, str) else json.dumps(content) if content else ""
        calls = []
        for tc in m.get("tool_calls") or []:
            f = tc.get("function", tc)
            args = f.get("arguments")
            try:
                args = json.loads(args) if isinstance(args, str) else args
            except ValueError:
                pass
            calls.append({"name": f.get("name"), "args": args})
        out.append({"role": m.get("role"), "text": text, "tool_calls": calls, "name": m.get("name"),
                    "error": None, "usage": None})
    return out


def load(path, task_filter=None):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    prompts = OrderedDict()
    for r in rows:
        if task_filter and task_filter not in r.get("task", ""):
            continue
        key = r.get("prompt_id") or r.get("task")
        prompts.setdefault(key, []).append(r)
    return prompts


def outcome(r):
    correct = r.get("correctness", r.get("correct"))
    if correct is None and r.get("reward") is None and (r.get("error") or r.get("capture_error")):
        return "error"
    if correct == 1:
        return "correct"
    if r.get("submitted"):
        return "wrong"
    if r.get("context_exhausted"):
        return "ctx"
    if (r.get("seconds") or 0) >= 300:
        return "timeout"
    return "noanswer"


def matches(r, kind):
    o = outcome(r)
    return kind == "all" or o == kind or (kind == "noanswer" and o in ("ctx", "timeout", "noanswer"))


# ---------------------------------------------------------------- rendering
COLORS = {"correct": "green", "wrong": "red", "ctx": "magenta", "timeout": "yellow", "noanswer": "yellow", "error": "red"}


def clip(text, full, lines=25, chars=2500):
    if full or (len(text) <= chars and text.count("\n") < lines):
        return text
    head = "\n".join(text.splitlines()[:lines])[:chars]
    return head + f"\n... [{len(text) - len(head):,} more chars hidden; press t for full output]"


def show(prompt_idx, n_prompts, rollouts, ridx, opts):
    r = rollouts[ridx]
    info = task_info(r.get("task", ""))
    instruction = r.get("instruction") or info["instruction"]
    msgs = None
    if r.get("trial"):
        pi = pi_session(r["trial"])
        msgs = from_pi(pi) if pi else None
    if msgs is None and r.get("trajectory"):
        msgs = from_trajectory(r["trajectory"])
    o = outcome(r)
    out = console.print if console else print
    if console:
        console.clear()
    head = (f"prompt {prompt_idx + 1}/{n_prompts}  rollout {ridx + 1}/{len(rollouts)}  "
            f"[{COLORS.get(o, 'white')}]{o.upper()}[/]   task={r.get('task')}  ({r.get('difficulty') or info['difficulty']}, {info['split']})")
    stats = (f"reward={r.get('reward')}  tool_calls={r.get('tool_calls')}  model_calls={r.get('model_calls', r.get('n_turns'))}  "
             f"gen_tokens={r.get('generated_tokens')}  final_ctx={r.get('final_context', r.get('prompt_tokens_last'))}  "
             f"seconds={r.get('seconds')}  submitted={r.get('submitted')}  ctx_exhausted={r.get('context_exhausted')}")
    ids = f"prompt_id={r.get('prompt_id', '-')}  rollout_id={r.get('rollout_id', '-')}  trial={r.get('trial')}  step={r.get('step', '-')}"
    codes = {"correct": "OK", "wrong": "WRONG", "ctx": "CTX", "timeout": "TIMEOUT", "noanswer": "NOANS", "error": "ERR"}
    summary = "  ".join(f"{i + 1}:{codes[outcome(x)]}" + (f"(s{x['step']})" if x.get("step") else "")
                        for i, x in enumerate(rollouts))
    if console:
        console.print(Panel(f"{head}\n{escape(stats)}\n[dim]{escape(ids)}[/]\nrollouts of this prompt: {summary}",
                            title="rollout", border_style=COLORS.get(o, "white")))
        console.print(Panel(escape(question(instruction)) + f"\n\n[bold]gold answer:[/] {escape(str(info['gold']))}"
                            f"  [dim](grader mode {info['reward_mode']})[/]", title="question", border_style="cyan"))
    else:
        print(head, stats, ids, "QUESTION:", question(instruction), "GOLD:", info["gold"], sep="\n")
    if not msgs:
        out("(no transcript found: trial directory missing and no embedded trajectory)")
        return
    call_no = 0
    for m in msgs:
        role = m["role"]
        if role == "system" and not opts["system"]:
            out("[dim]system prompt hidden (press s)[/]" if console else "system prompt hidden (s)")
            continue
        if role == "user" and console:
            console.print(Rule("user", style="cyan"))
            console.print(escape(clip(m["text"], opts["full"], 12, 1200)))
            continue
        if role == "assistant":
            call_no += 1
            u = m.get("usage") or {}
            tag = f"assistant #{call_no}" + (f"  (prompt {u.get('input')} tok, generated {u.get('output')} tok)" if u else "")
            if console:
                console.print(Rule(tag, style="blue"))
                if m["text"].strip():
                    console.print(escape(m["text"].strip()))
                for c in m["tool_calls"]:
                    args = c["args"] if isinstance(c["args"], dict) else {"raw": c["args"]}
                    if c["name"] == "bash" and "command" in args:
                        console.print(f"[bold blue]→ bash[/]")
                        console.print(Syntax(str(args["command"]), "bash", word_wrap=True, background_color="default"))
                    elif c["name"] == "write" and "content" in args:
                        console.print(f"[bold blue]→ write[/] {escape(str(args.get('path')))}")
                        console.print(Syntax(clip(str(args["content"]), opts["full"]), "python", word_wrap=True,
                                             background_color="default"))
                    else:
                        console.print(f"[bold blue]→ {escape(str(c['name']))}[/] {escape(json.dumps(args)[:500])}")
            else:
                print(f"--- {tag}\n{m['text']}\n" + "\n".join(f"-> {c['name']} {c['args']}" for c in m["tool_calls"]))
            continue
        if role == "tool":
            text = m["text"] or ""
            label = f"tool result: {m.get('name') or ''}  ({len(text):,} chars)" + ("  ERROR" if m.get("error") else "")
            if console:
                console.print(Rule(label, style="red" if m.get("error") else "grey50"))
                console.print(f"[grey70]{escape(clip(text, opts['full']))}[/]")
            else:
                print(f"--- {label}\n{clip(text, opts['full'])}")


# ---------------------------------------------------------------- main loop
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("jsonl", help="rollouts.jsonl from an eval (runs/eval_*/) or a training run (runs/abl_*/)")
    ap.add_argument("--filter", default="all", help="all|correct|wrong|noanswer|ctx|timeout")
    ap.add_argument("--task", help="only tasks whose name contains this")
    ap.add_argument("--full", action="store_true", help="show full tool outputs")
    a = ap.parse_args()

    allp = load(a.jsonl, a.task)
    opts = {"full": a.full, "system": False, "filter": a.filter}

    def visible():
        return [(k, [r for r in v if matches(r, opts["filter"])]) for k, v in allp.items()
                if any(matches(r, opts["filter"]) for r in v)]

    prompts = visible()
    if not prompts:
        sys.exit(f"no rollouts match filter={opts['filter']}")
    pi, ri = 0, 0
    while True:
        key, rolls = prompts[pi]
        ri = min(ri, len(rolls) - 1)
        show(pi, len(prompts), rolls, ri, opts)
        try:
            cmd = input(f"\n[{opts['filter']}] n/p rollout · N/P prompt · g <k> · / <text> · f <kind> · t full · s system · q > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if cmd in ("q", "quit"):
            break
        elif cmd in ("", "n"):
            if ri + 1 < len(rolls):
                ri += 1
            else:
                pi, ri = (pi + 1) % len(prompts), 0
        elif cmd == "p":
            if ri > 0:
                ri -= 1
            else:
                pi = (pi - 1) % len(prompts)
                ri = len(prompts[pi][1]) - 1
        elif cmd in ("N", "]"):
            pi, ri = (pi + 1) % len(prompts), 0
        elif cmd in ("P", "["):
            pi, ri = (pi - 1) % len(prompts), 0
        elif cmd.startswith("g "):
            try:
                pi, ri = max(0, min(len(prompts) - 1, int(cmd[2:]) - 1)), 0
            except ValueError:
                pass
        elif cmd.startswith("/"):
            text = cmd[1:].strip().lower()
            for step in range(1, len(prompts) + 1):
                k = (pi + step) % len(prompts)
                r0 = prompts[k][1][0]
                hay = (r0.get("task", "") + " " + question(r0.get("instruction") or task_info(r0.get("task", ""))["instruction"])).lower()
                if text in hay:
                    pi, ri = k, 0
                    break
        elif cmd.startswith("f "):
            opts["filter"] = cmd[2:].strip() or "all"
            new = visible()
            if new:
                prompts, pi, ri = new, 0, 0
            else:
                opts["filter"] = "all"
        elif cmd == "t":
            opts["full"] = not opts["full"]
        elif cmd == "s":
            opts["system"] = not opts["system"]


if __name__ == "__main__":
    main()
