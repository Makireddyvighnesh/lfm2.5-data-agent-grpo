"""Full-screen terminal viewer for Data Agent rollouts: task prompt -> rollout i, one key to move around.

usage:
    ../SmolDataEnvs/.venv/bin/python view_rollouts.py runs/grpo_r2              # a GRPO training run
    ../SmolDataEnvs/.venv/bin/python view_rollouts.py runs/val118_r2u12         # an evaluate.py run
  options:
    --update 13 | --min-update 12     only rollouts generated at that training update (or later)
    --filter correct|wrong|ctx|alt    start with a filter (ctx = ran out of context, alt = alt-start)
    --task 0054_670                   start at the first task containing this text
    --print                           no UI: print the first selected rollout as plain text and exit

keys:
    n  ->        next rollout of this task        p  <-        previous rollout
    N  ]  Tab    next task                        P  [         previous task
    j  Down      scroll down one line             k  Up        scroll up
    Space PgDn   page down                        b  PgUp      page up
    g  Home      top                              G  End       bottom
    e            expand / collapse tool outputs   s            show / hide pi's system prompt
    f            cycle filter: all, correct, wrong, out-of-context, alt-start
    /            jump to the next task containing a text (task id or question)
    ?            help                             q            quit

The transcript comes from pi's own session log (what the model saw, including the final tool output of a
rollout that ran out of context); the capture archive is the fallback. Gold and predicted answers come
from the verifier output of the trial.
"""

import argparse
import curses
import glob
import gzip
import json
import re
import sys
import textwrap
import tomllib
from collections import OrderedDict
from pathlib import Path

DA = Path(__file__).resolve().parent
SMOL = DA.parent / "SmolDataEnvs"
TRIAL_DIRS = [SMOL / "runs" / "trials_data_agent", SMOL / "runs" / "trials"]
TASK_DIRS = [DA / "prepared/datasets/eval/tasks", SMOL / "prepared/datasets/train/tasks",
             SMOL / "prepared/datasets/test/tasks"]
FILTERS = ["all", "correct", "wrong", "ctx", "alt"]
COLLAPSED_LINES, COLLAPSED_CHARS = 15, 1500


# ----------------------------------------------------------------------------------------- data
def load_index(run):
    run = Path(run)
    path = run / "rollouts.jsonl"
    rows = []
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    else:  # evaluate.py run still in progress: per-rollout files
        rows = [json.loads(Path(f).read_text()) for f in sorted(glob.glob(str(run / "rollouts" / "*.json")))]
    out = []
    for r in rows:
        correct = r.get("correct", r.get("correctness"))
        out.append({
            "raw": r, "run": run, "task": r.get("task"), "difficulty": r.get("difficulty"),
            "update": r.get("step", r.get("intended_step")), "correct": None if correct is None else correct == 1,
            "reward": r.get("reward"), "advantage": r.get("advantage"), "stratum": r.get("root_stratum"),
            "root_class": r.get("root_class"), "weight": r.get("root_weight"), "peak": r.get("peak_context"),
            "calls": r.get("tool_calls"), "ctx": bool(r.get("context_exhausted")), "seconds": r.get("seconds"),
            "trial": r.get("trial"), "rid": r.get("rollout_id"), "error": r.get("error"),
        })
    return out


def task_info(name, cache={}):
    if name in cache:
        return cache[name]
    info = {"question": "", "gold": "", "mode": "", "instruction": ""}
    for d in TASK_DIRS:
        p = d / name
        if (p / "task.toml").exists():
            spec = tomllib.loads((p / "task.toml").read_text())
            env = spec.get("verifier", {}).get("env", {})
            info = {"question": env.get("QUESTION", ""), "gold": env.get("EXPECTED_ANSWER", ""),
                    "mode": env.get("REWARD_MODE", ""), "instruction": (p / "instruction.md").read_text()}
            break
    cache[name] = info
    return info


def input_sizes():
    p = DA / "analysis" / "task_file_sizes.json"
    return {t["name"]: t["bytes"] for t in json.loads(p.read_text())} if p.exists() else {}


def trial_dir(trial):
    for d in TRIAL_DIRS:
        if trial and (d / trial).exists():
            return d / trial
    return None


def verdict(trial):
    d = trial_dir(trial)
    t = (d / "verifier" / "test-stdout.txt") if d else None
    text = t.read_text() if t and t.exists() else ""
    if "no answer at" in text:
        return "(no answer written)"
    m = re.search(r"pred=(['\"])(.*?)\1 (?:reward|correctness)=", text, re.S)
    return m.group(2) if m else ("(no verifier output)" if not text else text.strip()[-120:])


def _text(content):
    if isinstance(content, str):
        return content
    return "".join(c.get("text", "") for c in content or [] if isinstance(c, dict) and c.get("type") == "text")


def transcript(row, cache={}):
    """[{"role": system|user|assistant|tool, "text": str, "calls": [(name, args)], "tool": name}]"""
    key = (str(row["run"]), row["rid"], row["trial"])
    if key in cache:
        return cache[key]
    msgs = []
    d = trial_dir(row["trial"])
    sessions = sorted((d / "agent" / "pi" / "sessions").glob("*.jsonl")) if d else []
    if sessions:
        for line in sessions[-1].read_text().splitlines():
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            m = rec.get("message") or {}
            if rec.get("type") != "message":
                continue
            role, content = m.get("role"), m.get("content")
            if role == "assistant":
                calls = [(c.get("name"), c.get("arguments")) for c in content or [] if c.get("type") == "toolCall"]
                msgs.append({"role": "assistant", "text": _text(content), "calls": calls})
            elif role == "toolResult":
                msgs.append({"role": "tool", "text": _text(content), "tool": m.get("toolName")})
            elif role in ("system", "user"):
                msgs.append({"role": role, "text": _text(content)})
    if not msgs:  # fallback: capture archive (training) or the eval row's trajectory
        raw = row["raw"]
        conv = (raw.get("trajectory") or {}).get("messages")
        if conv is None and row["rid"]:
            gz = Path(row["run"]) / "rollouts" / f"{row['rid']}.json.gz"
            if gz.exists():
                turns = json.load(gzip.open(gz))["result"]["turns"]
                conv = list(turns[-1]["request_messages"]) + [{"role": "assistant", "content": turns[-1].get("text"),
                                                               "tool_calls": turns[-1].get("tool_calls")}]
        for m in conv or []:
            role = m.get("role")
            if role == "assistant":
                calls = []
                for c in m.get("tool_calls") or []:
                    fn = c.get("function", c)
                    args = fn.get("arguments")
                    try:
                        args = json.loads(args) if isinstance(args, str) else args
                    except ValueError:
                        pass
                    calls.append((fn.get("name"), args))
                msgs.append({"role": "assistant", "text": _text(m.get("content")) or "", "calls": calls})
            elif role == "tool":
                msgs.append({"role": "tool", "text": _text(m.get("content")), "tool": m.get("name")})
            else:
                msgs.append({"role": role, "text": _text(m.get("content"))})
    cache[key] = msgs
    return msgs


def matches(row, flt):
    return {"all": True, "correct": row["correct"] is True, "wrong": row["correct"] is False,
            "ctx": row["ctx"], "alt": row["stratum"] == "alt"}[flt]


# ----------------------------------------------------------------------------------------- render
def format_call(name, args):
    if not isinstance(args, dict):
        return [f"{name}({args})"]
    if name == "bash":
        return [f"bash $ {line}" if i == 0 else f"       {line}" for i, line in enumerate(str(args.get("command", "")).splitlines() or [""])]
    if name == "read":
        extra = ", ".join(f"{k}={args[k]}" for k in ("offset", "limit") if k in args)
        return [f"read {args.get('path')}" + (f"  ({extra})" if extra else "  (WHOLE FILE)")]
    if name == "write":
        return [f"write {args.get('path')} <- {json.dumps(args.get('content'))[:300]}"]
    return [f"{name} {json.dumps(args)[:600]}"]


def build_lines(row, sizes, width, expand, show_system):
    """Body as [(text, style)] wrapped to width. Styles: hdr ok bad asst call tool user dim."""
    out = []

    def add(text, style="", indent=0):
        for raw in str(text).splitlines() or [""]:
            raw = raw.expandtabs(4)
            pieces = textwrap.wrap(raw, max(20, width - indent), replace_whitespace=False, drop_whitespace=False) or [""]
            for p in pieces:
                out.append((" " * indent + p, style))

    msgs = transcript(row)
    turn = 0
    for m in msgs:
        if m["role"] == "system":
            if show_system:
                add("── SYSTEM PROMPT (pi) ──", "user")
                add(m["text"], "dim", 2)
                add("")
            continue
        if m["role"] == "user":
            add("── TASK PROMPT ──", "user")
            add(m["text"], "", 2)
            add("")
        elif m["role"] == "assistant" and not m["calls"] and m["text"].strip().startswith("Step budget exhausted"):
            # Not the model: OpenEnv's capture proxy answers with this zero-token reply when the next request
            # would not fit the context window (or the model-call budget is used up), which ends pi's loop.
            add("■ HARNESS (not the model): capture proxy stopped the rollout — the next request would not fit "
                "the 16k context window or the 17-call budget was used up", "bad")
        elif m["role"] == "assistant":
            turn += 1
            add(f"▶ MODEL CALL {turn}" + ("" if m["text"].strip() else "   (no text — went straight to the tool call)"
                                          if m["calls"] else ""), "asst")
            if m["text"].strip():
                add(m["text"].strip(), "asst", 2)
            for name, args in m["calls"]:
                for line in format_call(name, args):
                    add(line, "call", 2)
            if not m["calls"]:
                add("(no tool call — final message)", "dim", 2)
        elif m["role"] == "tool":
            text = m["text"] or ""
            lines = text.splitlines()
            add(f"◀ {m.get('tool') or 'tool'} output: {len(text):,} chars, {len(lines):,} lines", "dim", 2)
            if not expand and (len(lines) > COLLAPSED_LINES or len(text) > COLLAPSED_CHARS):
                shown = "\n".join(lines[:COLLAPSED_LINES])[:COLLAPSED_CHARS]
                add(shown, "tool", 4)
                add(f"… [{len(lines) - shown.count(chr(10)) - 1:,} more lines, {len(text) - len(shown):,} more chars — press e]", "dim", 4)
            else:
                add(text, "tool", 4)
        add("")
    if row["ctx"]:
        add("■ Rollout ended: the next request would not fit in the 16k context window.", "bad")
    if row.get("error"):
        add(f"■ infrastructure error: {row['error']}", "bad")
    return out


def header_lines(row, ti, ri, ntasks, nroll, flt, sizes):
    info = task_info(row["task"])
    size = sizes.get(row["task"])
    size_s = "" if size is None else (f"{size/1e6:.1f} MB" if size >= 1e6 else f"{size/1e3:.0f} KB")
    status = "✓ CORRECT" if row["correct"] else ("✗ WRONG" if row["correct"] is False else "? ungraded")
    upd = f"update {row['update']}  " if row["update"] is not None else ""
    adv = f"adv {row['advantage']:+.2f}  " if isinstance(row["advantage"], (int, float)) else ""
    rew = f"reward {row['reward']:.2f}  " if isinstance(row["reward"], (int, float)) else ""
    root = (f"root {row['stratum']}:{row['root_class']} W={row['weight']:.2f}  "
            if row["stratum"] and isinstance(row["weight"], (int, float)) else "")
    peak = f"peak ctx {row['peak']:,}  " if isinstance(row["peak"], int) else ""
    calls = f"tool calls {row['calls']}  " if row["calls"] is not None else ""
    secs = f"{row['seconds']:.0f}s" if isinstance(row["seconds"], (int, float)) else ""
    return [
        (f"{Path(row['run']).name}  task {ti + 1}/{ntasks}: {row['task']}  ({row['difficulty']}, input {size_s or '?'})"
         f"   rollout {ri + 1}/{nroll}   filter: {flt}", "hdr"),
        (f"{status}  {upd}{rew}{adv}{root}{calls}{peak}{'OUT OF CONTEXT  ' if row['ctx'] else ''}{secs}",
         "ok" if row["correct"] else "bad"),
        (f"Q: {info['question']}", ""),
        (f"gold: {info['gold']!r}   ({info['mode']})      model answered: {verdict(row['trial'])!r}", "user"),
    ]


# ----------------------------------------------------------------------------------------- app
class Viewer:
    def __init__(self, rows, flt, start_task):
        self.all = rows
        self.sizes = input_sizes()
        self.flt = flt
        self.expand = False
        self.system = False
        self.regroup()
        self.ti = 0
        if start_task:
            self.jump(start_task, from_start=True)

    def regroup(self):
        groups = OrderedDict()
        for r in sorted(self.all, key=lambda r: (r["update"] if r["update"] is not None else 0)):
            if matches(r, self.flt):
                groups.setdefault(r["task"], []).append(r)
        self.tasks = list(groups.items())
        self.ti, self.ri, self.top = 0, 0, 0

    def current(self):
        if not self.tasks:
            return None
        return self.tasks[self.ti][1][self.ri]

    def jump(self, text, from_start=False):
        n = len(self.tasks)
        order = range(n) if from_start else [(self.ti + k) % n for k in range(1, n + 1)]
        for i in order:
            name, _ = self.tasks[i]
            if text.lower() in name.lower() or text.lower() in task_info(name)["question"].lower():
                self.ti, self.ri, self.top = i, 0, 0
                return True
        return False

    def run(self, scr):
        curses.curs_set(0)
        curses.use_default_colors()
        palette = {"hdr": (curses.COLOR_CYAN, curses.A_BOLD), "ok": (curses.COLOR_GREEN, curses.A_BOLD),
                   "bad": (curses.COLOR_RED, curses.A_BOLD), "asst": (curses.COLOR_YELLOW, curses.A_BOLD),
                   "call": (curses.COLOR_MAGENTA, 0), "tool": (-1, 0), "user": (curses.COLOR_BLUE, curses.A_BOLD),
                   "dim": (-1, curses.A_DIM), "": (-1, 0)}
        attrs = {}
        for i, (k, (fg, extra)) in enumerate(palette.items(), start=1):
            curses.init_pair(i, fg, -1)
            attrs[k] = curses.color_pair(i) | extra
        message = ""
        while True:
            scr.erase()
            h, w = scr.getmaxyx()
            row = self.current()
            if row is None:
                scr.addnstr(0, 0, f"No rollouts match filter '{self.flt}'. Press f to change, q to quit.", w - 1)
                key = scr.getch()
                if key in (ord("q"), 27):
                    return
                if key == ord("f"):
                    self.flt = FILTERS[(FILTERS.index(self.flt) + 1) % len(FILTERS)]
                    self.regroup()
                continue
            name, rollouts = self.tasks[self.ti]
            head = header_lines(row, self.ti, self.ri, len(self.tasks), len(rollouts), self.flt, self.sizes)
            y = 0
            for text, style in head:
                for piece in textwrap.wrap(text, w - 1) or [""]:
                    if y < h - 2:
                        scr.addnstr(y, 0, piece, w - 1, attrs[style])
                        y += 1
            scr.hline(y, 0, curses.ACS_HLINE, w)
            y += 1
            body = build_lines(row, self.sizes, w - 1, self.expand, self.system)
            room = h - y - 1
            self.top = max(0, min(self.top, max(0, len(body) - room)))
            for i, (text, style) in enumerate(body[self.top:self.top + room]):
                try:
                    scr.addnstr(y + i, 0, text, w - 1, attrs.get(style, 0))
                except curses.error:
                    pass
            pct = 100 if len(body) <= room else int(100 * (self.top + room) / len(body))
            foot = (message or "n/p rollout  N/P task  j/k ↑↓ scroll  space/b page  g/G top/end  e expand  "
                    "s system  f filter  / find  ? help  q quit") + f"   [{pct}%]"
            scr.addnstr(h - 1, 0, foot.ljust(w - 1), w - 1, curses.A_REVERSE)
            message = ""
            key = scr.getch()
            if key in (ord("q"), 27):
                return
            if key in (ord("n"), curses.KEY_RIGHT):
                if self.ri + 1 < len(rollouts):
                    self.ri, self.top = self.ri + 1, 0
                else:
                    message = "last rollout of this task (N = next task)"
            elif key in (ord("p"), curses.KEY_LEFT):
                if self.ri > 0:
                    self.ri, self.top = self.ri - 1, 0
            elif key in (ord("N"), ord("]"), 9):
                self.ti, self.ri, self.top = (self.ti + 1) % len(self.tasks), 0, 0
            elif key in (ord("P"), ord("[")):
                self.ti, self.ri, self.top = (self.ti - 1) % len(self.tasks), 0, 0
            elif key in (ord("j"), curses.KEY_DOWN):
                self.top += 1
            elif key in (ord("k"), curses.KEY_UP):
                self.top = max(0, self.top - 1)
            elif key in (ord(" "), curses.KEY_NPAGE):
                self.top += max(1, room - 2)
            elif key in (ord("b"), curses.KEY_PPAGE):
                self.top = max(0, self.top - max(1, room - 2))
            elif key in (ord("g"), curses.KEY_HOME):
                self.top = 0
            elif key in (ord("G"), curses.KEY_END):
                self.top = len(body)
            elif key == ord("e"):
                self.expand = not self.expand
            elif key == ord("s"):
                self.system = not self.system
                self.top = 0
            elif key == ord("f"):
                self.flt = FILTERS[(FILTERS.index(self.flt) + 1) % len(FILTERS)]
                self.regroup()
                message = f"filter: {self.flt}  ({len(self.tasks)} tasks)"
            elif key == ord("/"):
                curses.echo()
                curses.curs_set(1)
                scr.addnstr(h - 1, 0, "find task / question: ".ljust(w - 1), w - 1, curses.A_REVERSE)
                text = scr.getstr(h - 1, 22, 60).decode(errors="ignore").strip()
                curses.noecho()
                curses.curs_set(0)
                if text and not self.jump(text):
                    message = f"no task contains {text!r}"
            elif key == ord("?"):
                scr.erase()
                for i, line in enumerate(__doc__.splitlines()[:h - 1]):
                    scr.addnstr(i, 0, line, w - 1)
                scr.getch()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", help="run directory, e.g. runs/grpo_r2 or runs/val118_r2u12")
    p.add_argument("--update", type=int)
    p.add_argument("--min-update", type=int)
    p.add_argument("--filter", choices=FILTERS, default="all")
    p.add_argument("--task")
    p.add_argument("--print", action="store_true")
    a = p.parse_args()
    rows = load_index(a.run)
    if a.update is not None:
        rows = [r for r in rows if r["update"] == a.update]
    if a.min_update is not None:
        rows = [r for r in rows if (r["update"] or 0) >= a.min_update]
    if not rows:
        sys.exit("no rollouts found (check the run directory and update filters)")
    viewer = Viewer(rows, a.filter, a.task)
    if a.print:
        row = viewer.current()
        if row is None:
            sys.exit(f"no rollouts match filter {a.filter}")
        name, rollouts = viewer.tasks[viewer.ti]
        for text, _ in header_lines(row, viewer.ti, viewer.ri, len(viewer.tasks), len(rollouts), viewer.flt, viewer.sizes):
            print(text)
        print("-" * 100)
        for text, _ in build_lines(row, viewer.sizes, 120, False, False):
            print(text)
        return
    curses.wrapper(viewer.run)


if __name__ == "__main__":
    main()
