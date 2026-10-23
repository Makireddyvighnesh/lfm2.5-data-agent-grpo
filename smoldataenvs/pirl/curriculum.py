"""Prompt selection for GRPO on a fixed task pool: regret priority with a floor, plus adaptive group size.

GRPO gets no gradient from a group whose rollouts all score the same. Following Frontier Learning
(arXiv 2609.35426), a prompt's learning potential is its regret from the last visit:

    regret = 1[s >= 1] - s/n        (0 if never solved or always solved, high if solved only sometimes)

Prompts are never dropped: every seen prompt keeps a floor weight plus a staleness bonus, so prompts
that were impossible can be retried once transfer from other prompts makes them solvable. Prompts that
scored 0/n last time get more rollouts (`hard_n`), raising the chance of the one success that turns an
all-fail group into a learning signal.

State is rebuilt from the run's rollouts.jsonl, so resuming needs nothing else.
"""

import random


class PromptSampler:
    def __init__(self, order, *, base_n=4, hard_n=8, new_frac=0.5, floor=0.05, stale_bonus=0.1,
                 stale_horizon=100, seed=0):
        self.order = list(order)
        self.base_n, self.hard_n = base_n, hard_n
        self.new_frac, self.floor = new_frac, floor
        self.stale_bonus, self.stale_horizon = stale_bonus, stale_horizon
        self.stats = {}  # task index -> {"s": successes, "n": scored rollouts, "step": last visit}
        self.rng = random.Random(seed)

    # ------------------------------------------------------------- state
    def update(self, idx, successes, n, step):
        if n > 0:
            self.stats[idx] = {"s": successes, "n": n, "step": step}

    def rebuild(self, rows, name_to_idx):
        """Replay a run's rollouts.jsonl rows (dicts with task, step, reward, correct)."""
        groups = {}
        for r in rows:
            if r.get("reward") is None:
                continue
            key = (r["step"], name_to_idx[r["task"]])
            g = groups.setdefault(key, [0, 0])
            g[0] += int(r.get("correct") == 1)
            g[1] += 1
        for (step, idx), (s, n) in sorted(groups.items()):
            self.update(idx, s, n, step)

    def state_dict(self):
        return {"stats": {str(i): dict(v) for i, v in self.stats.items()},
                "rng_state": self.rng.getstate()}

    def load_state_dict(self, state):
        self.stats = {int(i): dict(v) for i, v in state["stats"].items()}

        def tuples(value):
            return tuple(tuples(v) for v in value) if isinstance(value, (list, tuple)) else value

        self.rng.setstate(tuples(state["rng_state"]))

    def regret(self, idx):
        st = self.stats.get(idx)
        if not st:
            return 0.0
        return float(st["s"] >= 1) - st["s"] / st["n"]

    def weight(self, idx, step):
        st = self.stats[idx]
        staleness = min(1.0, (step - st["step"]) / self.stale_horizon)
        return self.regret(idx) + self.floor + self.stale_bonus * staleness

    def n_rollouts(self, idx):
        st = self.stats.get(idx)
        return self.hard_n if st and st["s"] == 0 else self.base_n

    # ------------------------------------------------------------- selection
    def next_batch(self, size, step):
        """`size` distinct prompts: unseen ones in fixed order (share `new_frac`), the rest by priority."""
        unseen = [i for i in self.order if i not in self.stats]
        seen = [i for i in self.order if i in self.stats]
        n_new = min(len(unseen), round(size * self.new_frac) if seen else size)
        batch = unseen[:n_new]
        pool = list(seen)
        while len(batch) < size and pool:
            weights = [self.weight(i, step) for i in pool]
            pick = self.rng.choices(range(len(pool)), weights=weights)[0]
            batch.append(pool.pop(pick))
        batch += unseen[n_new : n_new + size - len(batch)]  # pool exhausted early: fall back to new prompts
        return batch

    def summary(self):
        seen = list(self.stats.values())
        return {
            "prompts_seen": len(seen),
            "dead": sum(1 for st in seen if st["s"] == 0),
            "mastered": sum(1 for st in seen if st["s"] == st["n"]),
            "frontier": sum(1 for st in seen if 0 < st["s"] < st["n"]),
        }
