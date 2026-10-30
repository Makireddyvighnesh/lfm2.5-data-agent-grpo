import json
import sys
from types import ModuleType, SimpleNamespace

from pirl import grpo


def test_async_dynamic_sampling_trains_only_contrast_groups_and_skips_revisits(tmp_path, monkeypatch):
    import torch

    out = tmp_path / "dyn"
    # task i: i % 3 == 0 always correct, == 1 always wrong, == 2 mixed
    tasks = [{"name": f"task{i}", "instruction": f"q{i}", "difficulty": "hard"} for i in range(6)]
    task_module = ModuleType("smoldataenv_harbor.tasks")
    task_module.load_tasks = lambda *_: tasks
    monkeypatch.setitem(sys.modules, "smoldataenv_harbor", ModuleType("smoldataenv_harbor"))
    monkeypatch.setitem(sys.modules, "smoldataenv_harbor.tasks", task_module)
    args = SimpleNamespace(output=str(out), steps=4, seed=0, task_limit=0, server="fake", vllm_url="fake",
                           groups=1, num_generations=2, sampler="sequential", hard_generations=2, new_frac=0.5,
                           lr=2e-5, lora_r=32, lora_alpha=64, epsilon=0.2, epsilon_high=0.28,
                           step_limit=17, agent_timeout=300, concurrency=4, reward="calls", no_sleep=True,
                           train_zero_adv=False, max_infra_fail=0.5, max_infra_retries=2, save_steps=1,
                           token_budget=32, no_autocast=True, max_grad_norm=1.0, stream_groups=1,
                           async_groups=2, dynamic_sampling=True, dynamic_max_drop=32, dynamic_skip=True,
                           skip_wrong_frac=0.5)
    monkeypatch.setattr(grpo, "arguments", lambda: args)
    monkeypatch.setattr(grpo.httpx, "get", lambda *a, **k: SimpleNamespace(json=lambda: {"train": "fake"}))
    monkeypatch.setattr(grpo.time, "sleep", lambda *_: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))

        def save_pretrained(self, path):
            path.mkdir(parents=True)
            (path / "weights").write_text("test weights")

    class Engine:
        def __init__(self, *_):
            self.url = "fake"
            self.http = SimpleNamespace(get=lambda *a: SimpleNamespace(json=lambda: {"is_sleeping": False}), close=lambda: None)

        def load_adapter(self, *_):
            pass

        def unload_adapter(self, *_):
            pass

    monkeypatch.setattr(grpo, "Engine", Engine)
    monkeypatch.setattr(grpo, "build_model", lambda _: Model())
    counter = {"n": 0}

    def rollout(args, split, idx, task, model, reward, context):
        counter["n"] += 1
        rid = f"r{counter['n']}"
        value = {0: 1.0, 1: 0.0, 2: float(counter["n"] % 2)}[idx % 3]
        seq = SimpleNamespace(input_ids=[10, 20], completion_mask=[0, 1], old_log_probs=[0, -0.2], rollout_id=rid)
        return {"rollout_id": rid, "task": task["name"], "task_index": idx, "reward": value, "correct": value,
                "sequences": [seq], "seconds": 0, "submitted": True, "context_exhausted": False, "truncated": False,
                "tool_calls": 1, "model_calls": 1, "generated_tokens": 1, "final_context": 2, **context}

    def train(model, optimizer, samples, args):
        return {"ratio_mean": 1.0, "kl": 0.0, "entropy": 0.1, "loss": -0.1, "clip_frac": 0.0,
                "grad_norm": 0.1, "train_s": 0, "trained_completion_tokens": len(samples),
                "per_rollout": {s.rollout_id: {"trained_completion_tokens": 1} for s, _ in samples}}

    monkeypatch.setattr(grpo, "run_rollout", rollout)
    monkeypatch.setattr(grpo, "train_step", train)
    exits = []

    def fake_exit(code):
        exits.append(code)
        raise SystemExit(code)

    monkeypatch.setattr(grpo.os, "_exit", fake_exit)
    try:
        grpo.main()
    except SystemExit:
        pass
    assert exits == [0]  # in-flight groups are abandoned (positions saved), not waited for
    state = json.loads((out / "state_latest/state.json").read_text())
    assert state["step"] == 4 and state["stream_pending"]

    trained = [json.loads(x) for x in (out / "rollouts.jsonl").read_text().splitlines()]
    assert len(trained) == 8 and {r["task_index"] % 3 for r in trained} == {2}  # only mixed groups train
    assert all("pos" in r for r in trained)
    filtered = [json.loads(x) for x in (out / "filtered_rollouts.jsonl").read_text().splitlines()]
    assert filtered and {r["task_index"] % 3 for r in filtered} <= {0, 1}
    skipped = [json.loads(x) for x in (out / "skipped_tasks.jsonl").read_text().splitlines()]
    solved_skips = [x for x in skipped if x["reason"] == "solved"]
    assert solved_skips and all(x["idx"] % 3 == 0 for x in solved_skips)  # all-correct tasks are removed
    assert all(x["idx"] % 3 == 1 for x in skipped if x["reason"] == "wrong")
    # once removed, an all-correct task is never trained or filtered again after its first visit
    first = {}
    for r in filtered:
        first.setdefault(r["task_index"], r["pos"])
    for r in filtered:
        if r["task_index"] % 3 == 0:
            assert r["pos"] == first[r["task_index"]]
    metrics = [json.loads(x) for x in (out / "metrics.jsonl").read_text().splitlines()]
    assert len(metrics) == 4 and all(m["correctness_contrast_frac"] == 1.0 for m in metrics)
    assert json.loads((out / "results.json").read_text())["status"] == "complete"


def test_stop_at_pos_finishes_the_epoch_and_stops(tmp_path, monkeypatch):
    import torch

    out = tmp_path / "epoch"
    # task i: i % 3 == 0 always correct, == 1 always wrong, == 2 mixed
    tasks = [{"name": f"task{i}", "instruction": f"q{i}", "difficulty": "hard"} for i in range(6)]
    task_module = ModuleType("smoldataenv_harbor.tasks")
    task_module.load_tasks = lambda *_: tasks
    monkeypatch.setitem(sys.modules, "smoldataenv_harbor", ModuleType("smoldataenv_harbor"))
    monkeypatch.setitem(sys.modules, "smoldataenv_harbor.tasks", task_module)
    args = SimpleNamespace(output=str(out), steps=100, seed=0, task_limit=0, server="fake", vllm_url="fake",
                           groups=1, num_generations=2, sampler="sequential", hard_generations=2, new_frac=0.5,
                           lr=2e-5, lora_r=32, lora_alpha=64, epsilon=0.2, epsilon_high=0.28,
                           step_limit=17, agent_timeout=300, concurrency=4, reward="calls", no_sleep=True,
                           train_zero_adv=False, max_infra_fail=0.5, max_infra_retries=2, save_steps=1,
                           token_budget=32, no_autocast=True, max_grad_norm=1.0, stream_groups=1,
                           async_groups=2, dynamic_sampling=True, dynamic_max_drop=32, dynamic_skip=True,
                           skip_wrong_frac=0.5, stop_at_pos=6)
    monkeypatch.setattr(grpo, "arguments", lambda: args)
    monkeypatch.setattr(grpo.httpx, "get", lambda *a, **k: SimpleNamespace(json=lambda: {"train": "fake"}))
    monkeypatch.setattr(grpo.time, "sleep", lambda *_: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))

        def save_pretrained(self, path):
            path.mkdir(parents=True)
            (path / "weights").write_text("test weights")

    class Engine:
        def __init__(self, *_):
            self.url = "fake"
            self.http = SimpleNamespace(get=lambda *a: SimpleNamespace(json=lambda: {"is_sleeping": False}), close=lambda: None)

        def load_adapter(self, *_):
            pass

        def unload_adapter(self, *_):
            pass

    monkeypatch.setattr(grpo, "Engine", Engine)
    monkeypatch.setattr(grpo, "build_model", lambda _: Model())
    counter = {"n": 0}

    def rollout(args, split, idx, task, model, reward, context):
        counter["n"] += 1
        rid = f"r{counter['n']}"
        value = {0: 1.0, 1: 0.0, 2: float(counter["n"] % 2)}[idx % 3]
        seq = SimpleNamespace(input_ids=[10, 20], completion_mask=[0, 1], old_log_probs=[0, -0.2], rollout_id=rid)
        return {"rollout_id": rid, "task": task["name"], "task_index": idx, "reward": value, "correct": value,
                "sequences": [seq], "seconds": 0, "submitted": True, "context_exhausted": False, "truncated": False,
                "tool_calls": 1, "model_calls": 1, "generated_tokens": 1, "final_context": 2, **context}

    def train(model, optimizer, samples, args):
        return {"ratio_mean": 1.0, "kl": 0.0, "entropy": 0.1, "loss": -0.1, "clip_frac": 0.0,
                "grad_norm": 0.1, "train_s": 0, "trained_completion_tokens": len(samples),
                "per_rollout": {s.rollout_id: {"trained_completion_tokens": 1} for s, _ in samples}}

    monkeypatch.setattr(grpo, "run_rollout", rollout)
    monkeypatch.setattr(grpo, "train_step", train)
    exits = []

    def fake_exit(code):
        exits.append(code)
        raise SystemExit(code)

    monkeypatch.setattr(grpo.os, "_exit", fake_exit)
    try:
        grpo.main()
    except SystemExit:
        pass
    assert exits == []  # nothing left in flight at the epoch end

    trained = [json.loads(x) for x in (out / "rollouts.jsonl").read_text().splitlines()]
    filtered = [json.loads(x) for x in (out / "filtered_rollouts.jsonl").read_text().splitlines()]
    positions = {r["pos"] for r in trained + filtered}
    assert positions == set(range(6))  # every epoch-1 position finished, none past the stop
    assert {r["task_index"] % 3 for r in trained} == {2}
    assert json.loads((out / "results.json").read_text())["status"] == "stopped"


def test_probe_hard_drops_hopeless_revisits_cheaply_and_completes_promising_ones(tmp_path, monkeypatch):
    import collections
    import torch

    out = tmp_path / "probe"
    # i % 4: 0 always correct, 1 always wrong, 2 mixed, 3 wrong on its first visit then mixed
    tasks = [{"name": f"task{i}", "instruction": f"q{i}", "difficulty": "hard"} for i in range(8)]
    task_module = ModuleType("smoldataenv_harbor.tasks")
    task_module.load_tasks = lambda *_: tasks
    monkeypatch.setitem(sys.modules, "smoldataenv_harbor", ModuleType("smoldataenv_harbor"))
    monkeypatch.setitem(sys.modules, "smoldataenv_harbor.tasks", task_module)
    args = SimpleNamespace(output=str(out), steps=100, seed=0, task_limit=0, server="fake", vllm_url="fake",
                           groups=1, num_generations=4, sampler="sequential", hard_generations=4, new_frac=0.5,
                           lr=2e-5, lora_r=32, lora_alpha=64, epsilon=0.2, epsilon_high=0.28,
                           step_limit=17, agent_timeout=300, concurrency=8, reward="calls", no_sleep=True,
                           train_zero_adv=False, max_infra_fail=0.5, max_infra_retries=2, save_steps=1,
                           token_budget=32, no_autocast=True, max_grad_norm=1.0, stream_groups=1,
                           async_groups=2, dynamic_sampling=True, dynamic_max_drop=32, dynamic_skip=False,
                           skip_wrong_frac=0.5, stop_at_pos=24, probe_hard=2)
    monkeypatch.setattr(grpo, "arguments", lambda: args)
    monkeypatch.setattr(grpo.httpx, "get", lambda *a, **k: SimpleNamespace(json=lambda: {"train": "fake"}))
    monkeypatch.setattr(grpo.time, "sleep", lambda *_: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))

        def save_pretrained(self, path):
            path.mkdir(parents=True)
            (path / "weights").write_text("test weights")

    class Engine:
        def __init__(self, *_):
            self.url = "fake"
            self.http = SimpleNamespace(get=lambda *a: SimpleNamespace(json=lambda: {"is_sleeping": False}), close=lambda: None)

        def load_adapter(self, *_):
            pass

        def unload_adapter(self, *_):
            pass

    monkeypatch.setattr(grpo, "Engine", Engine)
    monkeypatch.setattr(grpo, "build_model", lambda _: Model())
    import threading
    lock = threading.Lock()
    visits, seen_in_batch, counter = {}, collections.defaultdict(int), {"n": 0}

    def rollout(args, split, idx, task, model, reward, context):
        with lock:
            counter["n"] += 1
            rid = f"r{counter['n']}"
            visit = visits.setdefault((idx, context["batch_id"]), len([k for k in visits if k[0] == idx]) + 1)
            k = seen_in_batch[context["batch_id"]]
            seen_in_batch[context["batch_id"]] += 1
        kind = idx % 4
        value = {0: 1.0, 1: 0.0, 2: float(k % 2 == 0), 3: 0.0 if visit == 1 else float(k % 2 == 0)}[kind]
        seq = SimpleNamespace(input_ids=[10, 20], completion_mask=[0, 1], old_log_probs=[0, -0.2], rollout_id=rid)
        return {"rollout_id": rid, "task": task["name"], "task_index": idx, "reward": value, "correct": value,
                "sequences": [seq], "seconds": 0, "submitted": True, "context_exhausted": False, "truncated": False,
                "tool_calls": 1, "model_calls": 1, "generated_tokens": 1, "final_context": 2, **context}

    def train(model, optimizer, samples, args):
        return {"ratio_mean": 1.0, "kl": 0.0, "entropy": 0.1, "loss": -0.1, "clip_frac": 0.0,
                "grad_norm": 0.1, "train_s": 0, "trained_completion_tokens": len(samples),
                "per_rollout": {s.rollout_id: {"trained_completion_tokens": 1} for s, _ in samples}}

    monkeypatch.setattr(grpo, "run_rollout", rollout)
    monkeypatch.setattr(grpo, "train_step", train)
    monkeypatch.setattr(grpo.os, "_exit", lambda code: (_ for _ in ()).throw(SystemExit(code)))
    try:
        grpo.main()
    except SystemExit:
        pass

    trained = [json.loads(x) for x in (out / "rollouts.jsonl").read_text().splitlines()]
    filtered = [json.loads(x) for x in (out / "filtered_rollouts.jsonl").read_text().splitlines()]
    by_batch = collections.defaultdict(list)
    for r in trained + filtered:
        by_batch[r["batch_id"]].append(r)
    sizes = {b: len(v) for b, v in by_batch.items()}
    # always-wrong tasks: first visit is a full group (4), every revisit is a 2-rollout probe that gets dropped
    wrong = [v for v in by_batch.values() if v[0]["task_index"] % 4 == 1]
    assert sorted(len(v) for v in wrong)[0] == 2 and max(len(v) for v in wrong) == 4
    full = collections.Counter(v[0]["task_index"] for v in wrong if len(v) == 4)
    assert set(full.values()) == {1} and len(full) == len({v[0]["task_index"] for v in wrong})  # one full visit each
    assert all(r["batch_id"] not in {x["batch_id"] for x in trained} for v in wrong for r in v)  # never trained
    # first-visit-wrong tasks: the revisit's probe succeeds, the group is completed to 4 and trained
    late = [v for v in by_batch.values() if v[0]["task_index"] % 4 == 3 and any(r["correct"] == 1 for r in v)]
    assert late and all(len(v) == 4 for v in late) and all(r in trained for v in late for r in v)
    assert all(n in (2, 4) for n in sizes.values())
    assert {r["task_index"] % 4 for r in trained} <= {2, 3}  # only groups with contrast are trained
    metrics = [json.loads(x) for x in (out / "metrics.jsonl").read_text().splitlines()]
    assert sum(n == 2 for n in sizes.values()) >= 2  # dropped probes cost 2 rollouts, not 4
    assert sum(m.get("probe_started", 0) for m in metrics) >= 1 and sum(m.get("probe_extended", 0) for m in metrics) >= 1
    assert json.loads((out / "results.json").read_text())["status"] == "stopped"
