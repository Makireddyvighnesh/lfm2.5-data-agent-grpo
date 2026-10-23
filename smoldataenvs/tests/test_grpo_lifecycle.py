import json
import sys
from types import ModuleType, SimpleNamespace

from pirl.artifacts import save_rollout
from pirl import grpo


def test_main_retries_same_batch_and_preserves_all_attempts(tmp_path, monkeypatch):
    import torch
    from pirl.curriculum import PromptSampler

    out = tmp_path / "pilot"
    tasks = [{"name": f"task{i}", "instruction": f"question{i}", "difficulty": "hard"} for i in range(10)]
    task_module = ModuleType("smoldataenv_harbor.tasks")
    task_module.load_tasks = lambda *_: tasks
    monkeypatch.setitem(sys.modules, "smoldataenv_harbor", ModuleType("smoldataenv_harbor"))
    monkeypatch.setitem(sys.modules, "smoldataenv_harbor.tasks", task_module)
    args = SimpleNamespace(output=str(out), steps=1, seed=0, task_limit=0, server="fake", vllm_url="fake",
                           groups=2, num_generations=2, sampler="regret", hard_generations=2, new_frac=0.5,
                           lr=2e-5, lora_r=32, lora_alpha=64, epsilon=0.2, epsilon_high=0.28,
                           step_limit=17, agent_timeout=300, concurrency=1, reward="calls", no_sleep=True,
                           train_zero_adv=False, max_infra_fail=0.5, max_infra_retries=2, save_steps=1,
                           token_budget=32, no_autocast=True, max_grad_norm=1.0)
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
    selections = []
    original_selection = PromptSampler.next_batch

    def next_batch(self, *a):
        selected = original_selection(self, *a)
        selections.append(selected)
        return selected

    monkeypatch.setattr(PromptSampler, "next_batch", next_batch)
    seen = []

    def rollout(args, split, idx, task, model, reward, context):
        rid = f"r{len(seen)}"
        seen.append((context["batch_attempt"], context["batch_id"], idx))
        row = {"rollout_id": rid, "task": task["name"], "reward": None, "correct": None,
               "sequences": [], "seconds": 0, **context}
        trace = None
        if context["batch_attempt"] == 1:
            row["error"] = "injected infrastructure failure"
        else:
            value = float(len(seen) % 2)
            seq = SimpleNamespace(input_ids=[10, 20], completion_mask=[0, 1],
                                  old_log_probs=[0, -0.2], rollout_id=rid)
            row.update(reward=value, correct=value, sequences=[seq], submitted=True,
                       context_exhausted=False, truncated=False, tool_calls=1, model_calls=1,
                       generated_tokens=1, final_context=2)
            trace = {"turns": [{"prompt_token_ids": [10], "completion_token_ids": [20],
                               "per_token_logps": [-0.2], "loss_mask": [0, 1]}]}
        save_rollout(out, row, task, {"test": True}, trace)
        return row

    def train(model, optimizer, samples, args):
        return {"ratio_mean": 1.0, "kl": 0.0, "entropy": 0.1, "loss": -0.1, "clip_frac": 0.0,
                "grad_norm": 0.1, "train_s": 0, "trained_completion_tokens": len(samples),
                "per_rollout": {s.rollout_id: {"trained_completion_tokens": 1, "ratio_mean": 1.0,
                                               "kl": 0.0, "clip_frac": 0.0, "loss_mean": -0.1}
                                for s, _ in samples}}

    monkeypatch.setattr(grpo, "run_rollout", rollout)
    monkeypatch.setattr(grpo, "train_step", train)
    grpo.main()
    assert len(selections) == 1  # retry does not draw a different regret-sampler batch
    assert [x[2] for x in seen[:4]] == [x[2] for x in seen[4:]]
    assert len({x[1] for x in seen}) == 1
    assert len(list((out / "rollouts").glob("*.json.gz"))) == 8
    committed = [json.loads(x) for x in (out / "rollouts.jsonl").read_text().splitlines()]
    events = [json.loads(x) for x in (out / "rollout_metrics.jsonl").read_text().splitlines()]
    assert len(committed) == 4 and all(r["trained"] for r in committed)
    assert sum(e["event"] == "rollout_finished" for e in events) == 8
    assert sum(e["event"] == "optimizer_update" for e in events) == 4
    state = json.loads((out / "state_latest/state.json").read_text())
    assert state["step"] == 1 and state["sampler"]["rng_state"]
    assert json.loads((out / "results.json").read_text())["status"] == "complete"


def test_stop_marker_prevents_model_loading(tmp_path, monkeypatch):
    (tmp_path / "STOP").touch()
    monkeypatch.setattr(grpo, "arguments", lambda: SimpleNamespace(output=str(tmp_path)))
    monkeypatch.setattr(grpo, "build_model", lambda _: (_ for _ in ()).throw(AssertionError("must not load model")))
    grpo.main()
    assert not (tmp_path / "state_latest").exists()
