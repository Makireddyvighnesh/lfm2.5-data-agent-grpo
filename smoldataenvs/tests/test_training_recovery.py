import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from pirl.artifacts import save_rollout
from pirl.curriculum import PromptSampler
from pirl.grpo import pack, recover_state, save_state, trim_jsonl


def test_sampler_preserves_random_sequence_through_json_roundtrip():
    sampler = PromptSampler(range(100), seed=3)
    for i in range(20):
        sampler.update(i, i % 4, 4, 1)
    sampler.next_batch(8, 2)
    saved = json.loads(json.dumps(sampler.state_dict()))
    resumed = PromptSampler(range(100), seed=3)
    resumed.load_state_dict(saved)
    assert resumed.stats == sampler.stats
    assert [resumed.next_batch(8, s) for s in range(3, 10)] == [sampler.next_batch(8, s) for s in range(3, 10)]


def test_checkpoint_recovers_if_killed_between_directory_renames(tmp_path, monkeypatch):
    import torch

    class Model:
        def __init__(self, tag):
            self.tag = tag

        def save_pretrained(self, path):
            path.mkdir()
            (path / "weights").write_text(self.tag)

    dest = tmp_path / "state_latest"
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.1)
    sampler = PromptSampler(range(10), seed=7)
    sampler.update(0, 1, 4, 1)
    sampler.next_batch(4, 2)
    save_state(dest, 1, Model("old"), optimizer, 8, sampler)
    rename = Path.rename

    def interrupted(path, target):
        if path.name == ".state_latest.tmp":
            raise OSError("injected interruption")
        return rename(path, target)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "rename", interrupted)
        with pytest.raises(OSError, match="injected"):
            save_state(dest, 2, Model("new"), optimizer, 16, sampler)
    assert not dest.exists()
    assert recover_state(dest)
    state = json.loads((dest / "state.json").read_text())
    assert state["step"] == 1 and state["task_cursor"] == 8
    assert (dest / "adapter/weights").read_text() == "old"
    restored = PromptSampler(range(10), seed=7)
    restored.load_state_dict(state["sampler"])
    assert restored.next_batch(4, 3) == sampler.next_batch(4, 3)
    save_state(dest, 2, Model("new"), optimizer, 16, sampler)
    assert json.loads((dest / "state.json").read_text())["step"] == 2


def test_trim_jsonl_recovers_partial_last_append(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text('{"step":1}\n{"step":2}\n{"step":')
    trim_jsonl(path, 1)
    assert path.read_text() == '{"step":1}\n'


def test_trim_jsonl_does_not_hide_corruption_in_middle(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text('{"step":1}\nbroken\n{"step":2}\n')
    with pytest.raises(json.JSONDecodeError):
        trim_jsonl(path, 1)


def test_pack_rejects_oversized_single_sequence():
    with pytest.raises(ValueError, match="exceeding"):
        pack([(SimpleNamespace(input_ids=[1] * 11), 1.0)], 10)


def test_rollout_capture_keeps_exact_tokens_before_update(tmp_path):
    logp = -0.123456789012345
    seq = SimpleNamespace(input_ids=[10, 20, 30], completion_mask=[0, 0, 1],
                          old_log_probs=[0.0, 0.0, logp], rollout_id="r1")
    row = {"rollout_id": "r1", "intended_step": 1, "reward": 1.0, "sequences": [seq]}
    trace = {"turns": [{"prompt_token_ids": [10, 20], "completion_token_ids": [30],
                       "per_token_logps": [logp], "loss_mask": [0, 0, 1]}]}
    save_rollout(tmp_path, row, {"name": "task", "instruction": "question", "difficulty": "hard"},
                 {"reward": 1.0, "turns": trace["turns"]}, trace)
    with gzip.open(tmp_path / row["capture_file"], "rt") as stream:
        bundle = json.load(stream)
    assert bundle["training_trace"] == trace
    assert bundle["training_sequences"][0]["old_log_probs"][-1] == logp
    assert bundle["result"]["reward"] == 1.0
    assert not (tmp_path / "state_latest").exists()  # available even if optimization never commits
    metric = json.loads((tmp_path / "rollout_metrics.jsonl").read_text())
    assert metric["event"] == "rollout_finished" and metric["capture_file"] == row["capture_file"]


def test_per_rollout_optimizer_stats_follow_packed_boundaries(monkeypatch):
    import contextlib
    import torch
    from pirl.grpo import train_step

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(()))

        def forward(self, input_ids, **kwargs):
            shape = (1, input_ids.shape[1] - 1)
            return SimpleNamespace(log_probs=self.weight.expand(shape), entropy=torch.ones(shape))

    tensor = torch.tensor
    monkeypatch.setattr(torch, "tensor", lambda *a, **kw: tensor(*a, **{k: v for k, v in kw.items() if k != "device"}))
    monkeypatch.setattr(torch, "autocast", lambda *a, **kw: contextlib.nullcontext())
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)
    model = Model()
    samples = [(SimpleNamespace(input_ids=[1, 2, 3], completion_mask=[0, 1, 1],
                                old_log_probs=[0, 0, 0], rollout_id=rid), adv)
               for rid, adv in (("a", 1), ("b", -0.5))]
    args = SimpleNamespace(token_budget=10, no_autocast=True, epsilon=0.2, epsilon_high=0.28, max_grad_norm=1)
    stats = train_step(model, torch.optim.SGD(model.parameters(), lr=0.1), samples, args)
    assert stats["trained_completion_tokens"] == 4
    assert stats["per_rollout"]["a"]["trained_completion_tokens"] == 2
    assert stats["per_rollout"]["b"]["trained_completion_tokens"] == 2
    assert stats["per_rollout"]["a"]["loss_mean"] == pytest.approx(-1)
    assert stats["per_rollout"]["b"]["loss_mean"] == pytest.approx(0.5)
    assert stats["loss"] == pytest.approx(-0.25)
    assert model.weight.item() > 0  # a real CPU optimizer update occurred
