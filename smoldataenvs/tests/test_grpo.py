from types import SimpleNamespace

import pytest

from pirl.grpo import group_advantages, pack


def rollout(reward, has_seq=True):
    return {"reward": reward, "sequences": [object()] if has_seq else []}


def test_advantages_are_group_normalized():
    group = [rollout(1.08), rollout(0.0), rollout(0.0), rollout(1.02)]
    assert group_advantages(group) is True
    advs = [r["advantage"] for r in group]
    assert sum(advs) == pytest.approx(0.0, abs=1e-6)
    assert advs[0] > advs[3] > 0 > advs[1] == advs[2]


def test_all_equal_group_has_no_contrast():
    group = [rollout(0.0) for _ in range(8)]
    assert group_advantages(group) is False
    assert all(r["advantage"] == 0.0 for r in group)


def test_unscorable_rollouts_are_excluded_from_baseline():
    group = [rollout(1.0), rollout(0.0), rollout(None), rollout(1.0, has_seq=False)]
    group_advantages(group)
    assert group[2]["advantage"] == 0.0 and group[3]["advantage"] == 0.0
    assert group[0]["advantage"] == pytest.approx(1.0, abs=1e-6)
    assert group[1]["advantage"] == pytest.approx(-1.0, abs=1e-6)


def test_single_scorable_rollout_gives_no_signal():
    group = [rollout(1.0), rollout(None), rollout(None)]
    assert group_advantages(group) is False


def seq(n):
    return SimpleNamespace(input_ids=[0] * n)


def test_pack_respects_budget_and_keeps_every_sample():
    samples = [(seq(n), 0.5) for n in (30000, 2000, 15000, 15000, 500, 32768)]
    rows = pack(samples, 32768)
    assert all(r["tokens"] <= 32768 for r in rows)
    assert sorted(len(s[0].input_ids) for r in rows for s in r["items"]) == sorted(len(s[0].input_ids) for s in samples)
    assert len(rows) == 3  # first-fit decreasing: 32768 | 30000+2000+500 | 15000+15000


def test_trim_jsonl_drops_unsaved_updates(tmp_path):
    import json

    from pirl.grpo import trim_jsonl

    f = tmp_path / "metrics.jsonl"
    f.write_text("".join(json.dumps({"step": s}) + "\n" for s in (1, 2, 3, 3, 4)))
    trim_jsonl(f, 3)
    assert [json.loads(line)["step"] for line in f.read_text().splitlines()] == [1, 2, 3, 3]


def test_infra_failure_detection():
    from pirl.grpo import infra_failed

    assert infra_failed({"error": "RuntimeError: harbor rollout call failed 4 times"})
    assert infra_failed({"findings": ["[FATAL] no_turns: the intercept saw no model calls"]})
    assert not infra_failed({"findings": ["[FATAL] turn_mismatch: ..."]})  # capture problem, not an outage
    assert not infra_failed({"reward": 0.0, "sequences": []})


def test_save_state_is_atomic_and_replaces(tmp_path):
    import json

    import torch

    from pirl.grpo import save_state

    class FakeModel:
        def __init__(self, tag):
            self.tag = tag

        def save_pretrained(self, path):
            path.mkdir(parents=True)
            (path / "tag").write_text(self.tag)

    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.1)
    dest = tmp_path / "state_latest"
    save_state(dest, 1, FakeModel("a"), opt, 8)
    save_state(dest, 2, FakeModel("b"), opt, 16)
    assert json.loads((dest / "state.json").read_text())["step"] == 2
    assert (dest / "adapter" / "tag").read_text() == "b"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["state_latest"]  # no tmp/old leftovers
