from pirl.grpo import correctness_contrast, group_advantages
from pirl.reward import context_value


def test_context_reward_pays_bonus_only_on_correct_answers():
    assert context_value(None, 5_000) is None
    assert context_value(0.0, 2_000) == 0.0
    assert context_value(1.0, 8_000) == 1.1
    assert context_value(1.0, 10_000) == 1.1
    assert abs(context_value(1.0, 12_500) - 1.05) < 1e-9
    assert context_value(1.0, 15_000) == 1.0
    assert context_value(1.0, 30_000) == 1.0
    assert context_value(1.0, None) == 1.0


def _group(rewards, correct):
    return [{"reward": r, "correct": c, "sequences": [object()]} for r, c in zip(rewards, correct)]


def test_std_floor_keeps_bonus_only_advantages_small():
    group = _group([1.10, 1.09, 1.10, 1.08], [1, 1, 1, 1])
    group_advantages(group)
    assert max(abs(r["advantage"]) for r in group) > 1.0  # TRL behaviour: tiny differences become full-size
    group_advantages(group, std_floor=0.1)
    assert max(abs(r["advantage"]) for r in group) < 0.2
    assert not correctness_contrast(group)


def test_std_floor_leaves_correctness_contrast_intact():
    group = _group([1.1, 0.0, 0.0, 0.0], [1, 0, 0, 0])
    group_advantages(group, std_floor=0.1)
    assert group[0]["advantage"] > 1.5
    assert correctness_contrast(group)


def test_root_weight_lookup_and_weighted_baseline(tmp_path):
    import hashlib
    import json

    from pirl import grpo

    ids, prompt = [11, 22, 33], [1, 2, 3]
    sha = lambda x: hashlib.sha1(json.dumps(x).encode()).hexdigest()
    path = tmp_path / "root.jsonl"
    rec = {"model": "m-v1", "prompt_sha1": sha(prompt), "ids_sha1": sha(ids), "weight": 0.125, "class": "bash",
           "stratum": "alt", "p_alt": 0.03}
    path.write_text(json.dumps(rec) + "\n")
    grpo._ROOT.update(offset=0, weights={})
    assert grpo.root_weight(str(path), "m-v1", prompt, ids)["weight"] == 0.125
    assert grpo.root_weight(str(path), "m-v1", prompt, [1, 2]) is None
    # same opening text under a newer policy / other task gets its own record (the v1 collision bug)
    with path.open("a") as f:
        f.write(json.dumps(dict(rec, model="m-v2", weight=0.75)) + "\n")
    assert grpo.root_weight(str(path), "m-v2", prompt, ids)["weight"] == 0.75
    assert grpo.root_weight(str(path), "m-v1", prompt, ids)["weight"] == 0.125
    # A rare (down-weighted) success barely moves the baseline but still gets a positive advantage.
    group = [{"reward": 1.1, "correct": 1, "sequences": [object()], "root_weight": 0.125}] + \
            [{"reward": 0.0, "correct": 0, "sequences": [object()], "root_weight": 1.29} for _ in range(7)]
    group_advantages(group, std_floor=0.1)
    assert group[0]["advantage"] > 1.0 and all(r["advantage"] < 0 for r in group[1:])


def test_tool_reward_is_correctness_with_blog_tool_bonus():
    from pirl.reward import tool_efficiency_bonus, tool_value

    assert tool_value(None, 5) is None
    assert tool_value(0.0, 1) == 0.0                       # wrong answer: 0, never a bonus
    assert abs(tool_efficiency_bonus(5) - 0.075) < 1e-9 and tool_efficiency_bonus(0) == 0.0
    assert abs(tool_value(1.0, 4) - (1 + 0.1 * 15 / 19)) < 1e-9
    assert tool_value(1.0, 4) > tool_value(1.0, 12) > 1.0  # among correct answers, fewer calls win


def test_tool_context_reward_adds_context_bonus_only_on_correct_answers():
    from pirl.reward import REWARDS, context_efficiency_bonus, tool_context_value, tool_efficiency_bonus
    assert tool_context_value(None, 5, 6000) is None
    assert tool_context_value(0.0, 5, 3000) == 0.0
    assert context_efficiency_bonus(3000) == 0.1 and context_efficiency_bonus(8000) == 0.05
    assert context_efficiency_bonus(12000) == 0.0 and context_efficiency_bonus(None) == 0.0
    assert abs(tool_context_value(1.0, 5, 8000) - (1.0 + tool_efficiency_bonus(5) + 0.05)) < 1e-12
    assert tool_context_value(1.0, 5, 5000) > tool_context_value(1.0, 5, 9000) > 1.0
    assert "tool_context" in REWARDS


def test_tool_fast_reward_time_bonus_under_5_min_penalty_over_on_correct_answers_only():
    from pirl.reward import REWARDS, time_term, token_penalty, tool_efficiency_bonus, tool_fast_value
    assert tool_fast_value(None, 5, 50, 900) is None
    assert tool_fast_value(0.0, 30, 590, 9000) == 0.0 and tool_fast_value(0.0, 1, 5, 100) == 0.0
    assert time_term(0) == 0.05 and abs(time_term(60) - 0.04) < 1e-12 and time_term(300) == 0.0
    assert abs(time_term(450) + 0.025) < 1e-12 and time_term(600) == -0.05 and time_term(900) == -0.05
    assert time_term(None) == 0.0
    assert token_penalty(200) == 0.0 and abs(token_penalty(1650) - 0.05) < 1e-12 and token_penalty(9000) == 0.1
    assert abs(tool_fast_value(1.0, 5, 450, 1650) - (1.0 + tool_efficiency_bonus(5) - 0.025 - 0.05)) < 1e-12
    assert tool_fast_value(1.0, 5, 30, 500) > tool_fast_value(1.0, 5, 400, 2500) >= 0.85
    assert "tool_fast" in REWARDS
