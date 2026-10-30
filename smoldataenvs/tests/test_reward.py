from types import SimpleNamespace

import pytest

from pirl.reward import correctness_only, reward_value, rollout_reward


def test_unscorable_is_none_not_wrong():
    assert reward_value(None, 3) is None


@pytest.mark.parametrize("calls", [0, 5, 30, 100])
def test_wrong_answers_get_no_bonus(calls):
    assert reward_value(0.0, calls) == 0.0


def test_bonus_shrinks_with_tool_calls():
    assert reward_value(1.0, 0) == pytest.approx(1.1)
    assert reward_value(1.0, 15) == pytest.approx(1.05)
    assert reward_value(1.0, 30) == pytest.approx(1.0)
    assert reward_value(1.0, 60) == pytest.approx(1.0)  # clipped at zero, never below correctness
    assert reward_value(1.0, 3) > reward_value(1.0, 10)


def test_bonus_bounded_by_point_one():
    assert all(1.0 <= reward_value(1.0, n) <= 1.1 for n in range(0, 50))


def test_unknown_call_count_still_rewards_correctness():
    assert reward_value(1.0, None) == 1.0


def test_outcome_hooks():
    right = SimpleNamespace(env_reward=1.0, tool_call_count=6)
    wrong = SimpleNamespace(env_reward=0.0, tool_call_count=1)
    broken = SimpleNamespace(env_reward=None, tool_call_count=0)
    assert rollout_reward(right) == pytest.approx(1.08)
    assert rollout_reward(wrong) == 0.0
    assert rollout_reward(broken) is None
    assert correctness_only(right) == 1.0


from pirl.reward import REWARDS, efficiency_value


def test_efficiency_reward_bounds_and_gating():
    assert efficiency_value(None, 3, 1000, 100) is None
    assert efficiency_value(0.0, 1, 100, 10) == 0.0  # wrong: no bonus however efficient
    best = efficiency_value(1.0, 0, 0, 0)
    worst = efficiency_value(1.0, 60, 40_000, 9_000)
    assert best == pytest.approx(1.13) and worst == pytest.approx(1.0)
    assert all(1.0 <= efficiency_value(1.0, c, x, g) <= 1.13 + 1e-9
               for c in (0, 5, 40) for x in (0, 15_000, 30_000) for g in (0, 2_000, 5_000))


def test_efficiency_prefers_small_context_among_correct():
    peek = efficiency_value(1.0, 5, 6_000, 800)
    read_whole = efficiency_value(1.0, 5, 24_000, 800)
    assert peek > read_whole


def test_variants_registry():
    o = SimpleNamespace(env_reward=1.0, tool_call_count=6, final_context=10_000, generated_tokens=1_000)
    assert REWARDS["correctness"](o) == 1.0
    assert REWARDS["calls"](o) == pytest.approx(1.08)
    assert 1.0 < REWARDS["efficiency"](o) < 1.13


def test_tool_ctx_reward():
    from pirl.reward import tool_ctx_value
    assert tool_ctx_value(None, 3, 5000, 900) is None
    assert tool_ctx_value(0.0, 3, 5000, 900) == 0.0
    tool = 0.1 * 15 / (15 + 5)
    assert abs(tool_ctx_value(1.0, 5, 15_900, 900) - (1.1 + tool)) < 1e-9   # own 15,000: bonus
    assert abs(tool_ctx_value(1.0, 5, 15_901, 900) - (1.0 + tool)) < 1e-9   # own 15,001: no bonus
    assert abs(tool_ctx_value(1.0, 5, 21_000, 6_330) - (1.1 + tool)) < 1e-9  # heavy harness, own 14,670
    assert tool_ctx_value(1.0, 5, None, None) == 1.0 + tool                 # unknown context: no bonus


def test_harness_assignment():
    import types
    from pirl.grpo import harness_for
    a = types.SimpleNamespace(harnesses="pi,mini-swe-agent,codex")
    assert [harness_for(a, i) for i in range(4)] == ["pi", "mini-swe-agent", "codex", "pi"]
    assert harness_for(types.SimpleNamespace(), 7) == "pi"


def test_lr_schedule():
    import types
    from pirl.grpo import lr_at
    a = types.SimpleNamespace(lr=3e-5, steps=110, lr_schedule="cosine", warmup_steps=10, min_lr=3e-6)
    assert abs(lr_at(a, 0) - 3e-6) < 1e-12 and abs(lr_at(a, 9) - 3e-5) < 1e-12
    assert abs(lr_at(a, 10) - 3e-5) < 1e-12 and abs(lr_at(a, 60) - 1.65e-5) < 1e-12 and abs(lr_at(a, 110) - 3e-6) < 1e-12
    assert lr_at(types.SimpleNamespace(lr=1e-5, steps=5), 3) == 1e-5


def test_tool_ctx_err_penalises_unknown_tools_on_every_scored_rollout():
    from pirl.reward import tool_ctx_err_value, tool_ctx_value
    base = tool_ctx_value(1.0, 5, 3000, 900)
    assert tool_ctx_err_value(1.0, 5, 3000, 900, 0) == base
    assert abs(tool_ctx_err_value(1.0, 5, 3000, 900, 2) - (base - 0.2)) < 1e-9
    assert abs(tool_ctx_err_value(1.0, 5, 3000, 900, 9) - (base - 0.3)) < 1e-9  # capped
    assert abs(tool_ctx_err_value(0.0, 5, 3000, 900, 1) - (-0.1)) < 1e-9  # wrong answers are penalised too
    assert tool_ctx_err_value(0.0, 5, 3000, 900, None) == 0.0
    assert tool_ctx_err_value(None, 5, 3000, 900, 3) is None  # infra failure stays unscorable


def test_tool_ctx_read_grades_own_context_and_penalises_capped_outputs():
    from pirl.reward import tool_ctx_read_value, tool_efficiency_bonus
    tool = tool_efficiency_bonus(5)
    assert abs(tool_ctx_read_value(1.0, 5, 4_900, 900, 0, 0) - (1.1 + tool)) < 1e-9    # own 4k: full bonus
    assert abs(tool_ctx_read_value(1.0, 5, 8_900, 900, 0, 0) - (1.05 + tool)) < 1e-9   # own 8k: half
    assert abs(tool_ctx_read_value(1.0, 5, 13_900, 900, 0, 0) - (1.0 + tool)) < 1e-9   # own 13k: none
    assert abs(tool_ctx_read_value(1.0, 5, 4_900, 900, 0, 2) - (1.0 + tool)) < 1e-9    # 2 capped outputs: -0.1
    assert abs(tool_ctx_read_value(1.0, 5, 4_900, 900, 9, 9) - (0.6 + tool)) < 1e-9    # both penalties at their caps
    assert abs(tool_ctx_read_value(0.0, 5, 4_900, 900, 1, 1) - (-0.15)) < 1e-9         # wrong: penalties only
    assert tool_ctx_read_value(0.0, 5, 4_900, 900, None, None) == 0.0
    assert tool_ctx_read_value(None, 5, 4_900, 900, 3, 3) is None


def test_capped_tool_outputs_counts_each_long_tool_message_once():
    from pirl.grpo import capped_tool_outputs
    long, short = "x" * 4001, "y" * 10
    turns = [{"request_messages": [{"role": "user", "content": "q"}]},
             {"request_messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": ""},
                                   {"role": "tool", "content": long}, {"role": "tool", "content": short}]},
             {"request_messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": ""},
                                   {"role": "tool", "content": long}, {"role": "tool", "content": short},
                                   {"role": "assistant", "content": ""},
                                   {"role": "tool", "content": [{"type": "text", "text": long}]}]}]
    assert capped_tool_outputs({"turns": turns}, 4000) == 2   # the first long output is not recounted on turn 3
    assert capped_tool_outputs({"result": {"turns": turns}}, 4000) == 2
    assert capped_tool_outputs({"turns": turns}, 0) is None
    assert capped_tool_outputs({"turns": [{"turn": 0}]}, 4000) is None  # no request messages recorded


def test_tool_ctx_read2_charges_capped_outputs_on_correct_answers_only():
    from pirl.reward import tool_ctx_read2_value, tool_ctx_read_value
    assert tool_ctx_read2_value(1.0, 5, 4_900, 900, 0, 2) == tool_ctx_read_value(1.0, 5, 4_900, 900, 0, 2)
    assert tool_ctx_read2_value(0.0, 5, 4_900, 900, 0, 3) == 0.0                   # wrong: no capped penalty
    assert abs(tool_ctx_read2_value(0.0, 5, 4_900, 900, 2, 3) - (-0.2)) < 1e-9    # unknown tools still cost
    assert tool_ctx_read2_value(None, 5, 4_900, 900, 1, 1) is None


def test_keep_group_reward_spread_keeps_ranked_all_correct_groups():
    from pirl.grpo import keep_group
    row = lambda reward: {"reward": reward, "correct": 1.0, "sequences": [1]}
    ranked, flat = [row(1.15), row(1.05)], [row(1.10), row(1.08)]
    assert not keep_group(ranked) and not keep_group(ranked, min_reward_spread=0.0)
    assert keep_group(ranked, min_reward_spread=0.05) and not keep_group(flat, min_reward_spread=0.05)
    assert not keep_group([row(1.1), {"reward": None, "correct": None, "sequences": []}], min_reward_spread=0.01)


def test_skip_reason_can_revisit_solved_tasks():
    import types
    from pirl.grpo import skip_reason
    off = types.SimpleNamespace(skip_wrong_frac=0.5)
    on = types.SimpleNamespace(skip_wrong_frac=0.5, solved_revisit_frac=0.5)
    assert skip_reason(["c"], 0.1, off) == "solved"       # default: solved tasks are never revisited
    assert skip_reason(["c"], 0.1, on) is None and skip_reason(["c"], 0.7, on) == "solved"
    assert skip_reason(["m", "w"], 0.3, on) == "wrong" and skip_reason(["m", "w"], 0.6, on) is None
    assert skip_reason(["m"], 0.0, on) is None
