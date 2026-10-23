"""Rollout rewards. Correctness dominates; small efficiency bonuses are paid on correct answers only.

Rules from the multi-harness RL blog that every variant keeps:
- unscorable (verifier did not run) -> None, excluded from the group baseline, never "wrong";
- wrong or missing answer -> 0, with no bonus of any kind;
- no term for merely submitting (a +0.2 submit bonus collapsed held-out accuracy 0.740 -> 0.178).

Variants (ablation A/B/C):
- correctness: 0/1.
- calls:       1 + 0.1 * (1 - calls/30)                                   (the blog's LFM recipe)
- efficiency:  1 + 0.05 * (1 - calls/30) + 0.05 * (1 - context/28,672) + 0.03 * (1 - generated/4,096)
  The context term targets the main failure seen in rollouts (loading a whole CSV into the window); among
  correct rollouts it prefers peeking with head/pandas over `read`.
"""

MAX_CALLS = 30
CONTEXT_BUDGET = 28_672  # vLLM max_model_len 32,768 minus the 4,096 output cap
GEN_BUDGET = 4_096


def _frac_left(used, budget):
    return max(0.0, 1.0 - used / budget) if used is not None else 0.0


def reward_value(env_reward, n_tool_calls, *, bonus=0.1, max_calls=MAX_CALLS):
    """Correctness + tool-call bonus (variant `calls`; `bonus=0` gives `correctness`)."""
    if env_reward is None:
        return None
    if float(env_reward) < 1.0:
        return 0.0
    if not bonus or n_tool_calls is None:
        return 1.0
    return 1.0 + bonus * max(0.0, 1.0 - n_tool_calls / max_calls)


def efficiency_value(env_reward, n_tool_calls, final_context, generated_tokens):
    """Variant `efficiency`: correctness + calls/context/generation bonuses, at most +0.13."""
    if env_reward is None:
        return None
    if float(env_reward) < 1.0:
        return 0.0
    return (1.0 + 0.05 * _frac_left(n_tool_calls, MAX_CALLS) + 0.05 * _frac_left(final_context, CONTEXT_BUDGET)
            + 0.03 * _frac_left(generated_tokens, GEN_BUDGET))


def correctness_only(outcome):
    return reward_value(outcome.env_reward, outcome.tool_call_count, bonus=0.0)


def rollout_reward(outcome):
    """Variant `calls` (the original recipe)."""
    return reward_value(outcome.env_reward, outcome.tool_call_count)


def efficiency_reward(outcome):
    return efficiency_value(outcome.env_reward, outcome.tool_call_count,
                            getattr(outcome, "final_context", None), getattr(outcome, "generated_tokens", None))


CONTEXT_FULL_BONUS, CONTEXT_NO_BONUS = 10_000, 15_000


def context_value(env_reward, peak_context, *, bonus=0.1):
    """Variant `context`: 1 + bonus * clip((15k - peak) / 5k, 0, 1) on correct answers, else 0.

    Peak context is the largest prompt + completion over all model calls. The full bonus is paid at
    <= 10k tokens and falls linearly to 0 at 15k, so among correct rollouts the model is pushed toward
    inspecting data instead of loading it into the window. Wrong answers get 0 and no bonus.
    """
    if env_reward is None:
        return None
    if float(env_reward) < 1.0:
        return 0.0
    if peak_context is None:
        return 1.0
    span = CONTEXT_NO_BONUS - CONTEXT_FULL_BONUS
    return 1.0 + bonus * min(1.0, max(0.0, (CONTEXT_NO_BONUS - peak_context) / span))


def context_reward(outcome):
    return context_value(outcome.env_reward, getattr(outcome, "peak_context", None))


def tool_efficiency_bonus(n_tool_calls, *, bonus=0.1, half=15):
    """The multi-harness blog's tool-efficiency term (FineEnvs 05 train/opencode.py): bonus * 15 / (15 + calls).

    1 call -> 0.094, 5 -> 0.075, 15 -> 0.05, 30 -> 0.033. No bonus when the count is unknown or zero.
    """
    if not n_tool_calls:
        return 0.0
    return bonus * half / (half + n_tool_calls)


def tool_value(env_reward, n_tool_calls):
    """Variant `tool` (the multi-harness blog's recipe): correctness * (1 + 0.1 * 15 / (15 + calls)).

    Wrong, missing or out-of-context answers score 0; a correct answer scores 1 plus a tool-efficiency bonus of at
    most 0.1 that favours fewer calls. Context is enforced by the environment's 16k window, not by the reward.
    """
    if env_reward is None:
        return None
    if float(env_reward) < 1.0:
        return 0.0
    return 1.0 + tool_efficiency_bonus(n_tool_calls)


def tool_reward(outcome):
    return tool_value(outcome.env_reward, outcome.tool_call_count)


TOOL_CONTEXT_FULL, TOOL_CONTEXT_NONE = 4_000, 12_000


def context_efficiency_bonus(peak_context, *, bonus=0.1):
    """bonus * clip((12k - peak) / 8k, 0, 1): 4k or less -> 0.1, 8k -> 0.05, 12k or more -> 0. Unknown -> 0."""
    if peak_context is None:
        return 0.0
    span = TOOL_CONTEXT_NONE - TOOL_CONTEXT_FULL
    return bonus * min(1.0, max(0.0, (TOOL_CONTEXT_NONE - peak_context) / span))


def tool_context_value(env_reward, n_tool_calls, peak_context):
    """Variant `tool_context`: correctness * (1 + blog tool bonus + peak-context bonus), at most 1.2.

    With the proxy's 4k tool-output cap, dumping a file with `read` no longer overflows the window, so the outcome no
    longer punishes it (FINDINGS C15). The context term restores a cost among correct rollouts: the one that inspected
    the data with a small peek keeps a smaller window and earns more. Wrong or missing answers still score 0.
    """
    if env_reward is None:
        return None
    if float(env_reward) < 1.0:
        return 0.0
    return 1.0 + tool_efficiency_bonus(n_tool_calls) + context_efficiency_bonus(peak_context)


def tool_context_reward(outcome):
    return tool_context_value(outcome.env_reward, outcome.tool_call_count, getattr(outcome, "peak_context", None))


# Variant `tool_fast` (p6): agent time capped at 600 s (Harbor timeout). On a correct answer: a small time term that is a
# bonus under 5 min and a penalty over it, plus up to -0.1 for generated tokens. Scales from p5 correct rollouts: agent
# seconds p10 26, p50 46, p90 108; generated tokens p10 440, p50 919, p90 2,710.
FAST_PIVOT_SECONDS = 300.0
FAST_TOKENS = (300.0, 3_000.0)


def _ramp(value, lo, hi):
    return 0.0 if value is None else min(1.0, max(0.0, (float(value) - lo) / (hi - lo)))


def time_term(agent_seconds, *, scale=0.05, pivot=FAST_PIVOT_SECONDS):
    """scale * clip((300 - s) / 300, -1, 1): 0 s -> +0.05, 60 s -> +0.04, 300 s -> 0, 450 s -> -0.025, 600 s -> -0.05.

    Unknown time -> 0.
    """
    if agent_seconds is None:
        return 0.0
    return scale * min(1.0, max(-1.0, (pivot - float(agent_seconds)) / pivot))


def token_penalty(generated_tokens, *, penalty=0.1):
    """penalty * clip((tokens - 300) / 2700, 0, 1): 300 or less -> 0, 1650 -> 0.05, 3000 or more -> 0.1."""
    return penalty * _ramp(generated_tokens, *FAST_TOKENS)


def tool_fast_value(env_reward, n_tool_calls, agent_seconds, generated_tokens):
    """Variant `tool_fast`: correctness * (1 + blog tool bonus + time term - token penalty), in [0.85, 1.144].

    The blog's tool bonus already falls with every extra call. All terms apply to correct answers only: a wrong or
    missing answer stays 0 (a penalty there would teach giving up fast), and a slow correct answer (>= 0.85) still
    beats any failure.
    """
    if env_reward is None:
        return None
    if float(env_reward) < 1.0:
        return 0.0
    return 1.0 + tool_efficiency_bonus(n_tool_calls) + time_term(agent_seconds) - token_penalty(generated_tokens)


def tool_fast_reward(outcome):
    return tool_fast_value(outcome.env_reward, outcome.tool_call_count, getattr(outcome, "agent_seconds", None),
                           getattr(outcome, "generated_tokens", None))


# Variant `tool_ctx` (multi-harness, 24k window): on a correct answer, the blog's tool bonus plus a flat bonus when the
# model's own context stayed small. "Own" = peak context minus the harness's first prompt (system prompt + tool schemas
# + question: Pi ~870, mini-swe-agent ~945, Codex ~6,330 tokens), so a heavy harness is not penalised for its overhead.
# 15k own tokens ~ a 16k total window for Pi / mini-swe-agent.
OWN_CONTEXT_LIMIT = 15_000


def own_context_bonus(peak_context, first_prompt, *, bonus=0.1, limit=OWN_CONTEXT_LIMIT):
    """bonus if peak_context - first_prompt <= limit, else 0. Unknown -> 0."""
    if peak_context is None or first_prompt is None:
        return 0.0
    return bonus if peak_context - first_prompt <= limit else 0.0


def tool_ctx_value(env_reward, n_tool_calls, peak_context, first_prompt):
    """Variant `tool_ctx`: correctness * (1 + blog tool bonus + own-context bonus), in [1.0, 1.2).

    Correct within the limit: 1.1 + 0.1 * 15 / (15 + calls); correct but over it: 1 + tool bonus; wrong or missing: 0.
    """
    if env_reward is None:
        return None
    if float(env_reward) < 1.0:
        return 0.0
    return 1.0 + tool_efficiency_bonus(n_tool_calls) + own_context_bonus(peak_context, first_prompt)


def tool_ctx_reward(outcome):
    return tool_ctx_value(outcome.env_reward, outcome.tool_call_count, getattr(outcome, "peak_context", None),
                          getattr(outcome, "first_prompt", None))


REWARDS = {"correctness": correctness_only, "calls": rollout_reward, "efficiency": efficiency_reward,
           "context": context_reward, "tool": tool_reward, "tool_context": tool_context_reward,
           "tool_fast": tool_fast_reward, "tool_ctx": tool_ctx_reward}
