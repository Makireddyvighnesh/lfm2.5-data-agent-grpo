"""Shared settings for Pi-harness GRPO on one L4: model identity and the non-thinking template."""

from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "prepared"
MODEL = "LiquidAI/LFM2.5-2.6B"
REVISION = "654f9463ce32b05d0429d76fe1f580b27d4c1ac0"

# Token-faithful sampling. Same values for rollouts, training and evaluation.
TEMPERATURE = 1.0
TOP_P = 1.0
TOP_K = -1
MAX_TOKENS = 4096  # per model call; same cap in training and evaluation (blog fix #3)

LORA_NAME = "trl-policy"

# Render assistant history as "<think></think>..." so multi-turn prompts extend each other exactly.
# PIRL_PREFIX_CHAIN=0 restores the tutorial's template (ablation).
import os as _os

PREFIX_CHAIN_TEMPLATE = _os.environ.get("PIRL_PREFIX_CHAIN", "1") != "0"  # AsyncGRPOTrainer publishes adapters as f"{LORA_NAME}-v{N}"
# PIRL_THINKING=1 keeps LFM's native "<think>" generation prefix (reason, then act) and re-renders past assistant turns
# as "<think>" + content, so each turn's prompt still extends the previous prompt + completion. Default: thinking off.
THINKING = _os.environ.get("PIRL_THINKING", "0") == "1"


def tokenizer_for(model=MODEL):
    from transformers import AutoTokenizer
    from trl.chat_template_utils import lfm2_2_5_template

    tokenizer = AutoTokenizer.from_pretrained(model, revision=REVISION)
    # LFM's template opens <think> even when enable_thinking=False; close it immediately.
    start = r'{{- "<|im_start|>assistant\n<think>" -}}'
    if tokenizer.chat_template.count(start) != 1:
        raise ValueError("LFM template changed; check its non-thinking generation prefix")
    if THINKING:
        history = '{%- if thinking and keep_thinking -%}\n            {{- "<think>" + thinking + "</think>" -}}\n        {%- endif -%}'
        if tokenizer.chat_template.count(history) != 1:
            raise ValueError("LFM template changed; check how assistant history renders thinking")
        tokenizer.chat_template = tokenizer.chat_template.replace(
            history,
            '{%- if thinking and keep_thinking -%}\n            {{- "<think>" + thinking + "</think>" -}}\n'
            '        {%- elif message.content and "</think>" in parse_content(message.content) -%}\n'
            '            {{- "<think>" -}}\n        {%- endif -%}',
        )
        tokenizer.response_template = deepcopy(lfm2_2_5_template)
        return tokenizer
    tokenizer.chat_template = tokenizer.chat_template.replace(
        start, r'{{- "<|im_start|>assistant\n<think></think>" -}}'
    )
    if PREFIX_CHAIN_TEMPLATE:
        # Re-render past assistant turns with the same empty think block the model was prompted with.
        # Without it every agent turn's prompt diverges from the previous prompt + completion right
        # after "<|im_start|>assistant\n", so OpenEnv cannot chain turns: each turn becomes its own
        # training row repeating the whole context (per_turn_capture_only), ~N times the tokens.
        history = '{%- if thinking and keep_thinking -%}\n            {{- "<think>" + thinking + "</think>" -}}\n        {%- endif -%}'
        if tokenizer.chat_template.count(history) != 1:
            raise ValueError("LFM template changed; check how assistant history renders thinking")
        tokenizer.chat_template = tokenizer.chat_template.replace(
            history,
            '{%- if thinking and keep_thinking -%}\n            {{- "<think>" + thinking + "</think>" -}}\n'
            '        {%- else -%}\n            {{- "<think></think>" -}}\n        {%- endif -%}',
        )
    tokenizer.response_template = deepcopy(lfm2_2_5_template)
    return tokenizer
