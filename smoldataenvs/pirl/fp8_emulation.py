"""Make the BF16 trainer compute what vLLM's online FP8 (W8A8) LFM2 computes, so sampler and trainer agree.

vLLM `--quantization fp8` on a BF16 checkpoint (Fp8PerTensorOnlineLinearMethod, vLLM 0.25):
  * weights: one FP8-e4m3 scale per *fused* linear (amax/448): q/k/v share one (qkv_proj), w1/w3 share one
    (w13); attention out_proj, w2 and the short-conv in_proj/out_proj each have their own;
  * activations: dynamic per-token FP8 scales (CUTLASS path on SM89);
  * not quantized: embeddings / tied LM head, norms, the depthwise convolution, attention itself;
  * LoRA runs in BF16 on the *unquantized* layer input and is added to the FP8 base output.
Here: each base weight is replaced once by its FP8 quantize-dequantize value with vLLM's scale groups, and a
pre-hook FP8-rounds each base layer's input per token (straight-through gradient). The LoRA branch is left
untouched. FP8 products are emulated in BF16, whose rounding (2^-9) is far below FP8's (2^-4).
"""

import torch

FP8 = torch.float8_e4m3fn
FP8_MAX = 448.0
# vLLM's fused linears for LFM2 (vllm/model_executor/models/lfm2.py packed_modules_mapping)
GROUPS = (("self_attn", ("q_proj", "k_proj", "v_proj")), ("self_attn", ("out_proj",)),
          ("feed_forward", ("w1", "w3")), ("feed_forward", ("w2",)),
          ("conv", ("in_proj",)), ("conv", ("out_proj",)))


def _qdq(x, scale):
    return (x.float() / scale).clamp(-FP8_MAX, FP8_MAX).to(FP8).float().mul(scale).to(x.dtype)


class _PerTokenFP8(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        scale = x.float().abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / FP8_MAX
        return _qdq(x, scale)

    @staticmethod
    def backward(ctx, grad):
        return grad  # straight-through: the base weights are frozen, only LoRA (upstream) needs this


def _base_linear(module):
    return getattr(module, "base_layer", module)  # PEFT LoRA wrapper -> the frozen nn.Linear


def _quantize_input(module, args, kwargs):
    return (_PerTokenFP8.apply(args[0]),) + tuple(args[1:]), kwargs


@torch.no_grad()
def emulate_vllm_fp8(peft_model):
    """Quantize-dequantize base weights in place and hook activation quantization. Returns #linears done."""
    layers = peft_model.get_base_model().model.layers
    done = 0
    for layer in layers:
        for block_name, names in GROUPS:
            block = getattr(layer, block_name, None)
            if block is None:
                continue
            mods = [_base_linear(getattr(block, n)) for n in names if hasattr(block, n)]
            if not mods:
                continue
            amax = max(m.weight.abs().amax().float() for m in mods)
            scale = amax.clamp(min=1e-12) / FP8_MAX
            for m in mods:
                m.weight.copy_(_qdq(m.weight, scale))
                m.register_forward_pre_hook(_quantize_input, with_kwargs=True)
                done += 1
    return done


class _VllmFP8Matmul(torch.autograd.Function):
    """Forward = vLLM's own kernels (per-token FP8 act quant + CUTLASS scaled_mm); backward = BF16 with W_deq."""

    @staticmethod
    def forward(ctx, x, wq, w_scale):
        from vllm import _custom_ops as ops

        shape = x.shape
        x2 = x.reshape(-1, shape[-1]).to(torch.bfloat16).contiguous()
        xq, xs = ops.scaled_fp8_quant(x2, use_per_token_if_dynamic=True)
        y = ops.cutlass_scaled_mm(xq, wq.t(), scale_a=xs, scale_b=w_scale, out_dtype=torch.bfloat16)
        ctx.save_for_backward(wq, w_scale)
        return y.reshape(*shape[:-1], wq.shape[0])

    @staticmethod
    def backward(ctx, grad):
        wq, w_scale = ctx.saved_tensors
        w = wq.to(torch.bfloat16) * w_scale.to(torch.bfloat16)
        return (grad.to(torch.bfloat16) @ w), None, None


def _vllm_forward(module):
    def forward(x, *args, **kwargs):
        y = _VllmFP8Matmul.apply(x, module.fp8_weight, module.fp8_scale)
        return y + module.bias if module.bias is not None else y
    return forward


@torch.no_grad()
def use_vllm_fp8_kernels(peft_model):
    """Replace each frozen base linear by vLLM's exact online-FP8 computation (same quantizer, scale groups and
    CUTLASS kernel as the sampler); the BF16 base weights are freed. LoRA branches are untouched."""
    from vllm import _custom_ops as ops

    layers = peft_model.get_base_model().model.layers
    done = 0
    for layer in layers:
        for block_name, names in GROUPS:
            block = getattr(layer, block_name, None)
            if block is None:
                continue
            mods = [_base_linear(getattr(block, n)) for n in names if hasattr(block, n)]
            if not mods:
                continue
            fused = torch.cat([m.weight.to(torch.bfloat16) for m in mods], dim=0)
            wq, scale = ops.scaled_fp8_quant(fused, scale=None)  # one per-tensor scale per vLLM fused linear
            row = 0
            for m in mods:
                n = m.weight.shape[0]
                m.register_buffer("fp8_weight", wq[row:row + n].contiguous(), persistent=False)
                m.register_buffer("fp8_scale", scale.clone(), persistent=False)
                row += n
                m.weight = torch.nn.Parameter(torch.empty(0, dtype=torch.bfloat16, device=wq.device), requires_grad=False)
                m.forward = _vllm_forward(m)
                done += 1
            del fused
    torch.cuda.empty_cache()
    return done
