"""LFM2 packed training: the short convolutions must restart at every packed sequence boundary."""


def enable_packed_convolutions(peft_model):
    import torch
    from transformers import KernelConfig

    base = peft_model.get_base_model()
    base.train()
    # mamba-ssm causal_conv1d honours seq_idx; the PyTorch fallback silently leaks across sequences.
    base.set_use_kernels(
        True,
        kernel_config=KernelConfig(
            kernel_mapping={
                n: (f"kernels-community/mamba-ssm:{n}", {"version": 2})
                for n in ("causal_conv1d_fn", "causal_conv1d_update")
            },
            inherit_mapping=False,
        ),
    )

    def add_seq_idx(module, args, kwargs):
        pos = kwargs.get("position_ids")
        if pos is not None:
            kwargs["seq_idx"] = ((pos == 0).cumsum(-1) - 1).to(torch.int32)
        return args, kwargs

    base.model.register_forward_pre_hook(add_seq_idx, with_kwargs=True)
    # Abort if packing is not exact: two packed sequences must equal the second one run alone.
    with torch.no_grad():
        ids = torch.tensor([[10, 20, 30, 40, 50, 60, 70, 80]], device=base.device)
        alone = base(input_ids=ids[:, 4:], position_ids=torch.arange(4, device=ids.device)[None], use_cache=False).logits
        packed = base(input_ids=ids, position_ids=torch.arange(4, device=ids.device).repeat(2)[None], use_cache=False).logits[:, 4:]
        torch.testing.assert_close(packed, alone, atol=0.05, rtol=0.01)
    base.train()
