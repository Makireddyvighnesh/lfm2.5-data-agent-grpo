"""Runtime fix for vLLM 0.25 FP8 KV wake-up on LFM2's hybrid cache layout."""
from vllm.v1.worker.gpu_worker import Worker
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

_original_init_scales = GPUModelRunner.init_fp8_kv_scales

def _flatten(values):
    for value in values:
        if isinstance(value, (list, tuple)):
            yield from _flatten(value)
        else:
            yield value

def init_fp8_kv_scales(self):
    # Upstream assumes every cache is a tensor. Hybrid short-conv state is
    # a list of tensors. Let the original code clear each tensor and restore
    # attention scales, then restore the model runner's original structure.
    caches = self.kv_caches
    self.kv_caches = list(_flatten(caches))
    try:
        return _original_init_scales(self)
    finally:
        self.kv_caches = caches

class PatchedWorker(Worker):
    def __init__(self, *args, **kwargs):
        GPUModelRunner.init_fp8_kv_scales = init_fp8_kv_scales
        super().__init__(*args, **kwargs)
