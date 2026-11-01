"""Batch size from the free GPU memory (TODO 2): measure one step at batch 1 and 2, then fit the largest batch that fits."""
import torch


# Free GPU memory in bytes (the device-level reading); None without CUDA.
def free_gpu_bytes():
    if not torch.cuda.is_available():
        return None
    return torch.cuda.mem_get_info()[0]


# Largest batch with fraction * free - fixed_bytes <= batch * per_item_bytes, kept within [floor, cap].
def batch_from_memory(per_item_bytes, fixed_bytes=0, free_bytes=None, fraction=0.8, cap=None, floor=1):
    if free_bytes is None:
        free_bytes = free_gpu_bytes()
    if free_bytes is None:
        raise RuntimeError("batch_from_memory needs free_bytes or a CUDA device")
    per_item = max(per_item_bytes, 1)
    batch = max(floor, int((fraction * free_bytes - fixed_bytes) // per_item))
    if cap is not None:
        batch = min(cap, batch)
    return batch


# Peak allocated memory of step_fn(1) and step_fn(2): per item = m2 - m1, fixed = m1 - per item. Returns (fixed, per_item).
def measure_per_item(step_fn):
    readings = []
    for b in (1, 2):
        torch.cuda.reset_peak_memory_stats()
        step_fn(b)
        torch.cuda.synchronize()
        readings.append(torch.cuda.max_memory_allocated())
        torch.cuda.empty_cache()
    per_item = readings[1] - readings[0]
    return readings[0] - per_item, per_item


# Batch for one step. The memory the models already hold is not free any more, so it is taken out of the fixed cost;
# extra_fixed_bytes adds what the step does not allocate yet (the optimizer state). One step_fn(b) call runs
# items_per_call * b items. floor is the smallest batch returned. Returns (batch in items, free, fixed, per_item).
def auto_batch(step_fn, cap=None, extra_fixed_bytes=0, fraction=0.8, items_per_call=1, floor=1):
    base = torch.cuda.memory_allocated()
    fixed, per_call = measure_per_item(step_fn)
    per_item = per_call / items_per_call
    torch.cuda.empty_cache()
    free = free_gpu_bytes()
    fixed = max(0, fixed - base) + extra_fixed_bytes
    batch = batch_from_memory(per_item, fixed_bytes=fixed, free_bytes=free, fraction=fraction, cap=cap, floor=floor)
    return batch, free, fixed, per_item
