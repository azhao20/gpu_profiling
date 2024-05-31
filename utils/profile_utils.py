import torch
from torch.cuda import nvtx

import traceback as tb
import numpy as np
import csv, os
import math
# This was hard-coded for nvidia a100s. Could also just try allocating
# torch.zeros(1, device="cuda"), then checking difference in mem usage
_PYTORCH_MIN_ALLOCATE = 512

# Do all of the fusion heuristics before timing.
# https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html#demonstrating-speedups.
WARMUP_REPS = 2
PROFILE_REPS = 10
NREPS = WARMUP_REPS + PROFILE_REPS
NCU_WARMUP_REPS = 5

# (b)mm parameters.
mm_sizes = [i for i in range(16, 512, 16)] + [i for i in range(512, 2048, 128)] + \
           [i for i in range(2048, 4096, 512)] + [i for i in range(4096, 32768 + 1, 1024)]
mm_batches=[i for i in range(32, 512+1, 32)]

dtype_map = {
    # 8: torch.float8_e5m2, # No support for float8 yet.
    "b16": torch.bfloat16,
    "16": torch.float16,
    "32": torch.float32
}

def get_dtype(dtype_flag: str):
    if dtype_flag not in dtype_map:
        print("dtype wasn't specified, defaulting to torch.float32")

    return dtype_map.get(dtype_flag, torch.float32)

def _time_fn(fn, *args):
    """
    Returns the median runtime in ms.

    Based on:
    https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html#demonstrating-speedups

    Could consider:
    https://www.speechmatics.com/company/articles-and-news/timing-operations-in-pytorch
    """
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(NREPS)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(NREPS)]

    torch.cuda.empty_cache()
    for i in range(NREPS):
        starts[i].record()
        result = fn(*args)
        ends[i].record()
    torch.cuda.synchronize()

    # Discard the warmup reps.
    start_ends = zip(starts[WARMUP_REPS:], ends[WARMUP_REPS:])
    times = np.array([start.elapsed_time(end) for start, end in start_ends])
    return times

def time_fn(fn, *args):
    return np.median(_time_fn(fn, *args))

def save_row(kernel_params, time, out_file):
    """
    Helper function to save the times to a csv.
    """
    with open(out_file, mode='a', newline='') as file:
        writer = csv.writer(file)

        if os.path.getsize(out_file) == 0:
            writer.writerow(['Kernel Name', 'Latency (ms)'])
        writer.writerow([kernel_params, time])

def profile_rep(fn, *args):
    torch.cuda.empty_cache()
    try:
        with torch.no_grad():
            for _ in range(NCU_WARMUP_REPS):
                res = fn(*args)

            torch.cuda.cudart().cudaProfilerStart()
            nvtx.range_push("profile_range")
            res = fn(*args)
            nvtx.range_pop()
            torch.cuda.cudart().cudaProfilerStop()

    except Exception as e:
        print("profile_rep crashed!")
        tb.print_exc()

def check_size(dtype: torch.dtype, *tensor_sizes):
    """
    Only use up to 80% of available memory.

    Computing element_size in this way is more robust than using a map.
    """
    required_memory = 0
    element_size = torch.tensor([], dtype=dtype).element_size()

    for tensor_size in tensor_sizes:
        tensor_bytes = math.prod(tensor_size) * element_size
        required_memory += math.ceil(tensor_bytes / _PYTORCH_MIN_ALLOCATE) * _PYTORCH_MIN_ALLOCATE

    torch.cuda.empty_cache()
    max_bytes = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved(0)) * 0.8
    return required_memory <= max_bytes
