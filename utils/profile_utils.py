import torch
from torch.cuda import nvtx

import traceback as tb
import numpy as np
import csv, os


# Do all of the fusion heuristics before timing.
# https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html#demonstrating-speedups.
WARMUP_REPS = 2
PROFILE_REPS = 10
NREPS = WARMUP_REPS + PROFILE_REPS
NCU_WARMUP_REPS = 5

def get_dtype(dtype_flag: str):
    dtype_map = {
        # 8: torch.float8_e5m2, # No support for float8 yet.
        "b16": torch.bfloat16,
        "16": torch.float16,
        "32": torch.float32
    }
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
    results = [0] * NREPS

    for i in range(NREPS):
        torch.cuda.empty_cache()
        starts[i].record()
        results[i] = fn(*args)
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
    res = [0] * (NCU_WARMUP_REPS + 1)
    torch.cuda.empty_cache()
    try:
        with torch.no_grad():
            for i in range(NCU_WARMUP_REPS):
                res[i] = fn(*args)

            torch.cuda.cudart().cudaProfilerStart()
            nvtx.range_push("profile_range")
            res[-1] = fn(*args)
            nvtx.range_pop()
            torch.cuda.cudart().cudaProfilerStop()

    except Exception as e:
        print("profile_rep crashed!")
        tb.print_exc()
