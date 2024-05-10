import torch
from torch.cuda import nvtx

import traceback as tb
import numpy as np
import csv, os


WARMUP_REPS = 10
NREPS = 30

def get_dtype(dtype_flag: int):
    dtype_map = {
        # 8: torch.float8_e5m2, # No support for float8 yet.
        161: torch.bfloat16,
        162: torch.float16,
        32: torch.float32
    }
    if dtype_flag not in dtype_map:
        print("dtype wasn't specified, defaulting to torch.float32")

    return dtype_map.get(dtype_flag, torch.float32)

def _time_iter(model, *args):
    """
    Time in ms.
    """
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = model(*args)
    end.record()
    torch.cuda.synchronize()
    return result, start.elapsed_time(end)

def time_model(model, *args):
    """
    Returns the median runtime in ms.
    
    Based on:
    https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html#demonstrating-speedups

    Could consider:
    https://www.speechmatics.com/company/articles-and-news/timing-operations-in-pytorch
    """
    torch.cuda.empty_cache()
    times = []
    # Warmup
    # Do all of the fusion heuristics, so the later call won't need to.
    for _ in range(NREPS):
        _ = model(*args)

    # Actual eval.
    for _ in range(NREPS):
        _, time = _time_iter(model, *args)
        times.append(time)
    return np.median(np.array(times))

def _time_model(model, *args):
    """
    Time in ms.
    """
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = model(*args)
    end.record()
    torch.cuda.synchronize()
    return result, start.elapsed_time(end)

def time_addmm(A, B, C = None):
    """
    Returns the median runtime in ms.
    C = None if we don't use bias.

    Based on:
    https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html#demonstrating-speedups

    Could consider:
    https://www.speechmatics.com/company/articles-and-news/timing-operations-in-pytorch
    """

    if C is not None:
        @torch.compile(backend="inductor")
        def addmm(a, b, bias):
            return torch.addmm(bias, a, b)

        fn = addmm
        args = (A, B, C)
    else:
        @torch.compile(backend="inductor")
        def mm(a, b):
            return torch.mm(a, b)

        fn = mm
        args = (A, B)

    torch.cuda.empty_cache()
    # Do all of the fusion heuristics, so the later call won't need to.
    for _ in range(WARMUP_REPS):
        _ = fn(*args)

    times = []
    # Actual eval.
    for _ in range(NREPS):
        _, time = _time_model(fn, *args)
        times.append(time)
    return np.median(np.array(times))

def time_conv2d(input, weight):
    """
    Returns the median runtime in ms.
    C = None if we don't use bias.

    Based on:
    https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html#demonstrating-speedups

    Could consider:
    https://www.speechmatics.com/company/articles-and-news/timing-operations-in-pytorch
    """

    # Define the convolution operation
    @torch.compile(backend="inductor")
    def conv2d(x, w):
        return F.conv2d(x, w, bias=bias_tensor, stride=stride, padding=padding, dilation=dilation, groups=groups)


    torch.cuda.empty_cache()
    # Do all of the fusion heuristics, so the later call won't need to.
    for _ in range(WARMUP_REPS):
        _ = fn(*args)

    times = []
    # Actual eval.
    for _ in range(NREPS):
        _, time = _time_model(fn, *args)
        times.append(time)
    return np.median(np.array(times))

def save_row(kernel_params, time, out_file):
    with open(out_file, mode='a', newline='') as file:
        writer = csv.writer(file)

        if os.path.getsize(out_file) == 0:
            writer.writerow(['Kernel Name', 'Latency (ms)'])
        writer.writerow([kernel_params, time])

def profile_rep(fn, args):
    res = [0] * (WARMUP_REPS + 1)
    torch.cuda.empty_cache()
    try:
        with torch.no_grad():
            for i in range(WARMUP_REPS):
                res[i] = fn(*args)
            
            torch.cuda.cudart().cudaProfilerStart()
            nvtx.range_push("profile_range")
            res[-1] = fn(*args)
            nvtx.range_pop()
            torch.cuda.cudart().cudaProfilerStop()

    except Exception as e:
        print("profile_rep crashed!")
        tb.print_exc()
