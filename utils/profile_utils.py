import torch
import numpy as np

WARMUP_REPS = 10
NREPS = 30

def get_precision(precision_flag: int):
    precision_map = {
        # 8: torch.float8_e5m2, # No support for float8 yet.
        161: torch.bfloat16,
        162: torch.float16,
        32: torch.float32
    }
    if precision_flag not in precision_map:
        print("Precision wasn't specified, defaulting to torch.float32")

    return precision_map.get(precision_flag, torch.float32)

def _time_iter(model, input):
    """
    Time in ms.
    """
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = model(input)
    end.record()
    torch.cuda.synchronize()
    return result, start.elapsed_time(end)

def time_model(model, warmup_input, input):
    """
    Returns the median runtime in ms.
    
    Based on:
    https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html#demonstrating-speedups

    Could consider:
    https://www.speechmatics.com/company/articles-and-news/timing-operations-in-pytorch
    """
    times = []
    # Warmup
    # Do all of the fusion heuristics, so the later call won't need to.
    for _ in range(NREPS):
        _ = model(warmup_input)

    # Actual eval.
    for _ in range(NREPS):
        _, time = _time_iter(model, input)
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

    @torch.compile(backend="inductor")
    def addmm(a, b, bias):
        return torch.addmm(bias, a, b)

    @torch.compile(backend="inductor")
    def mm(a, b):
        return torch.mm(a, b)

    if C is not None:
        fn = addmm
        args = (A, B, C)
    else:
        fn = mm
        args = (A, B)

    # Do all of the fusion heuristics, so the later call won't need to.
    for _ in range(WARMUP_REPS):
        _ = fn(*args)

    times = []
    # Actual eval.
    for _ in range(NREPS):
        _, time = _time_model(fn, *args)
        times.append(time)
    return np.median(np.array(times))

