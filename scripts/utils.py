import os, sys
import numpy as np
import torch

# Assume warmup reps = nreps = 10.
NREPS = 10

class HiddenPrints:
    """
    A class that suppresses print statements. Use inside of a context manager.
    """
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


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
