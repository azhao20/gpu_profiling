import torch
import argparse
from torch.cuda import nvtx

import traceback as tb
import numpy as np
import csv, os
import math

# (b)mm parameters. Included here because shared across bmm and mm.
# Up to 512: multiples of 16.
# 512-2048: multiples of 128
# 2048-4096: multiples of 512
# 4096-2^15 = 32768: multiples of 1024
mm_sizes = [i for i in range(16, 512, 16)] + [i for i in range(512, 2048, 128)] + \
           [i for i in range(2048, 4096, 512)] + [i for i in range(4096, 32768 + 1, 1024)]
# mm_sizes = [10**9]
# mm_sizes = [i for i in range(16, 128, 16)]

def get_args_mm():
    """
    This is used for both mm and bmm.
    dtype, m, and p are unspecified for timing, but n is required.

    Note: we don't profile with bias for now.
    """
    parser = argparse.ArgumentParser(description="(Batch) Matrix multiplication (mm/bmm).")
    parser.add_argument("--mode", type=str, required=True, choices=["profile", "time"], help="Profile or time.")
    parser.add_argument("--use_inductor", action="store_true", help="Should lower the function using inductor.")
    parser.add_argument("--dtype", type=str, required=False, choices=["32", "b16", "16"], help="Data type flag.")
    parser.add_argument("--b", type=int, required=False, help="See https://pytorch.org/docs/stable/generated/torch.bmm.html.")
    parser.add_argument("--n", type=int, required=True, help="See https://pytorch.org/docs/stable/generated/torch.mm.html.")
    parser.add_argument("--m", type=int, required=False, help="See https://pytorch.org/docs/stable/generated/torch.mm.html.")
    parser.add_argument("--p", type=int, required=False, help="See https://pytorch.org/docs/stable/generated/torch.mm.html.")
    parser.add_argument("--out_file", type=str, required=False, help="Path to the output CSV file.")
    # parser.add_argument("--bias", type=int, required=True, choices=[0, 1], help="Use bias (1) or not (0).")
    args = parser.parse_args()

    if args.mode == "time" and not args.out_file:
        raise ValueError("Time requires an outfile")
    return args

class ProfileBase:
    """
    A class used for timing and profiling GPU operators.
    """
    def __init__(self,
                 pytorch_min_allocate: int = 512,
                 warmup_reps: int = 2,
                 profile_reps: int = 10,
                 ncu_warmup_reps: int = 5):
        """
        pytorch_min_allocate = 512 was hard-coded for nvidia a100s. Could try allocating
        torch.zeros(1, device="cuda"), then checking difference in mem usage.

        TODO: for more GPUs, map {<gpu type>: <min_allocate>}
        """
        # Assume this script runs on one device.
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        assert(self.device == "cuda:0")

        self.dtype_map = {
            # 8: torch.float8_e5m2, # No support for float8 yet.
            "b16": torch.bfloat16,
            "16": torch.float16,
            "32": torch.float32
        }
        self.time_header = ['Kernel Name', 'Latency (ms)']
        self._PYTORCH_MIN_ALLOCATE = pytorch_min_allocate

        # Do all of the fusion heuristics before timing.
        # https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html#demonstrating-speedups.
        self.WARMUP_REPS = warmup_reps
        self.PROFILE_REPS = profile_reps
        self.NREPS = self.WARMUP_REPS + self.PROFILE_REPS
        self.NCU_WARMUP_REPS = ncu_warmup_reps

    def get_sizes(self, args) -> list:
        raise Exception("Not Implemented")

    def get_fn(self, use_inductor: bool):
        raise Exception("Not Implemented")

    def time(self, args):
        raise Exception("Not Implemented")

    def get_dtype(self, dtype: str):
        if dtype not in self.dtype_map:
            raise ValueError("dtype isn't supported. Consider adding it to self.dtype_map.")
        return self.dtype_map[dtype]

    def check_size(self, dtype: torch.dtype, tensor_sizes: list):
        """
        Only use up to 80% of available memory.

        Computing element_size in this way is more robust than using a map.
        """
        required_memory = 0
        element_size = torch.tensor([], dtype=dtype).element_size()

        for tensor_size in tensor_sizes:
            tensor_bytes = math.prod(tensor_size) * element_size
            required_memory += math.ceil(tensor_bytes / self._PYTORCH_MIN_ALLOCATE) * self._PYTORCH_MIN_ALLOCATE

        torch.cuda.empty_cache()
        max_bytes = 0.8 * (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved(0))
        return required_memory <= max_bytes

    def get_inputs(self, dtype, sizes):
        """
        Generate random tensors based on provided sizes.

        sizes: A list of of `torch.Size`s.
        """
        return [torch.randn(size, dtype=dtype, device=self.device) for size in sizes]

    def _time_fn(self, fn, *inputs):
        """
        Based on:
        https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html#demonstrating-speedups

        Could consider:
        https://www.speechmatics.com/company/articles-and-news/timing-operations-in-pytorch
        """
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(self.NREPS)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(self.NREPS)]

        torch.cuda.empty_cache()
        for i in range(self.NREPS):
            starts[i].record()
            result = fn(*inputs)
            ends[i].record()
        torch.cuda.synchronize()

        # Discard the warmup reps. It might be cleaner to select [self.WARMUP_REPS:] from times, but if
        # WARMUP_REPS is large, avoids computing unnecessary start times.
        start_ends = zip(starts[self.WARMUP_REPS:], ends[self.WARMUP_REPS:])
        times = np.array([start.elapsed_time(end) for start, end in start_ends])
        return times

    def time_fn(self, fn, *inputs):
        """
        Returns the median runtime in ms.
        """
        return np.median(self._time_fn(fn, *inputs))

    def time_rep(self, args):
        """
        Returns time the kernel took to run. If there isn't
        enough memory to store create the fn inputs, time = np.nan.
        """
        dtype = self.get_dtype(args.dtype)
        sizes = self.get_sizes(args)
        if not self.check_size(dtype, sizes):
            # Flag as incomplete.
            return np.nan

        inputs = self.get_inputs(dtype, sizes)
        fn = self.get_fn(args)
        time = self.time_fn(fn, *inputs)
        return time

    # Hardware profiling functions
    def profile_rep(self, fn, *inputs):
        """
        TODO: could allow user to specify range other than "profile_range"
        """
        torch.cuda.empty_cache()
        try:
            with torch.no_grad():
                for _ in range(self.NCU_WARMUP_REPS):
                    res = fn(*inputs)

                torch.cuda.cudart().cudaProfilerStart()
                nvtx.range_push("profile_range")
                res = fn(*inputs)
                nvtx.range_pop()
                torch.cuda.cudart().cudaProfilerStop()

        except Exception as e:
            print("profile_rep crashed!")
            tb.print_exc()

    def profile(self, args):
        """
        get_sizes, get_inputs, and get_fn are operator-specific.
        
        TODO: decide how to handle check_size failures.
        """
        dtype = self.get_dtype(args.dtype)
        sizes = self.get_sizes(args)
        if not self.check_size(dtype, sizes):
            return

        inputs = self.get_inputs(dtype, sizes)
        fn = self.get_fn(args)
        self.profile_rep(fn, *inputs)

    """
    TODO: this is deprecated. Delete!
    """
    # def save_row(self, kernel_params, time, out_file):
    #     """
    #     Helper function to save the times to a csv.
    #     """
    #     with open(out_file, mode='a', newline='') as file:
    #         writer = csv.writer(file)

    #         if os.path.getsize(out_file) == 0:
    #             writer.writerow(['Kernel Name', 'Latency (ms)'])
    #         writer.writerow([kernel_params, time])

# class ProfileTimeBase(ProfileBase):

# class Profiler(Base)
