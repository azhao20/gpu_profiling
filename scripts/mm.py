import os, csv
import argparse
import torch
import numpy as np

from utils.profile_utils import get_dtype, time_fn, profile_rep, check_size, dtype_map, mm_sizes

import warnings
warnings.filterwarnings("ignore")

# Assume this script runs on one device.
device = "cuda:0" if torch.cuda.is_available() else "cpu"
assert(device == "cuda:0")

def get_args():
    """
    Note: we don't profile with bias for now.
    dtype, m, and p are unspecified for timing.
    """
    parser = argparse.ArgumentParser(description="Matrix multiplication (mm).")
    parser.add_argument("--mode", type=str, required=True, choices=["profile", "time"], help="Profile or time.")
    parser.add_argument("--use_inductor", action="store_true", help="Should lower the function using inductor.")
    parser.add_argument("--dtype", type=str, required=False, choices=["32", "b16", "16"], help="Data type flag.")
    parser.add_argument("--n", type=int, required=True, help="See https://pytorch.org/docs/stable/generated/torch.mm.html.")
    parser.add_argument("--m", type=int, required=False, help="See https://pytorch.org/docs/stable/generated/torch.mm.html.")
    parser.add_argument("--p", type=int, required=False, help="See https://pytorch.org/docs/stable/generated/torch.mm.html.")
    parser.add_argument("--out_file", type=str, required=False, help="Path to the output CSV file.")
    # parser.add_argument("--bias", type=int, required=True, choices=[0, 1], help="Use bias (1) or not (0).")
    args = parser.parse_args()

    if args.mode == "time" and not args.out_file:
        raise ValueError("Time requires an outfile")

    return args

def get_fn(use_inductor):
    if use_inductor:
        @torch.compile(backend="inductor")
        def mm(a, b):
            return torch.mm(a, b)
        fn = mm
    else:
        fn = torch.mm
    return fn

def get_sizes(args):
    dtype = get_dtype(args.dtype)
    A_size = torch.Size([args.n, args.m])
    B_size = torch.Size([args.m, args.p])
    return dtype, A_size, B_size

def get_inputs(dtype, A_size, B_size):
    A = torch.randn(A_size, dtype=dtype, device=device)
    B = torch.randn(B_size, dtype=dtype, device=device)
    return A, B

def profile(args):
    dtype_sizes = get_sizes(args)
    if not check_size(*dtype_sizes):
        return

    A, B = get_inputs(*dtype_sizes)
    fn = get_fn(args.use_inductor)
    profile_rep(fn, A, B)

def time_rep(args):
    """
    Returns kernel_params, the identifier for joining on profile results,
    and the time the kernel took to run, as computed by time_fn. If there
    isn't enough memory to store create the fn inputs, as checked in
    get_sizes(), `time = np.nan`.
    """
    dtype_sizes = get_sizes(args)
    kernel_params=f"{args.dtype}.{args.n}.{args.m}.{args.p}"
    if not check_size(*dtype_sizes):
        # Flag as incomplete.
        return [kernel_params, np.nan]

    A, B = get_inputs(*dtype_sizes)
    fn = get_fn(args.use_inductor)
    time = time_fn(fn, A, B)
    return [kernel_params, time]

def time(args):
    with open(args.out_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        if os.path.getsize(args.out_file) == 0:
            writer.writerow(['Kernel Name', 'Latency (ms)'])
        for dname in dtype_map:
            # print(f"{dname}------------------------------")
            for m in mm_sizes:
                for p in mm_sizes:
                    args.m = m
                    args.p = p
                    args.dtype = dname
                    writer.writerow(time_rep(args))

            file.flush()

def main():
    args = get_args()
    if args.mode == "time":
        time(args)
    else:
        profile(args)

if __name__ == "__main__":
    main()
