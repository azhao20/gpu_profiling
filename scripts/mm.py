import argparse
import torch
import numpy as np

from utils.profile_utils import get_dtype, time_fn, save_row, profile_rep, check_size

import warnings
warnings.filterwarnings("ignore")

# Assume this script runs on one device.
device = "cuda:0" if torch.cuda.is_available() else "cpu"
assert(device == "cuda:0")

def get_args():
    """
    Note: we don't profile with bias for now.
    """
    parser = argparse.ArgumentParser(description="Matrix multiplication (mm).")
    parser.add_argument("--mode", type=str, required=True, choices=["profile", "time"], help="Profile or time.")
    parser.add_argument("--use_inductor", action="store_true", help="Should lower the function using inductor.")
    parser.add_argument("--dtype", type=str, required=True, choices=["32", "b16", "16"], help="Data type flag.")
    parser.add_argument("--n", type=int, required=True, help="See https://pytorch.org/docs/stable/generated/torch.mm.html.")
    parser.add_argument("--m", type=int, required=True, help="See https://pytorch.org/docs/stable/generated/torch.mm.html.")
    parser.add_argument("--p", type=int, required=True, help="See https://pytorch.org/docs/stable/generated/torch.mm.html.")
    parser.add_argument("--out_file", type=str, required=False, help="Path to the output CSV file.")
    # parser.add_argument("--bias", type=int, required=True, choices=[0, 1], help="Use bias (1) or not (0).")
    args = parser.parse_args()

    if args.mode == "time" and not args.out_file:
        raise ValueError("Time requires an outfile")

    return args

def main(args):
    dtype = get_dtype(args.dtype)
    kernel_params=f"{args.dtype}.{args.n}.{args.m}.{args.p}"

    A_size = torch.Size([args.n, args.m])
    B_size = torch.Size([args.m, args.p])

    if not check_size(dtype, A_size, B_size):
        # Flag as incomplete.
        save_row(kernel_params, np.nan, args.out_file)
        return

    A = torch.randn(A_size, dtype=dtype, device=device)
    B = torch.randn(B_size, dtype=dtype, device=device)

    if args.use_inductor:
        @torch.compile(backend="inductor")
        def mm(a, b):
            return torch.mm(a, b)
        fn = mm
    else:
        fn = torch.mm

    if args.mode == "time":
        time = time_fn(fn, A, B)
        save_row(kernel_params, time, args.out_file)
    else:
        profile_rep(fn, A, B)

if __name__ == "__main__":
    args = get_args()
    main(args)
