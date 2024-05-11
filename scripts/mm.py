import argparse
import torch

from utils.profile_utils import get_dtype, time_fn, save_row, profile_rep

import warnings
warnings.filterwarnings("ignore")

# Assume this script runs on one device.
device = "cuda:0" if torch.cuda.is_available() else "cpu"
assert(device == "cuda:0")

def main():
    """
    Note: we don't profile with bias for now.
    """

    parser = argparse.ArgumentParser(description="Time matrix multiplication (mm).")
    parser.add_argument("--mode", type=str, required=True, choices=["profile", "time"], help="Profile or time.")
    parser.add_argument("--dtype", type=str, required=True, choices=["32", "b16", "16"], help="Data type flag.")
    parser.add_argument("--n", type=int, required=True, help="See https://pytorch.org/docs/stable/generated/torch.mm.html.")
    parser.add_argument("--m", type=int, required=True, help="See https://pytorch.org/docs/stable/generated/torch.mm.html.")
    parser.add_argument("--p", type=int, required=True, help="See https://pytorch.org/docs/stable/generated/torch.mm.html.")
    parser.add_argument("--out_file", type=str, required=False, help="Path to the output CSV file.")
    # parser.add_argument("--bias", type=int, required=True, choices=[0, 1], help="Use bias (1) or not (0).")
    args = parser.parse_args()

    if args.mode == "time" and not args.out_file:
        raise ValueError("Time requires an outfile")

    dtype = get_dtype(args.dtype)
    A = torch.randn(args.n, args.m, dtype=dtype, device=device)
    B = torch.randn(args.m, args.p, dtype=dtype, device=device)

    @torch.compile(backend="inductor")
    def mm(a, b):
        return torch.mm(a, b)

    if args.mode == "time":
        kernel_params=f"{args.dtype}.{args.n}.{args.m}.{args.p}"
        time = time_fn(mm, A, B)
        save_row(kernel_params, time, args.out_file)
    else:
        profile_rep(mm, A, B)

if __name__ == "__main__":
    main()