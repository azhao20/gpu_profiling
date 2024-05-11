import argparse
import torch

from utils.profile_utils import get_dtype, _time_fn

import warnings
warnings.filterwarnings("ignore")

# Assume this script runs on one device.
device = "cuda:0" if torch.cuda.is_available() else "cpu"
assert(device == "cuda:0")

def main():
    parser = argparse.ArgumentParser(description="Time matrix multiplication (mm).")
    parser.add_argument("--dtype", type=str, required=True, choices=["32", "b16", "16"], help="Data type flag.")
    parser.add_argument("--b", type=int, required=True, help="See https://pytorch.org/docs/stable/generated/torch.mm.html.")
    parser.add_argument("--n", type=int, required=True, help="See https://pytorch.org/docs/stable/generated/torch.mm.html.")
    parser.add_argument("--m", type=int, required=True, help="See https://pytorch.org/docs/stable/generated/torch.mm.html.")
    parser.add_argument("--p", type=int, required=True, help="See https://pytorch.org/docs/stable/generated/torch.mm.html.")
    args = parser.parse_args()

    dtype = get_dtype(args.dtype)
    A = torch.randn(args.n, args.m, dtype=dtype, device=device)
    B = torch.randn(args.m, args.p, dtype=dtype, device=device)

    @torch.compile(backend="inductor")
    def mm(a, b):
        return torch.mm(a, b)

    print("Eager times:")
    print(_time_fn(torch.mm, A, B))

    print("Dynamo times")
    print(_time_fn(mm, A, B))

    A = torch.randn(args.b, args.n, args.m, dtype=dtype, device=device)
    B = torch.randn(args.b, args.m, args.p, dtype=dtype, device=device)

    @torch.compile(backend="inductor")
    def bmm(a, b):
        return torch.bmm(a, b)

    print("Eager times:")
    print(_time_fn(torch.bmm, A, B))

    print("Dynamo times")
    print(_time_fn(bmm, A, B))


if __name__ == "__main__":
    main()