import argparse, os, sys
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '..')))
from utils.profile_utils import get_dtype, time_model, save_row, profile_rep

import warnings
warnings.filterwarnings("ignore")

# Assume this script runs on one device.
device = "cuda:0" if torch.cuda.is_available() else "cpu"
assert(device == "cuda:0")

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Time matrix multiplication (addmm).")
    parser.add_argument("--mode", type=str, required=True, choices=["profile", "time"], help="Profile or time.")
    parser.add_argument("--batch_size", type=int, required=True, help="Batch size for the input tensor.")
    parser.add_argument("--dtype_flag", type=int, required=True, choices=[32, 161, 162], help="Data type flag (32: float32, 161: bfloat16, 162: float16).")
    parser.add_argument("--bias", type=int, required=True, choices=[0, 1], help="Use bias (1) or not (0).")
    parser.add_argument("--in_size", type=int, required=True, help="Input size (number of columns in the input matrix).")
    parser.add_argument("--out_size", type=int, required=True, help="Output size (number of columns in the output matrix).")
    parser.add_argument("--out_file", type=str, required=True, help="Path to the output CSV file.")
    args = parser.parse_args()
    
    dtype = get_dtype(args.dtype_flag)
    A = torch.randn(args.batch_size, args.in_size, dtype=dtype, device=device)
    B = torch.randn(args.in_size, args.out_size, dtype=dtype, device=device)

    use_bias = bool(args.bias)
    if use_bias:
        C = torch.randn(args.out_size, dtype=dtype, device=device)

        @torch.compile(backend="inductor")
        def addmm(a, b, bias):
            return torch.addmm(bias, a, b)
        fn = addmm
        inputs = (A, B, C)
    else:
        @torch.compile(backend="inductor")
        def mm(a, b):
            return torch.mm(a, b)
        fn = mm
        inputs = (A, B)

    if args.mode == "time":
        kernel_params=f"{args.batch_size}.{dtype}.{args.bias}.{args.in_size}.{args.out_size}"
        time = time_model(fn, inputs)
        save_row(kernel_params, time, args.out_file)
    else:
        profile_rep(fn, inputs)

if __name__ == "__main__":
    main()