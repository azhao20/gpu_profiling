import os, sys, argparse
import numpy as np

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '..')))
from utils.profile_utils import get_dtype, time_model, save_row, profile_rep

import warnings
warnings.filterwarnings("ignore")

# Assume this script runs on one device.
device = "cuda:0" if torch.cuda.is_available() else "cpu"
assert(device == "cuda:0")

def main():
    parser = argparse.ArgumentParser(description="Time 2D convolution (conv2d).")
    parser.add_argument("--batch_size", type=int, required=True, help="Batch size of the input tensor.")
    parser.add_argument("--dtype_flag", type=int, required=True, choices=[32, 161, 162], help="Data type flag (32: float32, 161: bfloat16, 162: float16).")
    parser.add_argument("--in_channels", type=int, required=True, help="Number of input channels.")
    parser.add_argument("--iH", type=int, required=True, help="Height of the input image.")
    parser.add_argument("--iW", type=int, required=True, help="Width of the input image.")
    parser.add_argument("--out_channels", type=int, required=True, help="Number of output channels.")
    parser.add_argument("--groups", type=int, required=True, help="Number of groups for grouped convolution.")
    parser.add_argument("--kH", type=int, required=True, help="Height of the convolution kernel.")
    parser.add_argument("--kW", type=int, required=True, help="Width of the convolution kernel.")
    parser.add_argument("--stride", type=str, required=True, help="Stride of the convolution (number or tuple).")
    parser.add_argument("--padding", type=str, required=True, help="Padding applied to the input ('same', 'valid', or specific size).")
    parser.add_argument("--dilation", type=int, required=True, help="Dilation of the convolution.")
    parser.add_argument("--transposed", type=int, required=True, choices=[0, 1], help="Use transposed convolution (1) or not (0).")
    parser.add_argument("--bias", type=int, required=True, choices=[0, 1], help="Use bias (1) or not (0).")
    parser.add_argument("--out_file", type=str, required=True, help="Path to the output CSV file.")
    args = parser.parse_args()

    dtype = get_dtype(args.dtype_flag)
    input_tensor = torch.randn(args.batch_size, args.in_channels, args.iH, args.iW, dtype=dtype, device=device)
    weight = torch.randn(args.out_channels, args.in_channels // args.groups, args.kH, args.kW, dtype=dtype, device=device)
    bias_tensor = torch.randn(args.out_channels, dtype=dtype, device=device) if bool(args.bias) else None

    inputs = (input_tensor, weight)
    if args.transposed:
        @torch.compile(backend="inductor")
        def conv2d(x, w):
            return F.conv_transpose2d(x, w, bias=bias_tensor, stride=args.stride, padding=args.padding, dilation=args.dilation, groups=args.groups)
    else:
        @torch.compile(backend="inductor")
        def conv2d(x, w):
            return F.conv2d(x, w, bias=bias_tensor, stride=args.stride, padding=args.padding, dilation=args.dilation, groups=args.groups)

    if args.mode == "time":
        kernel_params = f"{args.batch_size}.{dtype}.{args.in_channels}.{args.iH}.{args.iW}.{args.out_channels}.{args.groups}.{args.kH}.{args.kW}.{args.stride}.{args.padding}.{args.dilation}.{args.transposed}.{args.bias}"
        time = time_model(conv2d, inputs)
        save_row(kernel_params, time, args.out_file)
    else:
        profile_rep(conv2d, inputs)

if __name__ == "__main__":
    main()
