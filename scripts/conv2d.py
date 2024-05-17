import argparse
import numpy as np

import torch
from torch.nn.functional import conv2d, conv_transpose2d

from utils.profile_utils import get_dtype, time_fn, save_row, profile_rep, check_size

import warnings
warnings.filterwarnings("ignore")

# Assume this script runs on one device.
device = "cuda:0" if torch.cuda.is_available() else "cpu"
assert(device == "cuda:0")

def get_args():
    """
    TODO
    We assume that....
    kH == kW?
    stride: the same/int?

    padding: ignored.
    dilation: ??
    transposed: we don't care about?

    Arguments adapted from https://pytorch.org/docs/stable/generated/torch.nn.functional.conv2d.html
    """

    parser = argparse.ArgumentParser(description="2D convolution (conv2d).")
    parser.add_argument("--mode", type=str, required=True, choices=["profile", "time"], help="Profile or time.")
    parser.add_argument("--use_inductor", action="store_true", help="Should lower the function using inductor.")
    parser.add_argument("--dtype", type=str, required=True, choices=["32", "b16", "16"], help="Data type flag.")

    # input
    parser.add_argument("--b", type=int, required=True, help="Batch size.")
    parser.add_argument("--in_channels", type=int, required=True, help="Number of input channels.")
    parser.add_argument("--iH", type=int, required=True, help="Height of the input image.")
    parser.add_argument("--iW", type=int, required=True, help="Width of the input image.")

    # weight
    parser.add_argument("--out_channels", type=int, required=True, help="Number of output channels.")
    parser.add_argument("--groups", type=int, required=True, help="Number of groups for grouped convolution.")
    parser.add_argument("--kH", type=int, required=True, help="Height of the convolution kernel.")
    parser.add_argument("--kW", type=int, required=True, help="Width of the convolution kernel.")

    # other params
    # parser.add_argument("--bias", type=int, required=True, choices=[0, 1], help="Use bias (1) or not (0).")
    parser.add_argument("--stride", type=int, required=True, help="Stride of the convolution (only support number).")
    # parser.add_argument("--padding", type=int, required=True, help="We only support a number for now.")
    parser.add_argument("--dilation", type=int, required=True, help="Dilation of the convolution.")
    parser.add_argument("--transposed", type=int, required=True, choices=[0, 1], help="Use transposed convolution (1) or not (0).")

    parser.add_argument("--out_file", type=str, required=True, help="Path to the output CSV file.")
    args = parser.parse_args()

    if args.mode == "time" and not args.out_file:
        raise ValueError("Time requires an outfile")

    return args

def main(args):
    dtype = get_dtype(args.dtype)
    kernel_params = f"{args.dtype}.{args.b}.{args.in_channels}.{args.iH}.{args.iW}.{args.out_channels}.{args.groups}.{args.kH}.{args.kW}.{args.stride}.{args.dilation}.{args.transposed}"

    input_shape = torch.Size([args.b, args.in_channels, args.iH, args.iW])
    weight_shape = torch.Size([args.out_channels, args.in_channels // args.groups, args.kH, args.kW])

    if not check_size(dtype, input_shape, weight_shape):
        save_row(kernel_params, np.nan, args.out_file)
        return

    input = torch.randn(input_shape, dtype=dtype, device=device)
    weight = torch.randn(weight_shape, dtype=dtype, device=device)
    bias_tensor = None

    transposed = bool(args.transposed)

    if args.use_inductor:
        if transposed:
            @torch.compile(backend="inductor")
            def conv2d_t(x, w):
                return conv_transpose2d(x, w, bias=bias_tensor, stride=args.stride, dilation=args.dilation, groups=args.groups)
            fn = conv2d_t
        else:
            @torch.compile(backend="inductor")
            def _conv2d(x, w):
                return conv2d(x, w, bias=bias_tensor, stride=args.stride, dilation=args.dilation, groups=args.groups)
            fn = _conv2d
    else:
        if transposed:
            fn = lambda x, w: conv_transpose2d(x, w, bias=bias_tensor, stride=args.stride, dilation=args.dilation, groups=args.groups)
        else:
            fn = lambda x, w: conv2d(x, w, bias=bias_tensor, stride=args.stride, dilation=args.dilation, groups=args.groups)

    if args.mode == "time":
        time = time_fn(fn, input, weight)
        save_row(kernel_params, time, args.out_file)
    else:
        profile_rep(fn, input, weight)

if __name__ == "__main__":
    args = get_args()
    main(args)
