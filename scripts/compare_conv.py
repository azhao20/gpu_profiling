import argparse

import torch
from torch.nn.functional import conv2d, conv_transpose2d

from utils.profile_utils import get_dtype, _time_fn

import warnings
warnings.filterwarnings("ignore")

# Assume this script runs on one device.
device = "cuda:0" if torch.cuda.is_available() else "cpu"
assert(device == "cuda:0")

def main():
    """
    TODO
    We assume that....
    kH == kW?
    transposed: we don't care about?

    Arguments adapted from https://pytorch.org/docs/stable/generated/torch.nn.functional.conv2d.html
    """

    parser = argparse.ArgumentParser(description="Time 2D convolution (conv2d).")
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
    parser.add_argument("--padding", type=int, required=True, help="We only support a number for now.")
    parser.add_argument("--dilation", type=int, required=True, help="Dilation of the convolution.")

    args = parser.parse_args()

    dtype = get_dtype(args.dtype)
    input = torch.randn(args.b, args.in_channels, args.iH, args.iW, dtype=dtype, device=device)
    weight = torch.randn(args.out_channels, args.in_channels // args.groups, args.kH, args.kW, dtype=dtype, device=device)
    bias_tensor = None

    def fn_t(x, w):
        return conv_transpose2d(x, w, bias=bias_tensor, stride=args.stride, padding=args.padding, dilation=args.dilation, groups=args.groups)

    def fn(x, w):
        return conv2d(x, w, bias=bias_tensor, stride=args.stride, padding=args.padding, dilation=args.dilation, groups=args.groups)

    print("Eager times:")
    print(_time_fn(fn, input, weight))
    print(_time_fn(fn_t, input, weight))

    @torch.compile(backend="inductor")
    def fn_t(x, w):
        return conv_transpose2d(x, w, bias=bias_tensor, stride=args.stride, padding=args.padding, dilation=args.dilation, groups=args.groups)

    @torch.compile(backend="inductor")
    def fn(x, w):
        return conv2d(x, w, bias=bias_tensor, stride=args.stride, padding=args.padding, dilation=args.dilation, groups=args.groups)

    print("Dynamo times")
    print(_time_fn(fn, input, weight))
    print(_time_fn(fn_t, input, weight))

if __name__ == "__main__":
    main()
