import os, csv
import argparse
import torch
from torch.nn.functional import conv2d, conv_transpose2d

from utils.profile_utils import ProfileBase

import warnings
warnings.filterwarnings("ignore")

def get_args_conv2d():
    """
    Arguments adapted from https://pytorch.org/docs/stable/generated/torch.nn.functional.conv2d.html
    """

    parser = argparse.ArgumentParser(description="2D convolution (conv2d).")
    parser.add_argument("--mode", type=str, required=True, choices=["profile", "time"], help="Profile or time.")
    parser.add_argument("--use_inductor", action="store_true", help="Should lower the function using inductor.")
    parser.add_argument("--dtype", type=str, required=False, choices=["32", "b16", "16"], help="Data type flag.")

    # input
    parser.add_argument("--b", type=int, required=False, help="Batch size.")
    parser.add_argument("--in_channels", type=int, required=False, help="Number of input channels.")
    parser.add_argument("--iH", type=int, required=True, help="Height of the input image.")
    parser.add_argument("--iW", type=int, required=True, help="Width of the input image.")

    # weight
    parser.add_argument("--out_channels", type=int, required=False, help="Number of output channels.")
    parser.add_argument("--groups", type=int, required=False, help="Number of groups for grouped convolution.")
    parser.add_argument("--kH", type=int, required=False, help="Height of the convolution kernel.")
    parser.add_argument("--kW", type=int, required=False, help="Width of the convolution kernel.")

    # other params
    # parser.add_argument("--bias", type=int, required=True, choices=[0, 1], help="Use bias (1) or not (0).")
    parser.add_argument("--stride", type=int, required=False, help="Stride of the convolution (only support number).")
    # parser.add_argument("--padding", type=int, required=True, help="We only support a number for now.")
    parser.add_argument("--dilation", type=int, required=False, help="Dilation of the convolution.")
    parser.add_argument("--transposed", type=int, required=True, choices=[0, 1], help="Use transposed convolution (1) or not (0).")

    parser.add_argument("--backward", type=int, required=True, choices=[0, 1], help="Profile the backward pass.")
    parser.add_argument("--out_file", type=str, required=True, help="Path to the output CSV file.")
    args = parser.parse_args()

    if args.mode == "time" and not args.out_file:
        raise ValueError("Time requires an outfile")

    return args


class ProfileConv2d(ProfileBase):
    def __init__(self, backward):
        self.backward = backward
        super().__init__()

    def get_input_sizes(self, args) -> list:
        input_shape = torch.Size([args.b, args.in_channels, args.iH, args.iW])
        weight_shape = torch.Size([args.out_channels, args.in_channels // args.groups, args.kH, args.kW])
        return [input_shape, weight_shape]

    def get_requires_grad(self) -> list[bool] | None:
        """Only want gradients for the kernel (`weight`)."""
        return [False, True]

    def get_output_size(self, args):
        # Assume output padding is zero if not specified
        padding = getattr(args, 'padding', (0, 0))

        if bool(args.transposed):
            output_padding = getattr(args, 'output_padding', (0, 0))
            out_height = (args.iH - 1) * args.stride - 2 * padding[0] + args.kH + output_padding[0]
            out_width = (args.iW - 1) * args.stride - 2 * padding[1] + args.kW + output_padding[1]
        else:
            out_height = ((args.iH + 2 * padding[0] - args.dilation * (args.kH - 1) - 1) // args.stride) + 1
            out_width = ((args.iW + 2 * padding[1] - args.dilation * (args.kW - 1) - 1) // args.stride) + 1

        return torch.Size([args.b, args.out_channels, out_height, out_width])

    def get_fn(self, args):
        transposed = bool(args.transposed)
        bias_tensor = None

        if args.use_inductor:
            raise ValueError("Not using Inductor for now.")
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
        return fn

    def time_conv2d_channels(self, args, writer):
        """
        Assumes that all other parameters have already been set.
        """
        channel_sizes = [2, 8, 32, 128, 512, 1024, 2048, 4096]

        # channel_sizes = [32]

        for in_channels in channel_sizes:
            for out_channels in channel_sizes:
                if in_channels % args.groups or out_channels % args.groups:
                    continue
                args.in_channels = in_channels
                args.out_channels = out_channels
                kernel_params = f"{args.dtype}.{args.b}.{args.in_channels}.{args.iH}.{args.iW}.{args.out_channels}.{args.groups}.{args.kH}.{args.kW}.{args.stride}.{args.dilation}.{args.transposed}"
                writer.writerow([kernel_params, self.time_rep(args, self.backward)])

    def time_conv2d_transposed_channels(self, args, writer):
        """Transposed Conv2d sets in_channels == out_channels."""
        channel_sizes = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1536, 2048, 3072, 4096]

        # channel_sizes = [32]

        for channels in channel_sizes:
            if channels % args.groups:
                continue
            args.in_channels = channels
            args.out_channels = channels
            kernel_params = f"{args.dtype}.{args.b}.{args.in_channels}.{args.iH}.{args.iW}.{args.out_channels}.{args.groups}.{args.kH}.{args.kW}.{args.stride}.{args.dilation}.{args.transposed}"
            writer.writerow([kernel_params, self.time_rep(args, self.backward)])

    def time(self, args):
        """
        Could consider a param generator.
        """
        group_sizes = [1, 64, 128, 256, 512, 1024]
        batch_sizes = [2, 4, 8, 16, 32]
        kernel_sizes = [3, 5, 7]
        strides = [1]
        dilations = [1]

        # Uncomment for testing
        # group_sizes = [1]
        # batch_sizes = [4]

        with open(args.out_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            if os.path.getsize(args.out_file) == 0:
                writer.writerow(self.time_header)
            for dname in self.dtype_map:
                for b in batch_sizes:
                    for kH in kernel_sizes:
                        for kW in kernel_sizes:
                            if kH > args.iH or kW > args.iW:
                                continue
                            for stride in strides:
                                for dilation in dilations:
                                    for groups in group_sizes:
                                        args.dtype = dname
                                        args.b = b
                                        args.groups = groups
                                        args.kH = kH
                                        args.kW = kW
                                        args.stride = stride
                                        args.dilation = dilation

                                        if args.transposed:
                                            self.time_conv2d_transposed_channels(args, writer)
                                        else:
                                            self.time_conv2d_channels(args, writer)
                                # Flush intermittently in case something crashes
                                file.flush()

def main():
    args = get_args_conv2d()
    if args.mode == "time":
        ProfileConv2d(bool(args.backward)).time(args)
    else:
        ProfileConv2d(bool(args.backward)).profile(args)

if __name__ == "__main__":
    main()
