import os, csv
import argparse
import torch
from torch.nn.functional import conv2d, conv_transpose2d
from tqdm import tqdm
from itertools import product

from utils.profile_utils import ProfileBase

import warnings

warnings.filterwarnings("ignore")


def get_args_conv2d():
    """
    Arguments adapted from https://pytorch.org/docs/stable/generated/torch.nn.functional.conv2d.html
    """

    parser = argparse.ArgumentParser(description="2D convolution (conv2d).")
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["profile", "time"],
        help="Profile or time.",
    )
    parser.add_argument(
        "--use_inductor",
        action="store_true",
        help="Should lower the function using inductor.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        required=True,
        choices=["32", "b16", "16"],
        help="Data type flag.",
    )

    # input
    parser.add_argument("--b", type=int, required=True, help="Batch size.")
    parser.add_argument(
        "--in_channels", type=int, required=True, help="Number of input channels."
    )
    parser.add_argument(
        "--iH", type=int, required=True, help="Height of the input image."
    )
    parser.add_argument(
        "--iW", type=int, required=True, help="Width of the input image."
    )

    # weight
    parser.add_argument(
        "--out_channels", type=int, required=True, help="Number of output channels."
    )
    parser.add_argument(
        "--groups",
        type=int,
        required=True,
        help="Number of groups for grouped convolution.",
    )
    parser.add_argument(
        "--kH", type=int, required=True, help="Height of the convolution kernel."
    )
    parser.add_argument(
        "--kW", type=int, required=True, help="Width of the convolution kernel."
    )

    # other params
    # parser.add_argument("--bias", type=int, required=True, choices=[0, 1], help="Use bias (1) or not (0).")
    parser.add_argument(
        "--stride",
        type=int,
        required=True,
        help="Stride of the convolution (only support number).",
    )
    # parser.add_argument("--padding", type=int, required=True, help="We only support a number for now.")
    parser.add_argument(
        "--dilation", type=int, required=True, help="Dilation of the convolution."
    )
    parser.add_argument(
        "--transposed",
        type=int,
        required=True,
        choices=[0, 1],
        help="Use transposed convolution (1) or not (0).",
    )

    parser.add_argument(
        "--backward",
        type=int,
        required=True,
        choices=[0, 1],
        help="Profile the backward pass.",
    )
    parser.add_argument(
        "--out_file", type=str, required=True, help="Path to the output CSV file."
    )
    args = parser.parse_args()

    if args.mode == "time" and not args.out_file:
        raise ValueError("Time requires an outfile")

    return args


class ProfileConv2d(ProfileBase):
    def __init__(self, backward):
        self.backward = backward
        super().__init__()

    def get_input_sizes(self, args):
        input_shape = torch.Size([args.b, args.in_channels, args.iH, args.iW])
        weight_shape = torch.Size(
            [args.out_channels, args.in_channels // args.groups, args.kH, args.kW]
        )
        return [input_shape, weight_shape]

    def get_requires_grad(self) -> list[bool] | None:
        """Only want gradients for the kernel (`weight`)."""
        return [False, True]

    def get_output_size(self, args) -> torch.Size:
        # Assume output padding is zero if not specified
        padding = getattr(args, "padding", (0, 0))

        if bool(args.transposed):
            output_padding = getattr(args, "output_padding", (0, 0))
            out_height = (
                (args.iH - 1) * args.stride
                - 2 * padding[0]
                + args.kH
                + output_padding[0]
            )
            out_width = (
                (args.iW - 1) * args.stride
                - 2 * padding[1]
                + args.kW
                + output_padding[1]
            )
        else:
            out_height = (
                (args.iH + 2 * padding[0] - args.dilation * (args.kH - 1) - 1)
                // args.stride
            ) + 1
            out_width = (
                (args.iW + 2 * padding[1] - args.dilation * (args.kW - 1) - 1)
                // args.stride
            ) + 1

        return torch.Size([args.b, args.out_channels, out_height, out_width])

    def get_fn(self, args):
        transposed = bool(args.transposed)
        bias_tensor = None

        if args.use_inductor:
            raise ValueError("Not using Inductor for now.")
        else:
            if transposed:
                fn = lambda x, w: conv_transpose2d(
                    x,
                    w,
                    bias=bias_tensor,
                    stride=args.stride,
                    dilation=args.dilation,
                    groups=args.groups,
                )
            else:
                fn = lambda x, w: conv2d(
                    x,
                    w,
                    bias=bias_tensor,
                    stride=args.stride,
                    dilation=args.dilation,
                    groups=args.groups,
                )
        return fn

    def time(self, args):
        if (
            args.kH > args.iH
            or args.kW > args.iW
            or args.in_channels % args.groups
            or args.out_channels % args.groups
        ):
            # Do nothing.
            return

        with open(args.out_file, mode="a", newline="") as file:
            writer = csv.writer(file)
            if os.path.getsize(args.out_file) == 0:
                writer.writerow(self.time_header)

            kernel_params = f"{args.dtype}.{args.b}.{args.in_channels}.{args.iH}.{args.iW}.{args.out_channels}.{args.groups}.{args.kH}.{args.kW}.{args.stride}.{args.dilation}.{args.transposed}"
            writer.writerow([kernel_params, self.time_rep(args, self.backward)])


def main():
    args = get_args_conv2d()
    if args.mode == "time":
        ProfileConv2d(bool(args.backward)).time(args)
    else:
        ProfileConv2d(bool(args.backward)).profile(args)


if __name__ == "__main__":
    main()
