import os, csv
import argparse
import torch
from torch.nn.functional import scaled_dot_product_attention
from torch.nn.attention import SDPBackend, sdpa_kernel
from tqdm import tqdm
from itertools import product

from utils.profile_utils import ProfileBase

import warnings

warnings.filterwarnings("ignore")


def get_args_sdpa():
    """
    Parameter shapes:
    https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html

    Naming convention adapted from torch.utils.flop_counter:
    b, h, s_q, d_q = query_shape
    _b2, _h2, s_k, _d2 = key_shape
    _b3, _h3, _s3, d_v = value_shape

    More on backends:
    Math backend is C++ implementation, but still runs on GPU. We aren't using this for now.
    `flash` only works with float16 and bfloat16 dtypes.

    Relevant links:
    https://pytorch.org/docs/stable/generated/torch.nn.attention.sdpa_kernel.html#torch.nn.attention.sdpa_kernel
    https://github.com/huggingface/transformers/issues/26557
    """

    parser = argparse.ArgumentParser(description="Scaled dot-product attention (SDPA).")
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
    parser.add_argument(
        "--backend",
        type=str,
        required=True,
        choices=["flash", "efficient"],
        help="See https://pytorch.org/docs/stable/generated/torch.nn.attention.sdpa_kernel.html#torch.nn.attention.sdpa_kernel.",
    )
    parser.add_argument("--b", type=int, required=False, help="Batch size.")
    parser.add_argument(
        "--h", type=int, required=True, help="Number of attention heads."
    )
    parser.add_argument(
        "--s_q", type=int, required=False, help="Target sequence length."
    )
    parser.add_argument(
        "--s_kv",
        type=int,
        required=False,
        help="Source (key and value) sequence length.",
    )
    parser.add_argument(
        "--d_qk", type=int, required=False, help="Query and key embedding dimension."
    )
    parser.add_argument(
        "--d_v", type=int, required=False, help="Value embedding dimension."
    )
    parser.add_argument(
        "--is_causal",
        type=int,
        required=False,
        choices=[0, 1],
        help="Use causal attention.",
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


class ProfileSDPA(ProfileBase):
    def __init__(self, backward: bool):
        super().__init__()
        self.backward = backward
        self.contexts = {
            "efficient": SDPBackend.EFFICIENT_ATTENTION,
            "flash": SDPBackend.FLASH_ATTENTION,
            # "math": SDPBackend.MATH # We don't support for now.
        }

    def get_input_sizes(self, args):
        q_shape = torch.Size([args.b, args.h, args.s_q, args.d_qk])
        k_shape = torch.Size([args.b, args.h, args.s_kv, args.d_qk])
        v_shape = torch.Size([args.b, args.h, args.s_kv, args.d_v])
        return [q_shape, k_shape, v_shape]

    def get_requires_grad(self) -> list[bool] | None:
        """All matrices require gradients."""
        return [True] * 3

    def get_output_size(self, args):
        return torch.Size([args.b, args.h, args.s_q, args.d_v])

    def get_fn(self, args):
        is_causal = bool(args.is_causal)
        if args.use_inductor:
            raise ValueError("Not using Inductor for now.")

            @torch.compile(backend="inductor")
            def sdpa(q, k, v):
                return scaled_dot_product_attention(q, k, v, is_causal=is_causal)

            fn = sdpa
        else:
            fn = lambda q, k, v: scaled_dot_product_attention(
                q, k, v, is_causal=is_causal
            )
        return fn

    def time_efficient(self, args) -> None:
        batch_sizes = [2, 4, 8, 16, 32, 64, 128]
        sq_lengths = [32, 64, 128, 256, 512]
        skv_lengths = [32, 64, 128, 256, 512]
        dqk_sizes = [32, 64, 128, 256]
        dv_sizes = [32, 64, 128, 256]

        # Uncomment for testing
        # batch_sizes=[32]
        # sq_lengths=[32]
        # skv_lengths=[32]
        # dqk_sizes=[32]
        # dv_sizes=[32]

        with open(args.out_file, mode="a", newline="") as file:
            writer = csv.writer(file)
            if os.path.getsize(args.out_file) == 0:
                writer.writerow(self.time_header)

            with sdpa_kernel(self.contexts[args.backend]):
                for b, s_q, s_kv, d_qk, d_v, is_causal in tqdm(
                    product(
                        batch_sizes,
                        sq_lengths,
                        skv_lengths,
                        dqk_sizes,
                        dv_sizes,
                        [0, 1],
                    )
                ):
                    args.b = b
                    args.s_q = s_q
                    args.s_kv = s_kv
                    args.d_qk = d_qk
                    args.d_v = d_v
                    args.is_causal = is_causal

                    kernel_params = (
                        f"{args.dtype}.{args.backend}.{args.b}.{args.h}."
                        f"{args.s_q}.{args.s_kv}.{args.d_qk}.{args.d_v}.{args.is_causal}"
                    )

                    writer.writerow(
                        [
                            kernel_params,
                            self.time_rep(args, self.backward),
                        ]
                    )
                file.flush()

    def time_flash(self, args):
        """
        For the flash backend, there are constraints on the parameters.

        d_qk == d_v <= 256.
        For is_causal, s_q == s_kv
        """
        batch_sizes = [2, 4, 8, 16, 32, 64]
        dqkv_sizes = [i for i in range(16, 256 + 1, 16)]

        # For non-causal
        sq_lengths = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
        skv_lengths = [32, 64, 128, 256, 512, 1024, 2048]
        # For causal, these lengths are the same.
        sqkv_lengths = [
            2,
            4,
            8,
            16,
            32,
            64,
            128,
            256,
            512,
            1024,
            1536,
            2048,
            3072,
            4096,
        ]

        # Uncomment for testing
        # batch_sizes=[32]
        # dqkv_sizes=[32]
        # sq_lengths=[32]
        # skv_lengths=[32]
        # sqkv_lengths=[32]

        with open(args.out_file, mode="a", newline="") as file:
            writer = csv.writer(file)
            if os.path.getsize(args.out_file) == 0:
                writer.writerow(self.time_header)

            args.is_causal = 0
            with sdpa_kernel(self.contexts[args.backend]):
                for b, s_q, s_kv, d_qkv in tqdm(
                    product(batch_sizes, sq_lengths, skv_lengths, dqkv_sizes)
                ):
                    args.b = b
                    args.s_q = s_q
                    args.s_kv = s_kv
                    args.d_qk = d_qkv
                    args.d_v = d_qkv
                    kernel_params = f"{args.dtype}.{args.backend}.{args.b}.{args.h}.{args.s_q}.{args.s_kv}.{args.d_qk}.{args.d_v}.{args.is_causal}"
                    writer.writerow([kernel_params, self.time_rep(args, self.backward)])
                file.flush()

            args.is_causal = 1
            with sdpa_kernel(self.contexts[args.backend]):
                for b, s_qkv, d_qkv in tqdm(
                    product(batch_sizes, sqkv_lengths, dqkv_sizes)
                ):
                    args.b = b
                    args.s_q = s_qkv
                    args.s_kv = s_qkv
                    args.d_qk = d_qkv
                    args.d_v = d_qkv
                    kernel_params = f"{args.dtype}.{args.backend}.{args.b}.{args.h}.{args.s_q}.{args.s_kv}.{args.d_qk}.{args.d_v}.{args.is_causal}"
                    writer.writerow([kernel_params, self.time_rep(args, self.backward)])
                file.flush()

    def time(self, args):
        if args.backend == "efficient":
            self.time_efficient(args)
        elif args.backend == "flash":
            self.time_flash(args)
        else:
            raise ValueError(f"Backend {args.backend} not supported")


def main():
    args = get_args_sdpa()
    if args.mode == "time":
        ProfileSDPA(bool(args.backward)).time(args)
    else:
        ProfileSDPA(bool(args.backward)).profile(args)


if __name__ == "__main__":
    main()
