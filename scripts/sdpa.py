import os, csv
import argparse
import torch
from torch.nn.functional import scaled_dot_product_attention
from torch.nn.attention import SDPBackend, sdpa_kernel

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
    flash only works with float16 and bfloat16 dtypes.
    https://pytorch.org/docs/stable/generated/torch.nn.attention.sdpa_kernel.html#torch.nn.attention.sdpa_kernel
    https://github.com/huggingface/transformers/issues/26557
    """

    parser = argparse.ArgumentParser(description="Scaled dot-product attention (SDPA).")
    parser.add_argument("--mode", type=str, required=True, choices=["profile", "time"], help="Profile or time.")
    parser.add_argument("--use_inductor", action="store_true", help="Should lower the function using inductor.")
    parser.add_argument("--dtype", type=str, required=True, choices=["32", "b16", "16"], help="Data type flag.")
    parser.add_argument("--backend", type=str, required=True, choices=["flash", "efficient"], \
                        help="See https://pytorch.org/docs/stable/generated/torch.nn.attention.sdpa_kernel.html#torch.nn.attention.sdpa_kernel.")
    parser.add_argument("--b", type=int, required=False, help="Batch size.")
    parser.add_argument("--h", type=int, required=True, help="Number of attention heads.")
    parser.add_argument("--s_q", type=int, required=False, help="Target sequence length.")
    parser.add_argument("--s_kv", type=int, required=False, help="Source (key and value) sequence length.")
    parser.add_argument("--d_qk", type=int, required=False, help="Query and key embedding dimension.")
    parser.add_argument("--d_v", type=int, required=False, help="Value embedding dimension.")
    parser.add_argument("--is_causal", type=int, required=False, choices=[0, 1], help="Use causal attention.")

    parser.add_argument("--backward", type=int, required=True, choices=[0, 1], help="Profile the backward pass.")
    parser.add_argument("--out_file", type=str, required=True, help="Path to the output CSV file.")
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

    def get_input_sizes(self, args) -> list:
        q_shape = torch.Size([args.b, args.h, args.s_q, args.d_qk])
        k_shape = torch.Size([args.b, args.h, args.s_kv, args.d_qk])
        v_shape = torch.Size([args.b, args.h, args.s_kv, args.d_v])
        return [q_shape, k_shape, v_shape]

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
            fn = lambda q, k, v: scaled_dot_product_attention(q, k, v, is_causal=is_causal)
        return fn
    
    def time(self, args):
        """
        Could consider a param generator.
        """
        if args.backend == "efficient":
            batch_sizes = [2, 4, 8, 16, 32, 64, 128]
            sq_lengths = [32, 64, 128, 256]
            skv_lengths = [32, 64, 128, 256]
            dqk_sizes = [32, 64, 128]
            dv_sizes = [32, 64, 128, 256]
        elif args.backend == "flash":
            batch_sizes = [2, 4, 8, 16, 32, 64]
            sq_lengths = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
            skv_lengths = [32, 64, 128, 256, 512, 1024, 2048]
            dqk_sizes = [32, 64, 128]
            dv_sizes = [32, 64, 128, 256, 512, 1024, 2048]

        # Uncomment for testing
        # batch_sizes=[32]
        # sq_lengths=[32]
        # skv_lengths=[32]
        # dqk_sizes=[32]
        # dv_sizes=[32]

        with open(args.out_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            if os.path.getsize(args.out_file) == 0:
                writer.writerow(self.time_header)

            with sdpa_kernel(self.contexts[args.backend]):
                for b in batch_sizes:
                    for s_q in sq_lengths:
                        for s_kv in skv_lengths:
                            for d_qk in dqk_sizes:
                                for d_v in dv_sizes:
                                    for is_causal in [0, 1]:
                                        args.b = b
                                        args.s_q = s_q
                                        args.s_kv = s_kv
                                        args.d_qk = d_qk
                                        args.d_v = d_v
                                        args.is_causal = is_causal
                                        kernel_params = f"{args.dtype}.{args.backend}.{args.b}.{args.h}.{args.s_q}.{args.s_kv}.{args.d_qk}.{args.d_v}.{args.is_causal}"
                                        # Decide to profile backward or not.
                                        writer.writerow([kernel_params, self.time_rep(args, self.backward)])
                            # Flush intermittently in case something crashes
                            file.flush()


def main():
    args = get_args_sdpa()
    if args.mode == "time":
        ProfileSDPA(bool(args.backward)).time(args)
    else:
        ProfileSDPA(bool(args.backward)).profile(args)

if __name__ == "__main__":
    main()
