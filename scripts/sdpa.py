import argparse
import numpy as np

import torch
from torch.nn.functional import scaled_dot_product_attention
from torch.nn.attention import SDPBackend, sdpa_kernel

from utils.profile_utils import get_dtype, time_fn, save_row, profile_rep, check_size

import warnings
warnings.filterwarnings("ignore")

# Assume this script runs on one device.
device = "cuda:0" if torch.cuda.is_available() else "cpu"
assert(device == "cuda:0")

contexts = {
    "math": SDPBackend.MATH,
    "efficient": SDPBackend.EFFICIENT_ATTENTION,
    "flash": SDPBackend.FLASH_ATTENTION
}

def get_args():
    """
    Parameter shapes:
    https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html

    Naming convention adapted from torch.utils.flop_counter:
    b, h, s_q, d_q = query_shape
    _b2, _h2, s_k, _d2 = key_shape
    _b3, _h3, _s3, d_v = value_shape

    More on backends:
    MATH backend is C++ implementation, but still runs on GPU.
    "flash" only works with float16 and bfloat16 dtypes.
    https://pytorch.org/docs/stable/generated/torch.nn.attention.sdpa_kernel.html#torch.nn.attention.sdpa_kernel
    https://github.com/huggingface/transformers/issues/26557

    We aren't using is_causal for now, since there are too many params.
    """

    parser = argparse.ArgumentParser(description="Scaled dot-product attention (SDPA).")
    parser.add_argument("--mode", type=str, required=True, choices=["profile", "time"], help="Profile or time.")
    parser.add_argument("--use_inductor", action="store_true", help="Should lower the function using inductor.")
    parser.add_argument("--dtype", type=str, required=True, choices=["32", "b16", "16"], help="Data type flag.")
    parser.add_argument("--backend", type=str, required=True, choices=["math", "flash", "efficient"], \
                        help="See https://pytorch.org/docs/stable/generated/torch.nn.attention.sdpa_kernel.html#torch.nn.attention.sdpa_kernel.")
    parser.add_argument("--b", type=int, required=True, help="Batch size.")
    parser.add_argument("--h", type=int, required=True, help="Number of attention heads.")
    parser.add_argument("--s_q", type=int, required=True, help="Target sequence length.")
    parser.add_argument("--s_kv", type=int, required=True, help="Source (key and value) sequence length.")
    parser.add_argument("--d_qk", type=int, required=True, help="Query and key embedding dimension.")
    parser.add_argument("--d_v", type=int, required=True, help="Value embedding dimension.")
    parser.add_argument("--is_causal", type=int, required=True, choices=[0, 1], help="Use causal attention.")

    parser.add_argument("--out_file", type=str, required=True, help="Path to the output CSV file.")
    args = parser.parse_args()

    if args.mode == "time" and not args.out_file:
        raise ValueError("Time requires an outfile")

    return args

def main(args):
    dtype = get_dtype(args.dtype)
    kernel_params = f"{args.dtype}.{args.backend}.{args.b}.{args.h}.{args.s_q}.{args.s_kv}.{args.d_qk}.{args.d_v}.{args.is_causal}"

    q_shape = torch.Size([args.b, args.h, args.s_q, args.d_qk])
    k_shape = torch.Size([args.b, args.h, args.s_kv, args.d_qk])
    v_shape = torch.Size([args.b, args.h, args.s_kv, args.d_v])

    if not check_size(dtype, q_shape, k_shape, v_shape):
        save_row(kernel_params, np.nan, args.out_file)
        return

    query = torch.randn(q_shape, dtype=dtype, device=device)
    key = torch.randn(k_shape, dtype=dtype, device=device)
    value = torch.randn(v_shape, dtype=dtype, device=device)

    # Note: we don't use causal for now.
    is_causal = bool(args.is_causal)

    if args.use_inductor:
        @torch.compile(backend="inductor")
        def sdpa(q, k, v):
            return scaled_dot_product_attention(q, k, v, is_causal=is_causal)
        fn = sdpa
    else:
        fn = lambda q, k, v: scaled_dot_product_attention(q, k, v, is_causal=is_causal)

    context = contexts[args.backend]
    with sdpa_kernel(context):
        if args.mode == "time":
            time = time_fn(fn, query, key, value)
            save_row(kernel_params, time, args.out_file)
        else:
            profile_rep(fn, query, key, value)

if __name__ == "__main__":
    args = get_args()
    main(args)
