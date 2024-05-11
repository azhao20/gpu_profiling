import argparse

import torch
from torch.nn.functional import scaled_dot_product_attention
from torch.nn.attention import SDPBackend, sdpa_kernel

from utils.profile_utils import get_dtype, _time_fn

import warnings
warnings.filterwarnings("ignore")

# Assume this script runs on one device.
device = "cuda:0" if torch.cuda.is_available() else "cpu"
assert(device == "cuda:0")

def main():
    """
    Parameter shapes:
    https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html

    Naming convention adapted from torch.utils.flop_counter:
    b, h, s_q, d_q = query_shape
    _b2, _h2, s_k, _d2 = key_shape
    _b3, _h3, _s3, d_v = value_shape

    More on backends:
    https://pytorch.org/docs/stable/generated/torch.nn.attention.sdpa_kernel.html#torch.nn.attention.sdpa_kernel

    We aren't using is_causal for now, since there are too many params.

    No available kernel for --backend "flash".
    """

    parser = argparse.ArgumentParser(description="Time scaled dot-product attention (SDPA).")
    parser.add_argument("--dtype", type=str, required=True, choices=["32", "b16", "16"], help="Data type flag.")
    # parser.add_argument("--backend", type=str, required=True, choices=["flash", "efficient"], \
    #                     help="See https://pytorch.org/docs/stable/generated/torch.nn.attention.sdpa_kernel.html#torch.nn.attention.sdpa_kernel.")

    parser.add_argument("--b", type=int, required=True, help="Batch size.")
    parser.add_argument("--h", type=int, required=True, help="Number of attention heads.")
    parser.add_argument("--s_q", type=int, required=True, help="Target sequence length.")
    parser.add_argument("--s_kv", type=int, required=True, help="Source (key and value) sequence length.")
    parser.add_argument("--d_qk", type=int, required=True, help="Query and key embedding dimension.")
    parser.add_argument("--d_v", type=int, required=True, help="Value embedding dimension.")

    # parser.add_argument("--is_causal", action='store_true', required=True, help="Use causal attention.")

    args = parser.parse_args()

    dtype = get_dtype(args.dtype)
    query = torch.randn(args.b, args.h, args.s_q, args.d_qk, dtype=dtype, device=device)
    key = torch.randn(args.b, args.h, args.s_kv, args.d_qk, dtype=dtype, device=device)
    value = torch.randn(args.b, args.h, args.s_kv, args.d_v, dtype=dtype, device=device)

    # is_causal = bool(args.is_causal)
    is_causal = False

    @torch.compile(backend="inductor")
    def sdpa(q, k, v):
        return scaled_dot_product_attention(q, k, v, is_causal=is_causal)

    print("Eager times:")
    print(_time_fn(scaled_dot_product_attention, query, key, value))

    # with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    #     print(_time_fn(scaled_dot_product_attention, query, key, value))

    with sdpa_kernel([SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]):
        print(_time_fn(scaled_dot_product_attention, query, key, value))

    print("Dynamo times")
    print(_time_fn(sdpa, query, key, value))

    with sdpa_kernel([SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]):
        print(_time_fn(sdpa, query, key, value))

if __name__ == "__main__":
    main()