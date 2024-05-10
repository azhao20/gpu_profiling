import argparse, os, sys

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
    parser = argparse.ArgumentParser(description="Time scaled dot-product attention (SDPA).")
    parser.add_argument("--mode", type=str, required=True, choices=["profile", "time"], help="Profile or time.")
    parser.add_argument("--batch_size", type=int, required=True, help="Batch size for the input tensor.")
    parser.add_argument("--dtype_flag", type=int, required=True, choices=[32, 161, 162], help="Data type flag (32: float32, 161: bfloat16, 162: float16).")
    parser.add_argument("--num_heads", type=int, required=True, help="Number of attention heads.")
    parser.add_argument("--seq_len", type=int, required=True, help="Sequence length (number of tokens).")
    parser.add_argument("--d_embed", type=int, required=True, help="Embedding dimension.")
    parser.add_argument("--out_file", type=str, required=True, help="Path to the output CSV file.")
    args = parser.parse_args()

    # fn
    mask = None
    @torch.compile(backend="inductor")
    def sdpa(q, k, v):
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

    # args
    dtype = get_dtype(args.dtype_flag)
    query = torch.randn(args.batch_size, args.num_heads, args.seq_len, args.d_embed, dtype=dtype, device=device)
    key = torch.randn(args.batch_size, args.num_heads, args.seq_len, args.d_embed, dtype=dtype, device=device)
    value = torch.randn(args.batch_size, args.num_heads, args.seq_len, args.d_embed, dtype=dtype, device=device)
    inputs = (query, key, value)

    if args.mode == "time":
        kernel_params = f"{args.batch_size}.{dtype}.{args.num_heads}.{args.seq_len}.{args.d_embed}"
        time = time_model(sdpa, inputs)
        save_row(kernel_params, time, args.out_file)
    else:
        profile_rep(sdpa, inputs)

if __name__ == "__main__":
    main()