import os, sys, csv
import numpy as np

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '..')))

from utils.profile_utils import get_dtype, time_model

import warnings
warnings.filterwarnings("ignore")

# Assume this script runs on one device.
device = "cuda:0" if torch.cuda.is_available() else "cpu"
assert(device == "cuda:0")

def time_sdpa_rep(batch_size, dtype, seq_len, d_embed, num_heads, out_file):
    query = torch.randn(batch_size, num_heads, seq_len, d_embed, dtype=dtype, device=device)
    key = torch.randn(batch_size, num_heads, seq_len, d_embed, dtype=dtype, device=device)
    value = torch.randn(batch_size, num_heads, seq_len, d_embed, dtype=dtype, device=device)
    mask = None

    @torch.compile(backend="inductor")
    def sdpa(q, k, v):
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

    time = time_model(sdpa, (query, key, value))

    kernel_params = f"{batch_size}.{dtype}.{seq_len}.{d_embed}.{num_heads}"

    with open(out_file, mode='a', newline='') as file:
        writer = csv.writer(file)

        if os.path.getsize(out_file) == 0:
            writer.writerow(['Kernel Name', 'Latency (ms)'])
        writer.writerow([kernel_params, time])

if __name__ == "__main__":
    assert len(sys.argv) == 7, "Usage: python script.py batch_size dtype_flag seq_len d_embed num_heads out_file"
    batch_size, dtype_flag, seq_len, d_embed, num_heads = \
        int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
    dtype = get_dtype(dtype_flag)

    out_file = str(sys.argv[6])

    time_sdpa_rep(batch_size, dtype, seq_len, d_embed, num_heads, out_file)