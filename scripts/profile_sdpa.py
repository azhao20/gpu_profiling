import os, sys
import traceback as tb

import torch
import torch.nn.functional as F
from torch.cuda import nvtx

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '..')))
from utils.profile_utils import get_dtype, WARMUP_REPS

import warnings
warnings.filterwarnings("ignore")

device = "cuda:0" if torch.cuda.is_available() else "cpu"
assert(device == "cuda:0")

def profile_sdpa_rep(batch_size, dtype, seq_len, d_embed, num_heads):
    """
    Profile SDPA using d_embed directly.
    """
    query = torch.randn(batch_size, num_heads, seq_len, d_embed, dtype=dtype, device=device)
    key = torch.randn(batch_size, num_heads, seq_len, d_embed, dtype=dtype, device=device)
    value = torch.randn(batch_size, num_heads, seq_len, d_embed, dtype=dtype, device=device)
    mask = None

    @torch.compile(backend="inductor")
    def sdpa(q, k, v):
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

    res = [0] * (WARMUP_REPS + 1)
    torch.cuda.empty_cache()
    try:
        with torch.no_grad():
            for i in range(WARMUP_REPS):
                res[i] = sdpa(query, key, value)
            
            torch.cuda.cudart().cudaProfilerStart()
            nvtx.range_push("profile_range")
            res[-1] = sdpa(query, key, value)
            nvtx.range_pop()
            torch.cuda.cudart().cudaProfilerStop()

    except Exception as e:
        print("Failed during the attention operation!")
        tb.print_exc()

if __name__ == "__main__":
    assert len(sys.argv) == 6, "Usage: python script.py batch_size dtype_flag seq_len d_embed num_heads"
    batch_size, dtype_flag, seq_len, d_embed, num_heads = \
        int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])

    dtype = get_dtype(dtype_flag)
    profile_sdpa_rep(batch_size, dtype, seq_len, d_embed, num_heads)
