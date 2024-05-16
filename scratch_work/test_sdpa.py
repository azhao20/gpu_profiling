import torch
from torch.nn.functional import scaled_dot_product_attention
from torch.nn.attention import SDPBackend, sdpa_kernel

from torch.profiler import profile, record_function

device="cuda"

q32 = torch.randn(32, 32, 32, 32, dtype=torch.float32, device=device)
q16 = torch.randn(32, 32, 32, 32, dtype=torch.float16, device=device)
qb16 = torch.randn(32, 32, 32, 32, dtype=torch.bfloat16, device=device)

contexts = {
    "math": SDPBackend.MATH,
    "efficient": SDPBackend.EFFICIENT_ATTENTION,
    "flash": SDPBackend.FLASH_ATTENTION
}

def fn(context, q, k, v):
    with sdpa_kernel(context):
        return scaled_dot_product_attention(q, k, v)

for context in contexts.values():
    print("context:", context)
    with profile(activities=[torch.profiler.ProfilerActivity.CPU, 
                         torch.profiler.ProfilerActivity.CUDA]) as prof:
        fn(context, q16, q16, q16)
        fn(context, qb16, qb16, qb16)
        fn(context, q32, q32, q32)
    print(prof.key_averages().table(sort_by="cuda_time_total"))
