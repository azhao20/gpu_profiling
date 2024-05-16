import torch
from torch.nn.functional import scaled_dot_product_attention
from torch.nn.attention import SDPBackend, sdpa_kernel

q32 = torch.randn(32, 32, 32, 32, dtype=torch.float32, device="cuda")
q16 = torch.randn(32, 32, 32, 32, dtype=torch.float16, device="cuda")
qb16 = torch.randn(32, 32, 32, 32, dtype=torch.bfloat16, device="cuda")

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
    fn(context, q16, q16, q16)
    fn(context, qb16, qb16, qb16)
    fn(context, q32, q32, q32)
