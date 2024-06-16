import torch
from torch.utils._python_dispatch import TorchDispatchMode
import torch.nn.functional as F

class MyDispatchMode(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        print(f"function: {func.__name__}")
        return func(*args, **kwargs or {})

# s = torch.randn(224, 224, requires_grad=True)
# t = torch.randn(224, 224, requires_grad=False)
# output = torch.mm(s, t)

# For SDPA: QKV all require grad.
# For conv: just the kernel.

# Optionally use the context manager to ensure one of the fused kernels is run
query = torch.rand(32, 8, 128, 64, dtype=torch.float16, device="cuda", requires_grad=True)
key = torch.rand(32, 8, 128, 64, dtype=torch.float16, device="cuda", requires_grad=True)
value = torch.rand(32, 8, 128, 64, dtype=torch.float16, device="cuda", requires_grad=True)

with torch.backends.cuda.sdp_kernel(enable_math=False):
    output = F.scaled_dot_product_attention(query,key,value)
with MyDispatchMode():
    output.backward(output)