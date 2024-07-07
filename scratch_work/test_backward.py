import torch
from torch.utils._python_dispatch import TorchDispatchMode
import torch.nn.functional as F

from torch.nn.functional import scaled_dot_product_attention, conv_transpose2d
from torch.nn.attention import SDPBackend, sdpa_kernel

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
# query = torch.rand(32, 8, 128, 64, dtype=torch.float16, device="cuda", requires_grad=True)
# key = torch.rand(32, 8, 128, 64, dtype=torch.float16, device="cuda", requires_grad=True)
# value = torch.rand(32, 8, 128, 64, dtype=torch.float16, device="cuda", requires_grad=True)

# query = torch.randn(2, 8, 2, 128, dtype=torch.bfloat16, device="cuda")
# key = torch.randn(2, 8, 64, 128, dtype=torch.bfloat16, device="cuda")
# value = torch.randn(2, 8, 64, 64, dtype=torch.bfloat16, device="cuda")

# query = torch.randn(2, 8, 2, 32, dtype=torch.float16, device="cuda")
# key = torch.randn(2, 8, 32, 32, dtype=torch.float16, device="cuda")
# value = torch.randn(2, 8, 32, 64, dtype=torch.float16, device="cuda")

# with torch.backends.cuda.sdp_kernel(enable_math=False):
#     output = F.scaled_dot_product_attention(query,key,value)
# with MyDispatchMode():
#     output.backward(output)

# with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
#     print(scaled_dot_product_attention(query,key,value).shape) # , is_causal=True

# with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
#     print(scaled_dot_product_attention(query,key,value,is_causal=True).shape)

# Conv2d transposed. Takeaway: seems like we just need in_channels == out_channels.
x = torch.randn(2, 32, 1024, 512, dtype=torch.float32, device="cuda:0")
w = torch.randn(32, 64, 7, 5, dtype=torch.float32, device="cuda:0")

fn = lambda x, w: conv_transpose2d(x, w, bias=None, stride=3, dilation=2, groups=1)
print(fn(x, w).shape)