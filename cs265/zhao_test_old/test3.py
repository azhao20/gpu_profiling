from ctypes import c_void_p, c_long
import torch
import math
import random
import os
import tempfile
from torch._inductor.utils import maybe_profile

from torch import empty_strided, as_strided, device
from torch._inductor.codecache import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels

aten = torch.ops.aten
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
async_compile = AsyncCompile()

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_azhao/jr/cjroz6dqeleitkqj7466sibbsvdgoch5da3cgyaseyxcw36e27n3.py
# Original ATen: aten.relu, aten.threshold_backward

# aten.relu => relu, relu_1
# aten.threshold_backward => le_1
triton_poi_fused_relu_threshold_backward_0 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[512], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i1', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.where(0 != 0, 0, tl.where(0 > tmp0, 0, tmp0))
    tmp2 = tl.where(0 != 0, 0, tl.where(0 > tmp1, 0, tmp1))
    tmp3 = 0.0
    tmp4 = tmp1 <= tmp3
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp2, xmask)
    tl.store(out_ptr1 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp4, xmask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5 = args
    args.clear()
    start_graph()
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty_strided((1, 512), (512, 1), device='cuda', dtype=torch.float32)
        extern_kernels.addmm(primals_2, as_strided(primals_5, (1, 784), (784, 1)), as_strided(primals_1, (784, 512), (1, 784)), alpha=1, beta=1, out=buf0)
        del primals_1
        del primals_2
        buf1 = empty_strided((1, 512), (512, 1), device='cuda', dtype=torch.float32)
        buf3 = empty_strided((1, 512), (512, 1), device='cuda', dtype=torch.bool)
        stream0 = get_cuda_stream(0)
        triton_poi_fused_relu_threshold_backward_0.run(buf0, buf1, buf3, 512, grid=grid(512), stream=stream0)
        del buf0
        buf2 = empty_strided((1, 10), (10, 1), device='cuda', dtype=torch.float32)
        extern_kernels.addmm(primals_4, buf1, as_strided(primals_3, (512, 10), (1, 512)), alpha=1, beta=1, out=buf2)
        del primals_4
        end_graph()
        return (buf2, as_strided(primals_5, (1, 784), (784, 1)), buf1, as_strided(primals_3, (10, 512), (512, 1)), buf3, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((512, 784), (784, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((10, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((1, 28, 28), (784, 28, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.utils import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)