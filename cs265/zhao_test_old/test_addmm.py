
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




# import builtins
# import functools
# import inspect
# import itertools
# import logging
# import sys
# import textwrap
# import time
# from io import StringIO

# from typing import Any, List
# from unittest.mock import patch

# import sympy

# import torch
# from torch._dynamo.testing import rand_strided
# from torch._dynamo.utils import counters, identity

# from torch._inductor import config, ir
# from torch._inductor.autotune_process import BenchmarkRequest, TensorMeta
# from torch._inductor.codecache import code_hash, PersistentCache, PyCodeCache

# from torch._inductor.codegen.common import IndentedBuffer
# from torch._inductor.codegen.triton import (
#     config_of,
#     signature_of,
#     texpr,
#     TritonKernel,
#     TritonPrinter,
#     TritonScheduling,
# )

# from torch._inductor.utils import do_bench, sympy_dot, sympy_product
# from torch._inductor.virtualized import V








async_compile.wait(globals())
del async_compile

def call(args):
    from torch._inductor.select_algorithm import extern_kernels
    primals_1, primals_2, primals_3 = args
    args.clear()
    start_graph()
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty_strided((1, 512), (512, 1), device='cuda', dtype=torch.float32)
        extern_kernels.addmm(primals_2, as_strided(primals_3, (1, 784), (784, 1)), as_strided(primals_1, (784, 512), (1, 784)), alpha=1, beta=1, out=buf0)
        del primals_1
        del primals_2
        end_graph()
        return (buf0, as_strided(primals_3, (1, 784), (784, 1)), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((512, 784), (784, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((1, 28, 28), (784, 28, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_3]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.utils import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)


## Copied CUDA code for testing
## Problem: circular imports

# from torch._dynamo.testing import rand_strided
# from torch import empty_strided, as_strided, device
# from torch._inductor.codecache import AsyncCompile
# from torch._inductor.select_algorithm import extern_kernels

# primals_1 = rand_strided((512, 784), (784, 1), device='cuda:0', dtype=torch.float32)
# primals_2 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
# primals_3 = rand_strided((10, 512), (512, 1), device='cuda:0', dtype=torch.float32)
# primals_4 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
# primals_5 = rand_strided((1, 28, 28), (784, 28, 1), device='cuda:0', dtype=torch.float32)

# buf0 = empty_strided((1, 512), (512, 1), device='cuda', dtype=torch.float32)
# print(type())
# print(extern_kernels.addmm(primals_2, as_strided(primals_5, (1, 784), (784, 1)), as_strided(primals_1, (784, 512), (1, 784)), alpha=1, beta=1, out=buf0))

