import os,sys
import traceback as tb

import torch
from torch import nn
from torch.cuda import nvtx
# import torch._dynamo as dynamo

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '..')))

from utils.profile_utils import get_precision, WARMUP_REPS

import warnings
warnings.filterwarnings("ignore")

def main():
    """
    The features that are tested in a linear layer.

    Batch size, input size, output size: powers of 2 up to 1024 + multiples of 4
    """
    # Assume this script runs on one device.
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    assert(device == "cuda:0")

    assert(len(sys.argv) == 6)
    inputs, precision_flag, bias, in_size, out_size = int(sys.argv[1]), int(sys.argv[2]), bool(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
    precision = get_precision(precision_flag)

    A = torch.randn(inputs, in_size, dtype=precision, device=device)
    B = torch.randn(in_size, out_size, dtype=precision, device=device)
    C = torch.randn(out_size, dtype=precision, device=device)

    # class Linear(nn.Module):
    #     def __init__(self):
    #         super().__init__()
    #         self.lin = nn.Linear(in_size, out_size, bias=bias)

    #     def forward(self, x):
    #         return self.lin(x)

    # Have to put the function inside of the if statement.
    if bias:
        @torch.compile(backend="inductor")
        def addmm(a, b, bias):
            return torch.addmm(bias, a, b)

        model = addmm
        args = (A, B, C)
    else:
        @torch.compile(backend="inductor")
        def mm(a, b):
            return torch.mm(a, b)

        model = mm
        args = (A, B)

    # model = Linear().to(device, dtype=precision)
    # model = torch.compile(model, backend="inductor")
    # model.eval()

    res = [0] * (WARMUP_REPS + 1)
    torch.cuda.empty_cache()
    try:
        with torch.no_grad():
            # Do all of the fusion heuristics, so the later call won't need to.
            for i in range(WARMUP_REPS):
                res[i] = model(*args)

            torch.cuda.cudart().cudaProfilerStart()
            nvtx.range_push("profile_range")
            res[-1] = model(*args)
            nvtx.range_pop()
            torch.cuda.cudart().cudaProfilerStop()

            # for i in range(WARMUP_REPS):
            #     res[i] = model(A)
            # torch.cuda.cudart().cudaProfilerStart()
            # nvtx.range_push("profile_range")
            # res[-1] = model(A)
            # nvtx.range_pop()
            # torch.cuda.cudart().cudaProfilerStop()
    except:
        print("Failed!")
        tb.print_exc()

if __name__ == "__main__":
    main()