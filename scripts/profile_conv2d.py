import sys
import traceback as tb

import torch
from torch import nn
from torch.cuda import nvtx

from utils import get_precision, NREPS

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
    B = torch.randn(inputs, in_size, dtype=precision, device=device)

    class Linear(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(in_size, out_size, bias=bias)

        def forward(self, x):
            return self.lin(x)

    model = Linear().to(device, dtype=precision)
    model = torch.compile(model, backend="inductor")
    res = [0] * (NREPS + 1)

    torch.cuda.empty_cache()
    try:
        # Do all of the fusion heuristics, so the later call won't need to.
        for i in range(NREPS):
            res[i] = model(A)
        torch.cuda.cudart().cudaProfilerStart()
        nvtx.range_push("profile_range")
        res[-1] = model(B)
        nvtx.range_pop()
        torch.cuda.cudart().cudaProfilerStop()
    except:
        print("Failed!")
        tb.print_exc()

if __name__ == "__main__":
    main()