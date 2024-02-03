import sys
import traceback as tb

import torch
from torch import nn
from torch.cuda import nvtx
import torch._dynamo as dynamo

def main():
    """
    The features that are tested in a linear layer.

    Batch size, input size, output size: powers of 2 up to 1024 + multiples of 4
    """
    # Assume this script runs on one device.
    device = f"cuda:0" if torch.cuda.is_available() else "cpu"
    assert(device == f"cuda:0")

    assert(len(sys.argv) >= 3)
    precision_int, inputs, bias, in_size, out_size = int(sys.argv[1]), int(sys.argv[2]), bool(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])

    # TODO: put this into utils.py.
    precision_map = {
        # 8: torch.float8_e5m2,
        161: torch.bfloat16,
        162: torch.float16,
        32: torch.float32
    }
    if precision_int not in precision_map:
        print("Precision wasn't specified, defaulting to torch.float32")

    precision = precision_map.get(precision_int, torch.float32)
    A = torch.randn(inputs, in_size, dtype=precision, device=device)

    class Linear(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(in_size, out_size, bias=bias)

        def forward(self, x):
            return self.lin(x)

    model = Linear().to(device, dtype=precision)
    model = dynamo.optimize("inductor", nopython=True)(model)

    torch.cuda.empty_cache()
    try:
        torch.cuda.cudart().cudaProfilerStart()
        nvtx.range_push("profile_range")
        res = model(A)
        torch.cuda.synchronize()
        nvtx.range_pop()
        torch.cuda.cudart().cudaProfilerStop()
    except:
        print("Failed!")
        tb.print_exc()

if __name__ == "__main__":
    main()