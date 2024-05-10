import os,sys
import traceback as tb

import torch
from torch.cuda import nvtx
# import torch._dynamo as dynamo

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '..')))

from utils.profile_utils import get_dtype, WARMUP_REPS

import warnings
warnings.filterwarnings("ignore")

# Assume this script runs on one device.
device = "cuda:0" if torch.cuda.is_available() else "cpu"
assert(device == "cuda:0")

def profile_linear_rep(batch_size, dtype, bias, in_size, out_size):
    """
    The features that are tested in a linear layer.
    """
    # TODO: what about non randn?
    A = torch.randn(batch_size, in_size, dtype=dtype, device=device)
    B = torch.randn(in_size, out_size, dtype=dtype, device=device)
    C = torch.randn(out_size, dtype=dtype, device=device)

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

    res = [0] * (WARMUP_REPS + 1)
    torch.cuda.empty_cache()
    try:
        # Inductor computes backward pass, so don't track grads.
        with torch.no_grad():
            # Do all of the fusion heuristics, so the later call won't need to.
            for i in range(WARMUP_REPS):
                res[i] = model(*args)

            torch.cuda.cudart().cudaProfilerStart()
            nvtx.range_push("profile_range")
            res[-1] = model(*args)
            nvtx.range_pop()
            torch.cuda.cudart().cudaProfilerStop()
    except:
        print("Failed!")
        tb.print_exc()

if __name__ == "__main__":
    """
    azhao: can do this processing anywhere, e.g., when reading a pd.DataFrame.
    """
    assert(len(sys.argv) == 6)
    batch_size, dtype_flag, bias, in_size, out_size = int(sys.argv[1]), int(sys.argv[2]), bool(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
    dtype = get_dtype(dtype_flag)

    profile_linear_rep(batch_size, dtype, bias, in_size, out_size)
