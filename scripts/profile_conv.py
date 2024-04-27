import os, sys
import traceback as tb

import torch
import torch.nn.functional as F
from torch.cuda import nvtx

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '..')))
from utils.profile_utils import get_dtype, WARMUP_REPS

import warnings
warnings.filterwarnings("ignore")

# Assume this script runs on one device.
device = "cuda:0" if torch.cuda.is_available() else "cpu"
assert(device == "cuda:0")

def profile_conv2d_rep(batch_size, dtype, height, width, bias, in_channels, out_channels, kernel_size, stride, padding, dilation, groups):
    """
    The features that are tested in a conv2d layer.
    """
    # Create random inputs and weights based on the input and output sizes
    input = torch.randn(batch_size, in_channels, height, width, dtype=dtype, device=device)
    weight = torch.randn(out_channels, in_channels // groups, kernel_size, kernel_size, dtype=dtype, device=device)
    bias_tensor = torch.randn(out_channels, dtype=dtype, device=device) if bias else None

    @torch.compile(backend="inductor")
    def conv2d(x, w):
        return F.conv2d(x, w, bias=bias_tensor, stride=stride, padding=padding, dilation=dilation, groups=groups)

    res = [0] * (WARMUP_REPS + 1)
    torch.cuda.empty_cache()
    try:
        # Inductor computes backward pass, so don't track grads.
        with torch.no_grad():
            # Do all of the fusion heuristics, so the later call won't need to.
            for i in range(WARMUP_REPS):
                res[i] = conv2d(input, weight)

            torch.cuda.cudart().cudaProfilerStart()
            nvtx.range_push("profile_range")
            res[-1] = conv2d(input, weight)
            nvtx.range_pop()
            torch.cuda.cudart().cudaProfilerStop()
    except:
        print("Failed!")
        tb.print_exc()

if __name__ == "__main__":
    """
    Command line argument processing for profiling convolutional layers.
    """
    assert(len(sys.argv) == 13)
    batch_size = int(sys.argv[1])
    dtype_flag = int(sys.argv[2])
    height = int(sys.argv[3])
    width = int(sys.argv[4])
    bias = bool(int(sys.argv[5]))
    in_channels = int(sys.argv[6])
    out_channels = int(sys.argv[7])
    kernel_size = int(sys.argv[8])
    stride = int(sys.argv[9])
    padding = sys.argv[10]
    dilation = int(sys.argv[11])
    groups = int(sys.argv[12])

    dtype = get_dtype(dtype_flag)
    if padding.lower() == 'same':
        padding = (kernel_size // 2, kernel_size // 2)  # Adjust padding for 'same', only valid if stride is 1
    elif padding.lower() == 'valid':
        padding = 0

    profile_conv2d_rep(batch_size, dtype, height, width, bias, in_channels, out_channels, kernel_size, stride, padding, dilation, groups)
