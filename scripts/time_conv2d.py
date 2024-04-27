import os, sys, csv
import numpy as np

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '..')))

from utils.profile_utils import get_dtype, time_model

import warnings
warnings.filterwarnings("ignore")

# Assume this script runs on one device.
device = "cuda:0" if torch.cuda.is_available() else "cpu"
assert(device == "cuda:0")

def time_conv2d_rep(batch_size, dtype, height, width, bias, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, out_file):
    input = torch.randn(batch_size, in_channels, height, width, dtype=dtype, device=device)
    weight = torch.randn(out_channels, in_channels // groups, kernel_size, kernel_size, dtype=dtype, device=device)
    bias_tensor = torch.randn(out_channels, dtype=dtype, device=device) if bias else None

    @torch.compile(backend="inductor")
    def conv2d(x, w):
        return F.conv2d(x, w, bias=bias_tensor, stride=stride, padding=padding, dilation=dilation, groups=groups)

    time = time_model(conv2d, (input, weight))

    kernel_params = f"{batch_size}.{dtype}.{height}.{width}.{bias}.{in_channels}.{out_channels}.{kernel_size}.{stride}.{padding}.{dilation}.{groups}"

    with open(out_file, mode='a', newline='') as file:
        writer = csv.writer(file)

        if os.path.getsize(out_file) == 0:
            writer.writerow(['Kernel Name', 'Latency (ms)'])
        writer.writerow([kernel_params, time])

if __name__ == "__main__":
    assert(len(sys.argv) == 14)
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

    dtype = get_dtype(dtype_flag)
    if padding.lower() == 'same':
        padding = (kernel_size // 2, kernel_size // 2)  # Adjust padding for 'same', only valid if stride is 1
    elif padding.lower() == 'valid':
        padding = 0

    out_file = str(sys.argv[13])

    time_conv2d_rep(batch_size, dtype, height, width, bias, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, out_file)