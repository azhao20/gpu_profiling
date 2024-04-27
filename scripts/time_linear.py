import os, sys, csv

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '..')))

from utils.profile_utils import get_dtype, time_addmm

import warnings
warnings.filterwarnings("ignore")

# Assume this script runs on one device.
device = "cuda:0" if torch.cuda.is_available() else "cpu"
assert(device == "cuda:0")

def time_linear_rep(batch_size, dtype, bias, in_size, out_size, out_file):
    # TODO: what about non randn?
    A = torch.randn(batch_size, in_size, dtype=dtype, device=device)
    B = torch.randn(in_size, out_size, dtype=dtype, device=device)
    C = torch.randn(out_size, dtype=dtype, device=device)

    args = (A, B, C) if bias else (A, B)
    time = time_addmm(*args)

    kernel_params=f"{batch_size}.{dtype}.{bias}.{in_size}.{out_size}"

    # Save the result.
    with open(out_file, mode='a', newline='') as file:
        writer = csv.writer(file)

        if os.path.getsize(out_file) == 0:
            writer.writerow(['Kernel Name', 'Latency (ms)'])
        # Append the new row to the CSV file
        writer.writerow([kernel_params, time])

if __name__ == "__main__":
    assert(len(sys.argv) == 7)
    batch_size, dtype_flag, bias, in_size, out_size = int(sys.argv[1]), int(sys.argv[2]), bool(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
    dtype = get_dtype(dtype_flag)

    out_file = str(sys.argv[6])

    time_linear_rep(batch_size, dtype, bias, in_size, out_size, out_file)