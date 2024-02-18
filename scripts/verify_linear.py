import os, sys, csv
import traceback as tb

import torch
from torch import nn
# import torch._dynamo as dynamo

from utils import get_precision, time_model

import warnings
warnings.filterwarnings("ignore")

def main():
    # Assume this script runs on one device.
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    assert(device == "cuda:0")

    assert(len(sys.argv) == 7)
    inputs, precision_flag, bias, in_size, out_size = int(sys.argv[1]), int(sys.argv[2]), bool(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
    out_file = str(sys.argv[6])

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

    torch.cuda.empty_cache()
    time = time_model(model, A, B)

    kernel_params = ""
    for param in (inputs, precision_flag, str(sys.argv[3]), in_size, out_size):
        kernel_params += f"{param}" + "."
    kernel_params = kernel_params[:-1] # Remove the trailing period.

    # Save the result.
    with open(out_file, mode='a', newline='') as file:
        writer = csv.writer(file)

        if os.path.getsize(out_file) == 0:
            header = ['Kernel Name', 'Latency (ms)']
            writer.writerow(header)
        # Append the new row to the CSV file
        writer.writerow([kernel_params, time])

if __name__ == "__main__":
    main()