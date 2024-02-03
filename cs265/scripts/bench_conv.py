import torch
from torch import nn
from torch.profiler import profile, record_function, ProfilerActivity
import torch._dynamo as dynamo
import os, sys, csv


## Block prints
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def cf(num1,num2):
    n=[]
    for i in range(1, min(num1, num2)+1): 
        if num1%i==num2%i==0: 
            n.append(i)

    if len(n) > 6:
        n = n[:3] + n[-3:]
    return n


def main():
    device = f"cuda:{sys.argv[1]}" if torch.cuda.is_available() else "cpu"
    assert(device == f"cuda:{sys.argv[1]}")

    procNum = int(sys.argv[1])
    nodeNum = int(sys.argv[2])

    ## Write the result
    file_name = os.getcwd() + f"/zhao_conv2d/conv2d_2d{procNum}{nodeNum}.csv"

    channels = list(range(0, 12, 2))
    biases = [True] if nodeNum else [False]
        
    if procNum == 0:
        inputs = [1]
        # padding_modes = ['zeros']
    elif procNum == 1:
        inputs = [3]
        # padding_modes = ['reflect']
    elif procNum == 2:
        inputs = [5]
        # padding_modes = ['replicate']
    elif procNum == 3:
        inputs = [7]
        # padding_modes = ['circular']

    # padding_modes = ('zeros', 'reflect', 'replicate', 'circular')
    torch.cuda.empty_cache()

    with open (file_name, 'a') as f:
        writer = csv.writer(f)
        
        for num_inputs in inputs:
            for bias in biases:
                for N in range(50, 301, 50): 
                    for M in range(50, 301, 50):
                        # for L in range(50, 301, 50):
                        for k in channels:
                            for l in channels: 
                                in_channel, out_channel = 2 ** k, 2 ** l
                                groups = min(k, l) + 1 # common factors of powers of two are also powers of two.
                                for window in range(1, 12, 2): # Kernel size. Note that we only allow square kernels of odd lengths.
                                    for stride in range(1, 6, 2): 
                                        for padding in range(0, 6, 2):
                                            for dilation in (1, min(window, 6), 2):
                                                for grp in range(0, groups, 2):     
                                                    # for padding_mode in padding_modes: # Ignore for now.
                                                    
                                                    group = 2 ** grp
                                                    # N, M, in_channel, out_channel, window, stride = 800, 800, 50, 90, 3

                                                    print(num_inputs, bias, N, M, in_channel, out_channel, window, stride, padding, dilation, group)

                                                    A = torch.randn(num_inputs, in_channel, N, M, device=device) + 2
                                                    # B = torch.randn(in_channel, N, device=device) + 2
                                                    # C = torch.randn(in_channel, N, M, L, device=device) + 2

                                                    class CNN(nn.Module):
                                                        def __init__(self):
                                                            super().__init__()
                                                            # self.conv1d = nn.Conv1d(in_channel, out_channel, window, stride=stride, padding=padding, dilation=dilation, groups=group, bias=bias, padding_mode = padding_mode) ## Only test windows of the same size.
                                                            self.conv2d = nn.Conv2d(in_channel, out_channel, window, stride=stride, padding=padding, dilation=dilation, groups=group, bias=bias) ## Only test windows of the same size.
                                                            # self.conv3d = nn.Conv3d(in_channel, out_channel, window, stride=stride, padding=padding, dilation=dilation, groups=group, bias=bias, padding_mode = padding_mode) ## Only test windows of the same size.

                                                        def forward(self, x):
                                                            # return self.conv1d(x)
                                                            return self.conv2d(x)
                                                            # return self.conv3d(x)

                                                    model = CNN().to(device)
                                                    opt_model = dynamo.optimize("inductor", nopython=True)(model)
                                                    print("-------------Optimized model-------------------\n", opt_model)

                                                    # with HiddenPrints():
                                                    _ = opt_model(A)

                                                    ## Hmm: need to calculate flops here.
                                                    writer.writerow([num_inputs, bias, N, M, in_channel, out_channel, window, stride, padding, dilation, group])

                                                    ## Clear some memory
                                                    del model
                                                    del opt_model
                                                    del A
                                                    dynamo.reset()
                                                    torch.cuda.empty_cache()

                                                    # break
                                                    break
                                                break
                                            break
                                        break
                                    break
                                break
                                f.flush()
                            break
                        break # Keep this uncommented for 2D inputs
                        # break 
                    break
                break
            break


if __name__ == "__main__":
    main()
