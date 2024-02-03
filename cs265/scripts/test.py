# import os
# import csv
# import torch
# from torch import nn
import sys

arg0 = sys.argv[1]

# file_in = os.getcwd() + "/old_addmm.csv"
# file_out = os.getcwd() + "/addmm.csv"

# with open(file_in,'r') as csvinput:
#     with open(file_out, 'w') as csvoutput:
#         writer = csv.writer(csvoutput, lineterminator='\n')
#         reader = csv.reader(csvinput)

#         all = []
#         row = next(reader)
#         row.append('Inputs')
#         all.append(row)

#         for row in reader:
#             row.append(1)
#             all.append(row)

#         writer.writerows(all)

temp = list(range(1, 10)) + list(range(10, 100, 10)) + list(range(100, 10000, 100)) + list(range(10000, 80001, 1000)) # Input should have 0.
print(len(temp))
temp2 = list(range(1, 10)) + list(range(10, 100, 10)) + list(range(100, 1001, 100))
print(len(temp2))
# print(nn.Flatten()(torch.randn(2, 5, 2, device="cuda")))

print(temp2[:3] + temp2[-2:])

print(min(2, 6))

# for i in range(4):
#     s = f"cuda:{i}"
#     device = s if torch.cuda.is_available() else "cpu"
#     assert(device == s)
#     # device = torch.device("cuda:1,3" if torch.cuda.is_available() else "cpu") ## specify the GPU id's, GPU id's start from 0.
# print(torch.cuda.device_count())

# def main():
#     # print(arg0)

# if __name__ == "__main__":
#     main()
