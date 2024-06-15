import os, csv
import torch

from utils.profile_utils import ProfileBase, get_args_mm, mm_sizes

import warnings
warnings.filterwarnings("ignore")

class ProfileBMM(ProfileBase):
    def __init__(self, sizes = []):
        """
        For now, we don't use the other hyperparameters.
        """
        super().__init__()
        self.sizes = sizes
        self.batches = [i for i in range(32, 512+1, 32)]
        # self.sizes = [32, 224]
        # self.batches = [32]

    def get_input_sizes(self, args) -> list:
        A_size = torch.Size([args.b, args.n, args.m])
        B_size = torch.Size([args.b, args.m, args.p])
        return [A_size, B_size]
    
    def get_output_size(self, args):
        return torch.Size([args.b, args.n, args.p])

    def get_fn(self, args):
        """
        TODO: if use_inductor == True, then we might be doing
        redundant work in lowering the fn multiple times, since mm
        doesn't depend on the hyperparameters.
        """
        if args.use_inductor:
            raise ValueError("Not using Inductor for now.")
            @torch.compile(backend="inductor")
            def bmm(a, b):
                return torch.bmm(a, b)
            fn = bmm
        else:
            fn = torch.bmm
        return fn
    
    def time(self, args):
        """
        Could consider a param generator.
        """
        with open(args.out_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            if os.path.getsize(args.out_file) == 0:
                writer.writerow(self.time_header)

            for dname in self.dtype_map:
                for b in self.batches:
                    # print(f"{dname}, {b}---------------------------")
                    for m in self.sizes:
                        for p in self.sizes:
                            args.b = b
                            args.m = m
                            args.p = p
                            args.dtype = dname
                            kernel_params=f"{args.dtype}.{args.b}.{args.n}.{args.m}.{args.p}"
                            writer.writerow([kernel_params, self.time_rep(args)])
                    # Flush intermittently in case something crashes
                    file.flush()

def main():
    args = get_args_mm()
    if args.mode == "time":
        ProfileBMM(mm_sizes).time(args)
    else:
        # Don't need mm_sizes if NCU runs once per program.
        ProfileBMM().profile(args)

if __name__ == "__main__":
    main()
