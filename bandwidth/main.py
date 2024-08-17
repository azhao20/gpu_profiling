import argparse
import os
import torch

import numpy as np
import pandas as pd

from itertools import product
from tqdm import tqdm

class ProfileBandwidth:
    """
    Used to get the maximum performance for a GPU.

    Maximum bandwidth:
        A100: 2039 GB/s.
    """

    def __init__(
        self,
        gpu: str,
        smallest_size: int = 2**8,
        largest_size: int = 2**17, # 2^34 = O(17e9), 2^36 = O(69e9). Remember: need to 4x.
        granularity: int = 1024,
        warmup_reps: int = 2,
        profile_reps: int = 10,
    ):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        assert self.device == "cuda:0"

        self.gpu = gpu

        self.smallest_size = smallest_size
        self.largest_size = largest_size
        self.granularity = granularity

        self.WARMUP_REPS = warmup_reps
        self.PROFILE_REPS = profile_reps
        self.NREPS = self.WARMUP_REPS + self.PROFILE_REPS
        
        self.dtype_map = {
            # 8: torch.float8_e5m2, # No support for float8 yet.
            "b16": torch.bfloat16,
            "16": torch.float16,
            "32": torch.float32,
        }

    def _time_fn(self, size: int, dtype: str) -> np.float32:
        A = torch.randn(size, size, device=self.device, dtype=self.dtype_map[dtype])

        starts = [torch.cuda.Event(enable_timing=True) for _ in range(self.NREPS)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(self.NREPS)]

        with torch.no_grad():
            for i in range(self.NREPS):
                torch.cuda.synchronize()
                starts[i].record()

                _ = torch.relu(A)

                ends[i].record()
                torch.cuda.synchronize()

                torch.cuda.empty_cache()
        del A
        torch.cuda.empty_cache()

        return np.median(
            [
                starts[i].elapsed_time(ends[i])
                for i in range(self.WARMUP_REPS, self.NREPS)
            ]
        )

    def time(self, out_dir) -> None:
        """
        Returns time in ms.
        """
        results = []
        for size, dtype in tqdm(
            product(range(self.smallest_size, self.largest_size + 1, self.granularity), self.dtype_map.keys())
        ):
            try:
                time = self._time_fn(size, dtype)
            except Exception as e:
                time = e
            results.append([dtype, size, time])
        df = pd.DataFrame(results, columns=["dtype", "Size", "Time"])

        out_file = os.path.join(out_dir, f"{self.gpu}.csv")
        df.to_csv(out_file, index=False)


def get_args():
    parser = argparse.ArgumentParser(description="Profile Max Performance on GPU type.")

    parser.add_argument(
        "--gpu",
        type=str,
        required=True,
        help="Name of the GPU to profile.",
    )

    parser.add_argument("--out_dir", type=str, required=True, help="Output directory.")

    return parser.parse_args()


def main():
    args = get_args()
    ProfileBandwidth(args.gpu).time(args.out_dir)


if __name__ == "__main__":
    main()
