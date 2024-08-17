import argparse
import os
import torch
import tqdm

import numpy as np
import pandas as pd


class ProfileBandwidth:
    """
    Used to get the maximum performance for a GPU.

    Maximum bandwidth:
        A100: 2039 GB/s.
    """

    def __init__(
        self,
        gpu: str,
        smallest_size: int = 15,
        largest_size: int = 33,
        granularity: int = 512,
        warmup_reps: int = 2,
        profile_reps: int = 10,
    ):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        assert self.device == "cuda:0"

        self.gpu = gpu

        self.WARMUP_REPS = warmup_reps
        self.PROFILE_REPS = profile_reps
        self.NREPS = self.WARMUP_REPS + self.PROFILE_REPS

        self.smallest_size = smallest_size
        self.largest_size = largest_size
        self.granularity = granularity

    def _time_fn(self, size) -> np.float32:
        A = torch.randn(size, size, dtype=np.float32, device=self.device)

        starts = [torch.cuda.Event(enable_timing=True) for _ in range(self.NREPS)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(self.NREPS)]

        torch.cuda.empty_cache()
        for i in range(self.NREPS):
            starts[i].record()
            result = torch.relu(A)
            ends[i].record()
            del result
        torch.cuda.synchronize()

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
        for size in tqdm(
            range(self.smallest_size, self.largest_size + 1, self.granularity)
        ):
            try:
                time = self._time_fn(size)
            except Exception as _:
                time = -1
            results.append([size**2, time])
        df = pd.DataFrame(results, columns=["FLOPs", "Time"])

        out_file = os.path.join(out_dir, f"{self.gpu}.csv")
        df.to_csv(out_file)


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
