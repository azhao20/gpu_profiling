from abc import abstractmethod
import pandas as pd
import os
import math

from torch.utils.flop_counter import sdpa_flop_count, conv_flop_count
from tqdm import tqdm

from itertools import product

class TimeProcessorBase:
    kernel_type: str = "UNINITIALIZED"

    def __init__(self, base_dir: str):
        """base_dir points to the repository root."""
        self.base_dir = base_dir

    @property
    def data_dir(self) -> str:
        return os.path.join(self.base_dir, "data/final", self.kernel_type)

    @abstractmethod
    def split_params(self, row) -> pd.Series:
        """Time dataframe doesn't have kernel parameters separated out"""
        pass

    @abstractmethod
    def get_data(self, sample_rate: float = 0.5) -> pd.DataFrame:
        """Since the resulting dataframe might be large, we sample `sample_rate` from each strata"""
        pass


class TimeProcessorMM(TimeProcessorBase):
    def __init__(self, base_dir: str):
        super().__init__(base_dir)
        self.kernel_type = "mm"
        self.n_values = (
            list(range(16, 496, 16))
            + list(range(512, 1920, 128))
            + list(range(2048, 3584, 512))
            + list(range(4096, 32768, 1024))
        )

    def split_params(self, row):
        params = row["kernel_params"].split(".")
        return pd.Series(
            [row["kernel_params"], *params, row["time"]],
            index=["kernel_params", "dtype", "n", "m", "p", "time"],
        )

    def get_data(self, sample_rate: float = 0.5):
        dfs = []
        for n in tqdm(self.n_values):
            file_path = os.path.join(self.data_dir, f"time.{n}.pkl")
            df = pd.read_pickle(file_path).sample(frac=sample_rate)
            df.rename(
                columns={"Kernel Name": "kernel_params", "Latency (ms)": "time"},
                inplace=True,
            )
            dfs.append(df.apply(self.split_params, axis=1))

        dfs = pd.concat(dfs, axis=0, ignore_index=True)
        dfs["dtype"] = dfs["dtype"].astype("category")
        dfs[["n", "m", "p"]] = dfs[["n", "m", "p"]].astype(int)
        dfs["time"] = dfs["time"].astype(float)

        dfs["gflops"] = 2 * dfs["n"] * dfs["m"] * dfs["p"] / (10**9)
        return dfs


class TimeProcessorBMM(TimeProcessorBase):
    def __init__(self, base_dir: str):
        super().__init__(base_dir)
        self.kernel_type = "bmm"
        self.n_values = (
            list(range(16, 496, 16))
            + list(range(512, 1920, 128))
            + list(range(2048, 3584, 512))
            + list(range(4096, 32768, 1024))
        )

    def split_params(self, row) -> pd.Series:
        params = row["kernel_params"].split(".")
        return pd.Series(
            [row["kernel_params"], *params, row["time"]],
            index=["kernel_params", "dtype", "b", "n", "m", "p", "time"],
        )

    def get_data(self, sample_rate: float = 0.5):
        dfs = []
        for n in tqdm(self.n_values):
            file_path = os.path.join(self.data_dir, f"time.{n}.pkl")
            df = pd.read_pickle(file_path).sample(frac=sample_rate)
            df.rename(
                columns={"Kernel Name": "kernel_params", "Latency (ms)": "time"},
                inplace=True,
            )
            dfs.append(df.apply(self.split_params, axis=1))

        dfs = pd.concat(dfs, axis=0, ignore_index=True)
        dfs["dtype"] = dfs["dtype"].astype("category")
        dfs[["b", "n", "m", "p"]] = dfs[["b", "n", "m", "p"]].astype(int)
        dfs["time"] = dfs["time"].astype(float)

        dfs["gflops"] = 2 * dfs["b"] * dfs["n"] * dfs["m"] * dfs["p"] / (10**9)
        return dfs


class TimeProcessorSDPA(TimeProcessorBase):
    def __init__(self, base_dir: str, is_forward: bool = True):
        super().__init__(base_dir)
        self.kernel_type = "sdpa" if is_forward else "sdpa_backward"
        self.num_heads = [4, 8, 12, 16]

    def split_params(self, row) -> pd.Series:
        params = row["kernel_params"].split(".")
        return pd.Series(
            [row["kernel_params"], *params, row["time"]],
            index=[
                "kernel_params",
                "dtype",
                "backend",
                "b",
                "h",
                "s_q",
                "s_kv",
                "d_qk",
                "d_v",
                "is_causal",
                "time",
            ],
        )

    def compute_flops(self, df: pd.DataFrame) -> pd.DataFrame:
        flops = df.apply(
            lambda row: sdpa_flop_count(
                (row["b"], row["h"], row["s_q"], row["d_qk"]),
                (row["b"], row["h"], row["s_kv"], row["d_qk"]),
                (row["b"], row["h"], row["s_kv"], row["d_v"]),
            ),
            axis=1,
        )
        df["gflops"] = flops / (10**9)
        return df

    def get_data(self, sample_rate: float = 0.5) -> pd.DataFrame:
        dfs = []

        dtypes = ["b16", "16"]
        backends = ["flash", "efficient"]

        for dtype, backend, h in tqdm(product(dtypes, backends, self.num_heads)):
            file_path = os.path.join(self.data_dir, f"time.{dtype}.{backend}.{h}.pkl")
            df = pd.read_pickle(file_path).sample(frac=sample_rate)
            df.rename(
                columns={
                    "Kernel Name": "kernel_params",
                    "Latency (ms)": "time",
                },
                inplace=True,
            )
            dfs.append(df.apply(self.split_params, axis=1))

        dtype = "32"
        backend = "efficient"
        for h in tqdm(self.num_heads):
            file_path = os.path.join(self.data_dir, f"time.{dtype}.{backend}.{h}.pkl")
            df = pd.read_pickle(file_path).sample(frac=sample_rate)
            df.rename(
                columns={"Kernel Name": "kernel_params", "Latency (ms)": "time"},
                inplace=True,
            )
            dfs.append(df.apply(self.split_params, axis=1))

        dfs = pd.concat(dfs, axis=0, ignore_index=True)
        dfs["dtype"] = dfs["dtype"].astype("category")
        dfs[["b", "h", "s_q", "s_kv", "d_qk", "d_v"]] = dfs[
            ["b", "h", "s_q", "s_kv", "d_qk", "d_v"]
        ].astype(int)
        dfs["time"] = dfs["time"].astype(float)

        return self.compute_flops(dfs)


class TimeProcessorConv2d(TimeProcessorBase):
    def __init__(self, base_dir: str, is_forward: bool = True):
        super().__init__(base_dir)
        self.kernel_type = "conv2d" if is_forward else "conv2d_backward"
        self.iH = self.iW = [2, 8, 32, 128, 512, 1024]
        self.transposed = [0, 1]

    def split_params(self, row):
        params = row["kernel_params"].split(".")
        return pd.Series(
            [row["kernel_params"], *params, row["time"]],
            index=[
                "kernel_params",
                "dtype",
                "b",
                "in_channels",
                "iH",
                "iW",
                "out_channels",
                "groups",
                "kH",
                "kW",
                "stride",
                "dilation",
                "transposed",
                "time",
            ],
        )

    def compute_out_shape(
        self,
        x_shape: list[int],
        w_shape: list[int],
        stride: int,
        padding: int = 0,
        dilation: int = 1,
        transposed: bool = False,
        output_padding: int = 0,
    ) -> list[int]:
        batch_size, _, iH, iW = x_shape
        c_out, _, kH, kW = w_shape

        if transposed:
            oH = (
                stride * (iH - 1)
                + dilation * (kH - 1)
                + output_padding
                - 2 * padding
                + 1
            )
            oW = (
                stride * (iW - 1)
                + dilation * (kW - 1)
                + output_padding
                - 2 * padding
                + 1
            )
        else:
            oH = math.floor((iH + 2 * padding - dilation * (kH - 1) - 1) / stride + 1)
            oW = math.floor((iW + 2 * padding - dilation * (kW - 1) - 1) / stride + 1)

        return [batch_size, c_out, oH, oW]

    def compute_flops(self, df: pd.DataFrame) -> pd.DataFrame:
        flops = df.apply(
            lambda row: conv_flop_count(
                x_shape=[row["b"], row["in_channels"], row["iH"], row["iW"]],
                w_shape=[row["out_channels"], row["in_channels"], row["kH"], row["kW"]],
                out_shape=self.compute_out_shape(
                    x_shape=[row["b"], row["in_channels"], row["iH"], row["iW"]],
                    w_shape=[
                        row["out_channels"],
                        row["in_channels"],
                        row["kH"],
                        row["kW"],
                    ],
                    stride=row["stride"],
                    padding=0,
                    dilation=row["dilation"],
                    transposed=bool(row["transposed"]),
                    output_padding=0,
                ),
                transposed=bool(row["transposed"]),
            ),
            axis=1,
        )
        df["gflops"] = flops / (10**9)
        return df

    def get_data(self, sample_rate: float = 0.5) -> pd.DataFrame:
        """
        NOTE: for now, we ignore all csv's with -1 through -4, which denote errors.
        """
        dfs = []

        for iH, iW, transposed in tqdm(product(self.iH, self.iW, self.transposed)):
            file_name = f"time.{iH}.{iW}.{transposed}.pkl"
            file_path = os.path.join(self.data_dir, file_name)
            df = pd.read_pickle(file_path).sample(frac=sample_rate)
            df.rename(
                columns={
                    "Kernel Name": "kernel_params",
                    "Latency (ms)": "time",
                },
                inplace=True,
            )

            if (df["time"] < 0).sum() > 0:
                print(f"< 0 found in file {file_name}")
                continue
            dfs.append(df.apply(self.split_params, axis=1))

        dfs = pd.concat(dfs, axis=0, ignore_index=True)
        dfs["dtype"] = dfs["dtype"].astype("category")
        dfs[
            [
                "b",
                "in_channels",
                "iH",
                "iW",
                "out_channels",
                "groups",
                "kH",
                "kW",
                "stride",
                "dilation",
            ]
        ] = dfs[
            [
                "b",
                "in_channels",
                "iH",
                "iW",
                "out_channels",
                "groups",
                "kH",
                "kW",
                "stride",
                "dilation",
            ]
        ].astype(
            int
        )
        dfs["time"] = dfs["time"].astype(float)
        return self.compute_flops(dfs)