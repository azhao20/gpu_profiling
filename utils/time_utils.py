from abc import abstractmethod
import pandas as pd
import os
import math
import torch

from torch.utils.flop_counter import (
    sdpa_flop_count,
    conv_flop_count,
    sdpa_backward_flop_count,
    get_shape,
)
from tqdm import tqdm

from itertools import product


class TimeProcessorBase:
    def __init__(self, base_dir: str, kernel_type: str):
        """base_dir points to root root of the data directory."""
        self.data_dir = os.path.join(base_dir, kernel_type)

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
        super().__init__(base_dir, "mm")
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
            file_path = os.path.join(self.data_dir, f"time.{n}.parquet")
            df = pd.read_parquet(file_path).sample(frac=sample_rate)
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
        super().__init__(base_dir, "bmm")
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
            file_path = os.path.join(self.data_dir, f"time.{n}.parquet")
            df = pd.read_parquet(file_path).sample(frac=sample_rate)
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
        kernel_type = "sdpa" if is_forward else "sdpa_backward"
        super().__init__(base_dir, kernel_type)
        self.num_heads = [4, 8, 12, 16]
        self.is_forward = is_forward

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
        if self.is_forward:
            flops = df.apply(
                lambda row: sdpa_flop_count(
                    (row["b"], row["h"], row["s_q"], row["d_qk"]),
                    (row["b"], row["h"], row["s_kv"], row["d_qk"]),
                    (row["b"], row["h"], row["s_kv"], row["d_v"]),
                ),
                axis=1,
            )
        else:
            flops = df.apply(
                lambda row: sdpa_backward_flop_count(
                    grad_out_shape=(row["b"], row["h"], row["s_q"], row["d_v"]),
                    query_shape=(row["b"], row["h"], row["s_q"], row["d_qk"]),
                    key_shape=(row["b"], row["h"], row["s_kv"], row["d_qk"]),
                    value_shape=(row["b"], row["h"], row["s_kv"], row["d_v"]),
                ),
                axis=1,
            )
        df["gflops"] = flops / (10**9)
        return df

    def get_data(self, sample_rate: float = 0.5) -> pd.DataFrame:
        dfs = []

        dtypes = ["b16", "16"]
        backends = ["flash", "efficient", "cudnn"]

        for dtype, backend, h in tqdm(product(dtypes, backends, self.num_heads)):
            file_path = os.path.join(self.data_dir, f"time.{dtype}.{backend}.{h}.parquet")
            df = pd.read_parquet(file_path).sample(frac=sample_rate)
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
            file_path = os.path.join(self.data_dir, f"time.{dtype}.{backend}.{h}.parquet")
            df = pd.read_parquet(file_path).sample(frac=sample_rate)
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
        kernel_type = "conv2d" if is_forward else "conv2d_backward"
        super().__init__(base_dir, kernel_type)
        self.iH = [32, 64, 128, 224, 336, 448, 512, 784, 1120]
        self.transposed = [0, 1]
        self.group_sizes = [1, 16, 64, 128, 256, 512, 768, 1024]
        self.batch_sizes = [2, 4, 8, 16, 32]
        self.is_forward = is_forward

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

    @staticmethod
    def compute_out_shape(row: pd.Series) -> pd.Series:
        transposed = row["transposed"]
        iH = row["iH"]
        iW = row["iW"]
        stride = row["stride"]
        dilation = row["dilation"]
        kH = row["kH"]
        kW = row["kW"]
        padding = row.get("padding", 0)
        output_padding = row.get("output_padding", 0)

        if transposed:
            oH = (
                stride * (iH - 1)
                - 2 * padding
                + dilation * (kH - 1)
                + output_padding
                + 1
            )
            oW = (
                stride * (iW - 1)
                - 2 * padding
                + dilation * (kW - 1)
                + output_padding
                + 1
            )
        else:
            oH = math.floor((iH + 2 * padding - dilation * (kH - 1) - 1) / stride + 1)
            oW = math.floor((iW + 2 * padding - dilation * (kW - 1) - 1) / stride + 1)

        return pd.Series([oH, oW], index=["oH", "oW"])

    def conv_backward_flop(
        self,
        grad_out_shape,
        x_shape,
        w_shape,
        _bias,
        _stride,
        _padding,
        _dilation,
        transposed,
        _output_padding,
        _groups,
        output_mask,
        out_shape,
    ) -> int:

        def t(shape):
            return [shape[1], shape[0]] + list(shape[2:])

        flop_count = 0

        # grad_inp as conv_transpose(grad_out, weight)
        if output_mask[0]:
            grad_input_shape = get_shape(out_shape[0])
            flop_count += conv_flop_count(
                grad_out_shape, w_shape, grad_input_shape, not transposed
            )

        if output_mask[1]:
            grad_weight_shape = get_shape(out_shape[1])
            if transposed:
                # grad_weight of transposed conv as conv(grad_out, inp)
                flop_count += conv_flop_count(
                    t(grad_out_shape),
                    t(x_shape),
                    t(grad_weight_shape),
                    transposed=False,
                )
            else:
                # grad_weight as conv(inp, grad_out)
                flop_count += conv_flop_count(
                    t(x_shape),
                    t(grad_out_shape),
                    t(grad_weight_shape),
                    transposed=False,
                )

        return flop_count

    def compute_flops(self, df: pd.DataFrame) -> pd.DataFrame:
        df[["oH", "oW"]] = df.apply(TimeProcessorConv2d.compute_out_shape, axis=1)

        if self.is_forward:
            flops = df.apply(
                lambda row: conv_flop_count(
                    x_shape=[row["b"], row["in_channels"], row["iH"], row["iW"]],
                    w_shape=[
                        row["out_channels"],
                        row["in_channels"] // row["groups"],
                        row["kH"],
                        row["kW"],
                    ],
                    out_shape=[row["b"], row["out_channels"], row["oH"], row["oW"]],
                    transposed=bool(row["transposed"]),
                ),
                axis=1,
            )
        else:
            flops = df.apply(
                lambda row: self.conv_backward_flop(
                    grad_out_shape=[
                        row["b"],
                        row["out_channels"],
                        row["oH"],
                        row["oW"],
                    ],
                    x_shape=[row["b"], row["in_channels"], row["iH"], row["iW"]],
                    w_shape=[
                        row["out_channels"],
                        row["in_channels"] // row["groups"],
                        row["kH"],
                        row["kW"],
                    ],
                    _bias=row.get("bias", False),
                    _stride=row["stride"],
                    _padding=0,
                    _dilation=row["dilation"],
                    transposed=bool(row["transposed"]),
                    _output_padding=0,
                    _groups=row["groups"],
                    output_mask=[
                        row.get("compute_grad_input", False),
                        row.get("compute_grad_weight", True),
                    ],
                    out_shape=[
                        [row["b"], row["in_channels"], row["oH"], row["oW"]],
                        [
                            row["out_channels"],
                            row["in_channels"] // row["groups"],
                            row["kH"],
                            row["kW"],
                        ],
                    ],
                ),
                axis=1,
            )
        df["gflops"] = flops * 1e-9
        return df

    def get_data(
        self, sample_rate: float = 0.5, ignore_invalid: bool = False
    ) -> pd.DataFrame:
        """
        NOTE: for now, we ignore all csv's with -1 through -4, which denote errors.
        """
        dfs = []

        missing_file_count = 0
        for iH, transposed, group_size, batch_size in tqdm(
            product(self.iH, self.transposed, self.group_sizes, self.batch_sizes)
        ):
            file_name = f"time.{iH}.{transposed}.{group_size}.{batch_size}.parquet"
            file_path = os.path.join(self.data_dir, file_name)
            if not os.path.isfile(file_path):
                missing_file_count += 1
                continue
            df = pd.read_parquet(file_path).sample(frac=sample_rate)
            df.rename(
                columns={
                    "Kernel Name": "kernel_params",
                    "Latency (ms)": "time",
                },
                inplace=True,
            )

            if (df["time"] < 0).sum() > 0:
                print(f"< 0 found in file {file_name}")
                if ignore_invalid:
                    continue
            dfs.append(df.apply(self.split_params, axis=1))

        if missing_file_count > 0:
            print(f"Missing {missing_file_count} files")

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
