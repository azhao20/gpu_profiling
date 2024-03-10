import os, sys
from collections import defaultdict
import numpy as np
import pandas as pd
import torch

# TODO: figure out to create .__init__.py folders
# Once we do that, create a separate utils folder, and split this file into multiple folders.

# Assume warmup reps = nreps = 10.
NREPS = 10
LINEAR_SIZES = [1, 2] + [i for i in range(4, 128, 4)] + [i for i in range(128, 256, 8)] + [i for i in range(256, 384, 16)] + [i for i in range(384, 512, 32)] + [i for i in range(512, 1024 + 1, 64)]

# TODO: abstract out for convolution.
# Function Cache Configuration: only one unique value: "CachePreferNone"
non_numeric = {"Params", "Kernel Name", "Block Size", "Grid Size", "Function Cache Configuration"}
no_process = {"Params", "Inputs", "Precision", "Bias", "Input Size", "Output Size",
              "Context", "Device", "Stream", "CC", "Kernel Name", "Block Size", "Grid Size"}

class HiddenPrints:
    """
    A class that suppresses print statements. Use inside of a context manager.
    """
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def get_precision(precision_flag: int):
    precision_map = {
        # 8: torch.float8_e5m2, # No support for float8 yet.
        161: torch.bfloat16,
        162: torch.float16,
        32: torch.float32
    }
    if precision_flag not in precision_map:
        print("Precision wasn't specified, defaulting to torch.float32")

    return precision_map.get(precision_flag, torch.float32)

def _time_iter(model, input):
    """
    Time in ms.
    """
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = model(input)
    end.record()
    torch.cuda.synchronize()
    return result, start.elapsed_time(end)

def time_model(model, warmup_input, input):
    """
    Returns the median runtime in ms.
    
    Based on:
    https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html#demonstrating-speedups

    Could consider:
    https://www.speechmatics.com/company/articles-and-news/timing-operations-in-pytorch
    """
    times = []
    # Warmup
    # Do all of the fusion heuristics, so the later call won't need to.
    for _ in range(NREPS):
        _ = model(warmup_input)

    # Actual eval.
    for _ in range(NREPS):
        _, time = _time_iter(model, input)
        times.append(time)
    return np.median(np.array(times))


def combine_profile_csv(layer: str = "linear", sizes: list[int] = LINEAR_SIZES):
    """
    In order to use the relative pathing correctly, must call this script from this directory.
    Join a bunch of NCU-returned datasets. Since there are so many data points and the 
    NA values are rare, we just drop them here for ease.

    TODO: separate process utils.
    TODO: use logs instead of prints.
    """
    data_dir = f"../data/{layer}"
    data_path = f"{data_dir}/profile_data/"
    final_csv = f"{data_dir}/profile.csv"
    dfs = [pd.read_csv(data_path + f"{layer}.{i}.csv") for i in sizes]

    combined_df = pd.concat(dfs, ignore_index=True)
    
    to_drop = combined_df.isna().any(axis=1).sum()
    print(f"Dropping {to_drop} rows")
    combined_df.dropna(axis=0, inplace=True)

    combined_df.to_csv(final_csv, index=False)

def get_units(val: str):
    """
    val: str
        The string of the form "<val>[[<unit or None>]]"

    Used to get the units and create a dictionary mapping by manual inspection.
    Returns None if there are no units.
    """
    parts = val.split('[[')
    assert(len(parts) == 2)
    unit = parts[1][:-2]
    return unit if unit else None

def get_unique_mapping():
    """
    TODO: abstract this stuff out into linear, conv2d, etc.
    This should really be a helper function.

    Linear:
        DRAM Frequency: ['cycle/nsecond' 'cycle/usecond']
        Dynamic Shared Memory Per Block: ['byte/block' 'Kbyte/block']
        Memory Throughput: ['byte/second' 'Mbyte/second' 'Gbyte/second']
        SM Frequency: ['cycle/nsecond' 'cycle/usecond']
        Static Shared Memory Per Block: ['byte/block' 'Kbyte/block']

    """
    path = f"../data/linear/profile.csv"
    df = pd.read_csv(path)

    columns = sorted(set(df.columns) - no_process)
    res = defaultdict(set)
    for col in columns:
        res[col] = df[col].apply(get_units).unique()
    return res

def get_scaled_value(col: str, d, val_unit: str):
    """
    Extract units and scale value.
    For large units, rounds to avoid FLOP precision errors, since
    assumes ncu precision is two decimal places.
    """
    parts = val_unit.split('[[')
    assert(len(parts) == 2)
    unit = parts[1][:-2]
    val = parts[0].replace(',', '')
    if col in non_numeric:
        return val
    else:
        res = float(val) * d[unit]
        return round(res) if d[unit] >= 10e3 else res

def scale_data(path: str):
    print(path)
    df = pd.read_csv(path)
    # For now, drop nan values. 
    to_drop = df.isna().any(axis=1).sum()
    if to_drop > 0:
        print(f"Dropping {to_drop} rows")
        df.dropna(axis=0, inplace=True)

    # TODO: find a better way to do this.
    # This scales values for columns with different units.
    # All other columns should only have one unit.
    unit_scale = defaultdict(lambda: defaultdict(lambda: 1))
    unit_scale['DRAM Frequency']['cycle/nsecond'] = 1e3
    unit_scale['DRAM Frequency']['cycle/usecond'] = 1
    unit_scale['SM Frequency']['cycle/nsecond'] = 1e3
    unit_scale['SM Frequency']['cycle/usecond'] = 1

    unit_scale['Dynamic Shared Memory Per Block']['byte/block'] = 1
    unit_scale['Dynamic Shared Memory Per Block']['Kbyte/block'] = 1e3
    unit_scale['Static Shared Memory Per Block']['byte/block'] = 1
    unit_scale['Static Shared Memory Per Block']['Kbyte/block'] = 1e3
    # Kbyte
    unit_scale['Memory Throughput']['byte/second'] = 1e-3
    unit_scale['Memory Throughput']['Mbyte/second'] = 1e3
    unit_scale['Memory Throughput']['Gbyte/second'] = 1e6

    # unit_scale = defaultdict(lambda: 1)
    # unit_scale['cycle/nsecond'] = 1e3
    # unit_scale['cycle/usecond'] = 1
    # unit_scale['byte/block'] = 1
    # unit_scale['Kbyte/block'] = 1e3
    # # Kbyte
    # unit_scale['byte/second'] = 1e-3
    # unit_scale['Mbyte/second'] = 1e3
    # unit_scale['Gbyte/second'] = 1e6

    # If units are too large, choose different values as default.
    unit_map = {
        'DRAM Frequency' : 'cycle/usecond',
        'SM Frequency' : 'cycle/usecond',
        'Dynamic Shared Memory Per Block' : 'byte/block',
        'Static Shared Memory Per Block' : 'byte/block',
        'Memory Throughput' : 'Kbyte/second'
    }

    assert(df.shape[0] > 0)
    columns = sorted(set(df.columns) - no_process)
    # columns = ["Function Cache Configuration"]
    to_drop = []
    for col in columns:
        # if col not in unit_scale:
        # units should be the same and may be None.
        units = unit_map[col] if col in unit_scale else get_units(df[col][0])

        new_col = col
        if units:
            new_col += " (" + units + ")"
            to_drop.append(col)

        df[new_col] = df[col].apply(lambda x: get_scaled_value(col, unit_scale[col], x))

    # Drop the old columns.
    df.drop(columns=to_drop, inplace=True)

    # Safer to be explicit about conversion.
    for col in df.columns:
        df[col] = df[col].astype(str) if col in non_numeric else pd.to_numeric(df[col]) # errors='coerce'

    return df

def combine_csv(pdf: pd.DataFrame, time_file: str):
    tdf = pd.read_csv(time_file)
    return pdf.merge(tdf, how='inner', left_on='Params', right_on='Kernel Name')

def scale_csv(save_dfs: bool = False):
    """
    TODO: In order to use the relative pathing correctly, must call this script from this directory.
    """
    base_dir = "../data/linear"
    PROFILE_PATH = f"{base_dir}/profile_data/"
    TIME_PATH = f"{base_dir}/time_data/"

    # TODO: consider using for loops to get tqdm.
    profile_dfs = [scale_data(path=PROFILE_PATH + f"linear.{i}.csv") for i in [1]]
    combined_dfs = [combine_csv(profile_dfs[i], TIME_PATH + f"linear.time.{time}.csv") \
                    for i, time in enumerate([1])]

    print("Done!")
    assert(len(profile_dfs) == 1)
    assert(len(combined_dfs) == 1)

    if save_dfs:
        print("Not implemented yet!")
        # try:
        #     os.mkdir(os.path.join(base_dir, "combined_data"))
        # except:
        #     # TODO: make this a log statement.
        #     print("Directory already exists")
        # for df in combined_dfs:
        #     df.to_csv()

    combined_df = pd.concat(combined_dfs, ignore_index=True)
    print(combined_df.shape)
    path = f"{base_dir}/hello.csv"
    combined_df.to_csv(path, index=False)
