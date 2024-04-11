import os, sys
from collections import defaultdict
import numpy as np
import pandas as pd
import torch

# TODO: figure out to create .__init__.py folders
# Once we do that, create a separate utils folder, and split this file into multiple folders.

# Assume warmup reps = nreps = 10.
WARMUP_REPS = 10
NREPS = 30
LINEAR_SIZES = [1, 2] + [i for i in range(4, 128, 4)] + [i for i in range(128, 256, 8)] + [i for i in range(256, 384, 16)] + [i for i in range(384, 512, 32)] + [i for i in range(512, 1024 + 1, 64)]

# TODO: abstract out for convolution.
# Function Cache Configuration: only one unique value: "CachePreferNone"
NON_NUMERIC = {"Params", "Kernel Name", "Block Size", "Grid Size", "Function Cache Configuration"}
CATEGORICAL = {"Precision"}
NO_PROCESS = {"Params", "Inputs", "Precision", "Bias", "Input Size", "Output Size",
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

    columns = sorted(set(df.columns) - NO_PROCESS)
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
    if col in NON_NUMERIC:
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
    columns = sorted(set(df.columns) - NO_PROCESS)
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
        if col in NON_NUMERIC:
            df[col] = df[col].astype(str)
        elif col in CATEGORICAL:
            df[col] = df[col].astype(int).astype(str)
        else:
            df[col] = pd.to_numeric(df[col]) # errors='coerce'
    return df

def combine_csv(pdf: pd.DataFrame, time_file: str):
    tdf = pd.read_csv(time_file)
    tdf.rename(columns={'Kernel Name': 'Params'}, inplace=True)
    return pdf.merge(tdf, how='inner', on='Params')

def scale_csv(save_dfs: bool = False):
    """
    TODO: In order to use the relative pathing correctly, must call this script from this directory.
    """
    base_dir = "../data/linear"
    PROFILE_PATH = f"{base_dir}/profile_data/"
    TIME_PATH = f"{base_dir}/time_data/"

    # TODO: consider using for loops to get tqdm.
    profile_dfs = [scale_data(path=PROFILE_PATH + f"linear.{i}.csv") for i in LINEAR_SIZES]
    combined_dfs = [combine_csv(profile_dfs[i], TIME_PATH + f"linear.time.{time}.csv") \
                    for i, time in enumerate(LINEAR_SIZES)]

    print("Done!")
    # assert(len(profile_dfs) == 1)
    # assert(len(combined_dfs) == 1)

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
    path = f"{base_dir}/combined.csv"
    combined_df.to_csv(path, index=False)

# Function to calculate weighted average for a group
def _weighted_avg(group, avg_columns, weight_column):
    weighted_avgs = {col: np.average(group[col], weights=group[weight_column]) for col in avg_columns}
    weighted_avgs[weight_column] = group[weight_column].sum()  # Sum 'Duration (usecond)'
    weighted_avgs['Kernels Launched'] = group.shape[0]  # This counts the number of rows in the group
    # TODO: could consider adding asserts, since this assumes that there is only one precision value.
    # There should be only one value.
    weighted_avgs['Precision'] = group['Precision'].iloc[0]
    return pd.Series(weighted_avgs)

def merge_kernel_time(df: pd.DataFrame):
    """
    TODO: update comments.
    Note: this returns a df with only the integer values.

    Takes a weighted average based on "Duration (usecond)".

    Joins kernels based on "Params".

    Note: assumes that numerical columns have already been converted.
    """
    # Exclude categorical columns and columns not involved in the weighted average
    numerical_cols = df.select_dtypes(include=['number']).columns.drop(['Duration (usecond)', 'Precision'])

    # Compute weighted averages for each group
    return df.groupby('Params').apply(_weighted_avg, numerical_cols, 'Duration (usecond)').reset_index()


def optimized_merge_kernel_time(df: pd.DataFrame):
    """
    TODO: implement a faster version of merge_kernel_time
    """
    # Assuming 'Precision' and 'Params' are categorical with a limited number of unique values,
    # converting them to categorical if not already can save memory and speed up groupings
    df['Params'] = df['Params'].astype('str')
    
    # Pre-select columns to avoid repeated computation inside the loop
    numerical_cols = [col for col in df.select_dtypes(include=['number']).columns if col not in ['Duration (usecond)', 'Precision']]
    weight_col = 'Duration (usecond)'

    # Pre-calculate the weights sum and count (Kernels Launched) to use later
    df['TotalDuration'] = df.groupby('Params')[weight_col].transform('sum')
    df['KernelsLaunched'] = df.groupby('Params')['Params'].transform('count')

    # Calculate weighted averages outside of the custom function for better performance
    for col in numerical_cols:
        df[f'WeightedAvg_{col}'] = df[col] * df[weight_col] / df['TotalDuration']

    # Now, aggregate the pre-computed weighted averages and other necessary metrics
    agg_funcs = {f'WeightedAvg_{col}': 'sum' for col in numerical_cols}
    agg_funcs.update({
        'TotalDuration': 'first',  # Since it's the same within each group
        'KernelsLaunched': 'first',  # Ditto
        'Precision': 'first'  # Assuming Precision is consistent within groups, as per your assumption
    })

    result_df = df.groupby('Params').agg(agg_funcs).reset_index()

    # Cleanup and rename as needed
    # For example, drop the 'WeightedAvg_' prefix
    result_df.columns = [col.replace('WeightedAvg_', '') if 'WeightedAvg_' in col else col for col in result_df.columns]

    return result_df


def get_fc_flops(row):
    return row['Inputs'] * (2 * row['Input Size'] + 1) * row['Output Size'] / (10**3)

def _time_model(model, *args):
    """
    Time in ms.
    """
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = model(*args)
    end.record()
    torch.cuda.synchronize()
    return result, start.elapsed_time(end)

def time_addmm(A, B, C = None):
    """
    Returns the median runtime in ms.
    C = None if we don't use bias.

    Based on:
    https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html#demonstrating-speedups

    Could consider:
    https://www.speechmatics.com/company/articles-and-news/timing-operations-in-pytorch
    """

    @torch.compile(backend="inductor")
    def addmm(a, b, bias):
        return torch.addmm(bias, a, b)

    @torch.compile(backend="inductor")
    def mm(a, b):
        return torch.mm(a, b)

    if C is not None:
        fn = addmm
        args = (A, B, C)
    else:
        fn = mm
        args = (A, B)

    # Do all of the fusion heuristics, so the later call won't need to.
    for _ in range(WARMUP_REPS):
        _ = fn(*args)

    times = []
    # Actual eval.
    for _ in range(NREPS):
        _, time = _time_model(fn, *args)
        times.append(time)
    return np.median(np.array(times))
