from abc import abstractmethod
import pandas as pd
import os

from torch.utils.flop_counter import sdpa_flop_count, conv_flop_count

class TimeProcessorBase:
    kernel_type: str = "UNINITIALIZED"

    def __init__(self, base_dir):
        """base_dir points to the repository root."""
        self.base_dir = base_dir

    @property
    def data_dir(self):
        return os.path.join(self.base_dir, "data/final", self.kernel_type)

    @abstractmethod
    def split_params(self, row):
        """Time dataframe doesn't have kernel parameters separated out."""
        pass

    @abstractmethod
    def get_data(self):
        pass

        
class TimeProcessorMM(TimeProcessorBase):
    def __init__(self, base_dir):
        super().__init__(base_dir)
        self.kernel_type = "mm"
        self.n_values = (
            list(range(16, 496, 16)) +
            list(range(512, 1920, 128)) +
            list(range(2048, 3584, 512)) +
            list(range(4096, 32768, 1024))
        )

    def split_params(self, row):
        params = row['kernel_params'].split('.')
        return pd.Series([row['kernel_params'], *params, row['time']], 
                         index=['kernel_params', 'dtype', 'n', 'm', 'p', 'time'])

    def get_data(self, sample_rate: float = 0.5):
        """
        sample_rate:
            Since the resulting dataframe might be large, we sample `sample_rate` from each strata
        """
        dfs = []
        for n in self.n_values:
            file_path = os.path.join(self.data_dir, f"time.{n}.csv")
            df = pd.read_csv(file_path, header=0).sample(frac=sample_rate)
            # df = pd.read_csv(file_path, columns=['Kernel Name', 'Latency (ms)'])
            df.rename(columns={'Kernel Name': 'kernel_params', 'Latency (ms)' : 'time'}, inplace=True)
            dfs.append(df.apply(self.split_params, axis=1))

        dfs = pd.concat(dfs, axis=0, ignore_index=True)
        dfs['dtype'] = dfs['dtype'].astype('category')
        dfs['n'] = dfs['n'].astype(int)
        dfs['m'] = dfs['m'].astype(int)
        dfs['p'] = dfs['p'].astype(int)
        dfs['time'] = dfs['time'].astype(float)

        # Some other feature engineering.
        dfs['gflops'] = 2 * dfs['n'] * dfs['m'] * dfs['p'] / (10**9)
        return dfs

class TimeProcessorBMM(TimeProcessorBase):
    """
    Just add a batch value.
    """
    def __init__(self, base_dir):
        super().__init__(base_dir)
        self.kernel_type = "bmm"
        self.n_values = (
            list(range(16, 496, 16)) +
            list(range(512, 1920, 128)) +
            list(range(2048, 3584, 512)) +
            list(range(4096, 32768, 1024))
        )

    def split_params(self, row):
        params = row['kernel_params'].split('.')
        return pd.Series([row['kernel_params'], *params, row['time']], 
                         index=['kernel_params', 'dtype', 'b', 'n', 'm', 'p', 'time'])

    def get_data(self, sample_rate: float = 0.5):
        """
        sample_rate:
            Since the resulting dataframe might be large, we sample `sample_rate` from each strata
        """
        dfs = []
        for n in self.n_values:
            file_path = os.path.join(self.data_dir, f"time.{n}.csv")
            df = pd.read_csv(file_path, header=0).sample(frac=sample_rate)
            # df = pd.read_csv(file_path, columns=['Kernel Name', 'Latency (ms)'])
            df.rename(columns={'Kernel Name': 'kernel_params', 'Latency (ms)' : 'time'}, inplace=True)
            dfs.append(df.apply(self.split_params, axis=1))

        dfs = pd.concat(dfs, axis=0, ignore_index=True)
        dfs['dtype'] = dfs['dtype'].astype('category')
        dfs['b'] = dfs['b'].astype(int)
        dfs['n'] = dfs['n'].astype(int)
        dfs['m'] = dfs['m'].astype(int)
        dfs['p'] = dfs['p'].astype(int)
        dfs['time'] = dfs['time'].astype(float)

        # Some other feature engineering.
        dfs['gflops'] = 2 * dfs['b'] * dfs['n'] * dfs['m'] * dfs['p'] / (10**9)
        return dfs

class TimeProcessorSDPA(TimeProcessorBase):
    def __init__(self, base_dir):
        super().__init__(base_dir)
        self.kernel_type = "sdpa"
        self.num_heads = [4, 8, 12, 16]
        
    def split_params(self, row):
        params = row['kernel_params'].split('.')
        return pd.Series(
            [row['kernel_params'], *params, row['time']], 
            index=['kernel_params', 'dtype', 'backend', 'b', 'h', 's_q', 's_kv', 'd_qk', 'd_v', 'is_causal', 'time']
        )
        
    def compute_flops(self, df):
        flops = df.apply(
            lambda row: sdpa_flop_count(
                (row['b'], row['h'], row['s_q'], row['d_qk']),
                (row['b'], row['h'], row['s_kv'], row['d_qk']),
                (row['b'], row['h'], row['s_kv'], row['d_v'])
            )[0],
            axis=1
        )
        df['gflops'] = flops / (10 ** 9)
        return df


    def get_data(self, sample_rate: float = 0.5):
        """
        sample_rate:
            Since the resulting dataframe might be large, we sample `sample_rate` from each strata
        """
        dfs = []

        dtypes = ["b16", "16"]
        backends = ["flash", "efficient"]
        
        # TODO: maybe itertool combos?
        for dtype in dtypes:
            for backend in backends:
                for h in self.num_heads:
                    file_path = os.path.join(self.data_dir, f"time.{dtype}.{backend}.{h}.csv")
                    df = pd.read_csv(file_path, header=0).sample(frac=sample_rate)
                    # df = pd.read_csv(file_path, columns=['Kernel Name', 'Latency (ms)'])
                    df.rename(columns={'Kernel Name': 'kernel_params', 'Latency (ms)' : 'time'}, inplace=True)
                    dfs.append(df.apply(self.split_params, axis=1))
        
        dtype = "32"
        backend = "efficient"
        for h in self.num_heads:
            file_path = os.path.join(self.data_dir, f"time.{dtype}.{backend}.{h}.csv")
            df = pd.read_csv(file_path, header=0).sample(frac=sample_rate)
            # df = pd.read_csv(file_path, columns=['Kernel Name', 'Latency (ms)'])
            df.rename(columns={'Kernel Name': 'kernel_params', 'Latency (ms)' : 'time'}, inplace=True)
            dfs.append(df.apply(self.split_params, axis=1))

        dfs = pd.concat(dfs, axis=0, ignore_index=True)
        dfs['dtype'] = dfs['dtype'].astype('category')
        dfs[['b', 'h', 's_q', 's_kv', 'd_qk', 'd_v']] = dfs[['b', 'h', 's_q', 's_kv', 'd_qk', 'd_v']].astype(int)
        dfs['time'] = dfs['time'].astype(float)

        return self.compute_flops(dfs)

class TimeProcessorConv2d(TimeProcessorBase):
    def __init__(self, base_dir):
        super().__init__(base_dir)
        self.kernel_type = "conv2d"
        self.iH = [2, 8, 32, 128, 512, 1024]
        self.transposed = [0, 1]
        
    def split_params(self, row):
        params = row['kernel_params'].split('.')
        return pd.Series(
            [row['kernel_params'], *params, row['time']], 
            index=['kernel_params', 'dtype', 'b', 'in_channels', 'iH', 'iW', 'out_channels', 'groups', 'kH', 'kW', 'stride', 'dilation', 'transposed', 'time']
        )
        
    def compute_flops(self, df):
        flops = df.apply(
            lambda row: conv_flop_count(
                [row['b'], row['in_channels'], row['iH'], row['iW']],
                [row['out_channels'], row['in_channels'], row['kH'], row['kW']],
                [row['b'], row['out_channels'], row['iH'], row['iW']],
                transposed=bool(row['transposed'])
            )[0],
            axis=1
        )
        df['gflops'] = flops / (10 ** 9)
        return df

    def get_data(self, sample_rate: float = 0.5):
        """
        sample_rate:
            Since the resulting dataframe might be large, we sample `sample_rate` from each strata
        """
        dfs = []
        
        for iH in self.iH:
            for iW in self.iW:
                for transposed in self.transposed:
                    file_path = os.path.join(self.data_dir, f"time.{iH}.{iW}.{transposed}.csv")
                    df = pd.read_csv(file_path, header=0).sample(frac=sample_rate)
                    df.rename(columns={'Kernel Name': 'kernel_params', 'Latency (ms)' : 'time'}, inplace=True)
                    dfs.append(df.apply(self.split_params, axis=1))

        dfs = pd.concat(dfs, axis=0, ignore_index=True)
        dfs['dtype'] = dfs['dtype'].astype('category')
        dfs[['b', 'in_channels', 'iH', 'iW', 'out_channels', 'groups', 'kH', 'kW', 'stride', 'dilation']] \
            = dfs[['b', 'in_channels', 'iH', 'iW', 'out_channels', 'groups', 'kH', 'kW', 'stride', 'dilation']].astype(int)
        dfs['time'] = dfs['time'].astype(float)

        return self.compute_flops(dfs)
