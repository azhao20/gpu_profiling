import os
import pandas as pd
from tqdm import tqdm

def convert_pkl_to_parquet(root_dir):
    """
    Convert CSV files in the given directory to Parquet format.
    If the CSV file is empty, it is deleted.
    """
    for subdir, _, files in os.walk(root_dir):
        for file in tqdm(files, desc=f"Processing directory: {subdir}"):
            if file.endswith('.pkl'):
                csv_path = os.path.join(subdir, file)
                
                if os.path.getsize(csv_path) == 0:
                    print(f"Empty file: {csv_path}")
                    os.remove(csv_path)
                    continue

                try:
                    df = pd.read_pickle(csv_path)
                    
                    parquet_path = os.path.splitext(csv_path)[0] + '.parquet'
                    
                    df.to_parquet(parquet_path, compression='snappy')  # Snappy compression for Parquet
                    
                    if os.path.exists(parquet_path):
                        os.remove(csv_path)
                    else:
                        print(f"Failed to create Parquet file for: {csv_path}")

                except pd.errors.EmptyDataError:
                    print(f"Skipped file with no data: {csv_path}")

def main():
    """
    Convert CSV files to Parquet files in the specified root directory.
    """
    BASE_DIR = "/n/holyscratch01/idreos_lab/Users/azhao/conv2d_data/gpu_profiling/data/final"
    subdirectories = ["conv2d"]

    for subdir in subdirectories:
        path = os.path.join(BASE_DIR, subdir)
        convert_pkl_to_parquet(path)
        
    BASE_DIR = "/Users/andrew/Desktop/Harvard/idreos-research/gpu_profiling/experiments/data"
    subdirectories = ["sdpa", "sdpa_backward", "conv2d", "conv2d_backward", "mm", "bmm"]

    for subdir in subdirectories:
        path = os.path.join(BASE_DIR, subdir)
        convert_pkl_to_parquet(path)


if __name__ == "__main__":
    main()
