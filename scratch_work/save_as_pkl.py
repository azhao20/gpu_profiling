import os
import pandas as pd
from tqdm import tqdm

def convert_csvs_to_pickle(root_dir):
    for subdir, _, files in os.walk(root_dir):
        for file in tqdm(files, desc=f"Processing directory: {subdir}"):
            if file.endswith('.csv'):
                csv_path = os.path.join(subdir, file)
                
                if os.path.getsize(csv_path) == 0:
                    print(f"empty file: {csv_path}")
                    os.remove(csv_path)
                    continue

                try:
                    df = pd.read_csv(csv_path)
                    
                    pickle_path = os.path.splitext(csv_path)[0] + '.pkl'
                    
                    df.to_pickle(pickle_path)
                    
                    if os.path.exists(pickle_path):
                        # print(f"Converted {csv_path} to {pickle_path}")
                        os.remove(csv_path)
                        # print(f"Deleted original CSV file: {csv_path}")
                    else:
                        print(f"Failed to create pickle file for: {csv_path}")

                except pd.errors.EmptyDataError:
                    print(f"Skipped file with no data: {csv_path}")

def main():
    """
    Save csv's as pkl files.
    """
    BASE_DIR = "/n/holylabs/LABS/idreos_lab/Users/azhao/gpu_profiling/data"
    subdirectories = ["final", "linear"]

    for subdir in subdirectories:
        path = os.path.join(BASE_DIR, subdir)
        convert_csvs_to_pickle(path)

if __name__ == "__main__":
    main()
