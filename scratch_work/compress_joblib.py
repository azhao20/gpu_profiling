import os
import joblib
from tqdm import tqdm

def compress_joblib_files(root_dir, compress_level=3):
    for subdir, _, files in os.walk(root_dir):
        for file in tqdm(files, desc=f"Processing directory: {subdir}"):
            if file.endswith('.joblib'):
                joblib_path = os.path.join(subdir, file)

                if os.path.getsize(joblib_path) == 0:
                    print(f"Empty file: {joblib_path}")
                    os.remove(joblib_path)
                    continue

                try:
                    model = joblib.load(joblib_path)
                    joblib.dump(model, joblib_path, compress=compress_level)

                except Exception as e:
                    print(f"Error processing {joblib_path}: {e.__class__.__name__}: {str(e)}")

def main():
    """
    Compress all joblib files in the specified subdirectories.
    """
    BASE_DIR = "/n/holylabs/LABS/idreos_lab/Users/azhao/gpu_profiling"
    subdirectories = ["a100_models", "h100_models"]
    compress_level = 3  # Adjust the compression level (0-9)

    for subdir in subdirectories:
        path = os.path.join(BASE_DIR, subdir)
        compress_joblib_files(path, compress_level=compress_level)

if __name__ == "__main__":
    main()
