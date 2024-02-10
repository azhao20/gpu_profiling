import os, sys
import torch

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
