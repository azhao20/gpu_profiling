import os, sys

# Assume warmup reps = nreps = 10.
LINEAR_SIZES = [1, 2] + [i for i in range(4, 128, 4)] + [i for i in range(128, 256, 8)] + [i for i in range(256, 384, 16)] + [i for i in range(384, 512, 32)] + [i for i in range(512, 1024 + 1, 64)]

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
