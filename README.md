# gpu_profiling
Andrew's Research on profiling PyTorch kernels

## Environment
It's necessary to update `PYTHONPATH` to run the scripts described below. One such example is `sh/initconda.sh`.

## Parse Scripts
These are used to generate the hyperparameters. I cloned Pytorch from source in another repository (whoops, TODO: fix this), but this relies on my `pytorch/models` branch. To run, we simply call `scripts/parse_sizes.sh`; the outputs are stored in `data/*models`.

## Time Scripts
The `python` scripts are located in `scripts/`, and include `mm.py`, `bmm.py`, and `sdpa.py`. An example of how to run these scripts is `time_mm.sh`. (Remember to run `source sh/initconda.sh` beforehand or something similar to initialize the environment).

The helper functions used for timing are located in `utils/profile_utils.py`.

I will add `run_time_<operator>.sh` once we confirm the sizes.

## Old Scripts
The old scripts are located in `old_scripts`. This is how I generated times and NSight Compute statistics for linear sizes. The two main scripts for running are `run_time_linear.sh` and `profile_time_linear.sh`, which submits a series of jobs via `time_linear.sh` and `profile_linear.sh`, respectively.
