# gpu_profiling
Andrew's Research on profiling PyTorch kernels


## Time Scripts (The main experiment)
The `python` scripts, located in `scripts/`, are of the form `<operator>.py`. `scripts/<operator>.sh` is a parameterizable bash script that `run_experiment/run_<operator>.sh` spins up.


## Environment
It's necessary to update `PYTHONPATH` to run the scripts described below. One such example is `sh/initconda.sh`.


## Parse Scripts
These are used to generate the range of hyperparameters current models use. I cloned Pytorch from source in another repository (whoops, TODO: fix this), but this relies on my `pytorch/models` branch on my Pytorch fork. To run, call `scripts/parse_sizes.sh`, which invokes `scripts/parse.py`; the outputs are stored in `data/*models`. Note: could update this to point to `data/models/{train, eval, final}` on the Pytorch branch.


## Old Scripts
The old NCU scripts are located in `old_scripts` . This is how I generated times and NSight Compute statistics for linear sizes. The two main scripts for running are `run_time_linear.sh` and `profile_time_linear.sh`, which submits a series of jobs via `time_linear.sh` and `profile_linear.sh`, respectively.


## Utils

### More relevant:
`utils.profile_utils` times non-Inductor-fused kernels.

`utils.prediction_utils` has utils for doing runtime prediction.

`utils.time_utils` combines csvs and creates features for prediction.

### Less relevant:

`utils.process_utils` does NCU output processing.

`utils.utils` contains generally useful classes/functions.

## Jupyter
Some data analysis + prediction that I performed.

Note: the only incomplete predictor is `conv2d`.
