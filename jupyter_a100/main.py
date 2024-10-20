import pandas as pd
import os, sys
import numpy as np
import argparse

from utils.prediction_utils import *

import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer, mean_squared_error

random_seed = 42

def get_args():
    parser = argparse.ArgumentParser(description="Random Forest Experiment")
    parser.add_argument(
        "--op_type",
        type=str,
        required=True,
        help="Specify the operation type (e.g., sdpa, sdpa_backward, etc.)"
    )
    parser.add_argument(
        "--device",
        type=str,
        required=True,
        choices=["a100", "h100"],
        help="Device (a100 or h100)"
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=20,
        help="Number of iterations for RandomizedSearchCV"
    )
    return parser.parse_args()

def main():
    args = get_args()
    print(args)
    op_type = args.op_type
    if args.device == "a100":
        base_dir = "/n/holylabs/LABS/idreos_lab/Users/azhao/gpu_profiling/data/final/"
    else:
        base_dir = "/n/holylabs/LABS/idreos_lab/Users/azhao/gpu_profiling/experiments/data/"

    X, y = get_data(op_type, base_dir, sample_rate=1.0)
    X.info()

    df = pd.concat([X, y], axis=1)
    df = df.dropna()
    df = df[df["time"] > 0]
    X, y = df.drop(columns=["time"]), df["time"]

    n_iter = args.n_iter
    X_train, X_val, X_test, y_train, y_val, y_test = get_train_test_split(X, y, return_concat=False)
    X_cv = pd.concat([X_train, X_val])
    y_cv = pd.concat([y_train, y_val])

    tree_model = RandomForestRegressor(random_state=random_seed)

    param_dist = {
        "max_depth": [10, 50, 80, 100, 150, 200],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [2, 5, 10],
        "max_features": [10, 20, 50, 100, 1.0, "sqrt"],
        "n_estimators": [10, 50, 100],
    }

    randomized_search = RandomizedSearchCV(
        estimator=tree_model,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring=make_scorer(mean_squared_error, greater_is_better=False),
        cv=5,
        random_state=random_seed,
        n_jobs=-1,
    )

    randomized_search.fit(X_cv, y_cv)
    print("Best parameters found: ", randomized_search.best_params_)
    print("Best MSE found: ", randomized_search.best_score_)

    best_tree_model = randomized_search.best_estimator_
    best_tree_model.fit(X_cv, y_cv)
    run_val_pipeline(best_tree_model, X_cv, X_test, y_cv, y_test, bins=30)

    joblib.dump(best_tree_model, f'/n/holylabs/LABS/idreos_lab/Users/azhao/gpu_profiling/{args.device}_models/{op_type}.joblib')

if __name__ == "__main__":
    main()
