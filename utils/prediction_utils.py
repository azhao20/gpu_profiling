import pandas as pd
import os, sys
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
import statsmodels.api as sm

from utils.time_utils import *

random_seed = 42

def get_data(kernel: str,
             base_dir: str,
             sample_rate: float = 1.0) -> list[pd.DataFrame, pd.Series]:
    if kernel == "mm":
        time_processor = TimeProcessorMM(base_dir)
    elif kernel == "bmm":
        time_processor = TimeProcessorBMM(base_dir)
    elif kernel == "sdpa":
        time_processor = TimeProcessorSDPA(base_dir, is_forward=True)
    elif kernel == "sdpa_backward":
        time_processor = TimeProcessorSDPA(base_dir, is_forward=False)
    elif kernel == "conv2d":
        time_processor = TimeProcessorConv2d(base_dir, is_forward=True)
    elif kernel == "conv2d_backward":
        time_processor = TimeProcessorConv2d(base_dir, is_forward=False)

    df = time_processor.get_data(sample_rate=sample_rate)
    
    X, y = df.drop(["time", "kernel_params"], axis=1), df["time"]
    return pd.get_dummies(X, drop_first=False), y

def get_train_test_split(X: pd.DataFrame, y: pd.DataFrame, return_concat: bool = True):
    """
    return_concat: useful for doing feature engineering
    """
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=random_seed)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=random_seed)

    df_train = pd.concat([X_train, y_train], axis=1)
    df_val = pd.concat([X_val, y_val], axis=1)
    df_test = pd.concat([X_test, y_test], axis=1)
    if return_concat:
        return df_train, df_val, df_test
    return X_train, X_val, X_test, y_train, y_val, y_test

def get_r2_score(y, y_hat):
    mse = mean_squared_error(y, y_hat)
    mape = mean_absolute_percentage_error(y, y_hat)
    r2 = r2_score(y, y_hat)
    print(f"Mean Squared Error: {mse}")
    print(f"MAPE: {mape}")
    print(f"R-squared: {r2}")
    return mse, r2
    
def plot_residuals(y_val, y_pred, figsize=(10, 6)):
    residuals = y_val - y_pred
    
    plt.figure(figsize=figsize)
    plt.scatter(y_pred, residuals)
    plt.title('Residual Plot')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.show()