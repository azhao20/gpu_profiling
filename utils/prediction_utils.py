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


def get_data(
    kernel: str, base_dir: str, sample_rate: float = 1.0
) -> list[pd.DataFrame, pd.Series]:
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
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=random_seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=random_seed
    )

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


def plot_actual_vs_pred(y_val, y_pred, figsize=(10, 6)):
    plt.figure(figsize=figsize)
    sns.regplot(x=y_val, y=y_pred)

    # Plot y = x
    max_val = max(max(y_val), max(y_pred))
    min_val = min(min(y_val), min(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], "r--", label="y = x (Ideal)")

    plt.title("Actual vs. Predicted")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted")
    plt.show()


def binned_scatterplot(
    df, x_col, y_col, bins=10, statistic="mean", figsize=(10, 6), with_error_bars=True
):
    df["x_binned"] = pd.cut(df[x_col], bins=bins)

    # Group by the bins and calculate the summary statistic for y values
    if statistic == "mean":
        binned_stats = (
            df.groupby("x_binned", observed=True)[y_col]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
    elif statistic == "median":
        binned_stats = (
            df.groupby("x_binned", observed=True)[y_col]
            .agg(["median", "std", "count"])
            .reset_index()
        )
        binned_stats.rename(
            columns={"median": "mean"}, inplace=True
        )  # Renaming for consistency
    else:
        raise ValueError("Statistic must be either 'mean' or 'median'")

    binned_stats["x_mid"] = binned_stats["x_binned"].apply(lambda x: x.mid)

    # Calculate the standard error
    if with_error_bars:
        binned_stats["se"] = binned_stats["std"] / np.sqrt(binned_stats["count"])

    # Create the scatter plot using Seaborn
    plt.figure(figsize=figsize)
    plt.errorbar(
        x=binned_stats["x_mid"],
        y=binned_stats["mean"],
        yerr=binned_stats["se"] if with_error_bars else None,
        capsize=5,
        capthick=1,
        linestyle="None",
    )
    sns.scatterplot(x=binned_stats["x_mid"], y=binned_stats["mean"], color="b")

    plt.title(f"Binned Scatterplot: {x_col} vs. {y_col}")
    plt.xlabel(f"Binned {x_col}")
    plt.ylabel(f"{statistic.capitalize()} of {y_col}")

    plt.tight_layout()
    plt.show()

    df.drop(columns=["x_binned"], inplace=True)


def binned_scatterplot_cols(
    x_col,
    y_col,
    x_label: str,
    y_label: str,
    bins=10,
    statistic="mean",
    figsize=(10, 6),
    with_error_bars=True,
):
    x_binned = pd.cut(x_col, bins=bins)

    temp_df = pd.DataFrame({"x_binned": x_binned, "y_col": y_col})

    if statistic == "mean":
        binned_stats = (
            temp_df.groupby("x_binned", observed=True)["y_col"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
    elif statistic == "median":
        binned_stats = (
            temp_df.groupby("x_binned", observed=True)["y_col"]
            .agg(["median", "std", "count"])
            .reset_index()
        )
        binned_stats.rename(
            columns={"median": "mean"}, inplace=True
        )  # Renaming for consistency
    else:
        raise ValueError("Statistic must be either 'mean' or 'median'")

    binned_stats["x_mid"] = binned_stats["x_binned"].apply(lambda x: x.mid)

    if with_error_bars:
        binned_stats["se"] = binned_stats["std"] / np.sqrt(binned_stats["count"])

    plt.figure(figsize=figsize)
    plt.errorbar(
        x=binned_stats["x_mid"],
        y=binned_stats["mean"],
        yerr=binned_stats["se"] if with_error_bars else None,
        capsize=5,
        capthick=1,
        label="Mean ± SE",
        linestyle="None",
    )
    sns.scatterplot(x=binned_stats["x_mid"], y=binned_stats["mean"], color="b")

    plt.title(f"{x_label} vs. {y_label}")
    plt.xlabel(f"Binned {x_label}")
    plt.ylabel(f"{statistic.capitalize()} of {y_label}")

    plt.tight_layout()
    plt.show()


def plot_residuals(y_val, y_pred, bins=0, figsize=(10, 6)):
    residuals = y_val - y_pred

    if bins > 0:
        binned_scatterplot_cols(y_pred, residuals, "Predicted", "Residuals", bins)
    else:
        plt.figure(figsize=figsize)
        sns.scatterplot(x=y_pred, y=residuals)
        plt.title("Residual Plot")
        plt.xlabel("Predicted Values")
        plt.ylabel("Residuals")
        plt.axhline(y=0, color="r", linestyle="--")
        plt.show()


def run_val_pipeline(model, X_train, X_val, y_train, y_val, bins=100):
    """
    
    """
    y_hat_train = model.predict(X_train)
    y_hat_val = model.predict(X_val)

    print("Train--------")
    get_r2_score(y_train, y_hat_train)
    plot_residuals(y_train, y_hat_train)
    plot_residuals(y_train, y_hat_train, bins=bins)
    plot_actual_vs_pred(y_train, y_hat_train)

    print("Val--------")
    get_r2_score(y_val, y_hat_val)
    plot_residuals(y_val, y_hat_val)
    plot_residuals(y_val, y_hat_val, bins=bins)
    plot_actual_vs_pred(y_val, y_hat_val)
