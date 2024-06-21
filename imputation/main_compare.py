import argparse
import os
import sys
from typing import Any
import numpy as np
import pandas as pd
from signal import SIGINT, signal
import tensorflow.compat.v1 as tf


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__ ),'compare_methods')))

# methods absolute
from compare_methods import load_imputer


tf.disable_v2_behavior()
np.set_printoptions(suppress=True)


def handler(signal_received: Any, frame: Any) -> None:
    print("SIGINT or CTRL-C detected. Exiting gracefully")
    exit(0)

def binary_sampler(p: float, rows: int, cols: int) -> np.ndarray:
    """Sample binary random variables.

    Args:
      - p: probability of 1
      - rows: the number of rows
      - cols: the number of columns

    Returns:
      - binary_random_matrix: generated binary random matrix.
    """
    unif_random_matrix = np.random.uniform(0.0, 1.0, size=[rows, cols])
    binary_random_matrix = 1 * (unif_random_matrix < p)
    return binary_random_matrix

def rmse_loss(ori_data: np.ndarray, imputed_data: np.ndarray, data_m: np.ndarray) -> np.ndarray:
    numerator = np.sum(((1 - data_m) * ori_data - (1 - data_m) * imputed_data) ** 2)
    denominator = np.sum(1 - data_m)
    return np.sqrt(numerator / float(denominator))


import pandas as pd
import numpy as np

# Function to calculate MAPE
def calculate_mape(df1, df2):
    return (np.abs((df1 - df2) / df1).mean()) * 100



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--gpu", type=str, default="")
    parser.add_argument("--missingness", type=float, default=0.2)
    args = parser.parse_args()
    signal(SIGINT, handler)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    data_missing = pd.read_excel("data_missing_one_nan.xlsx")
    data_missing = data_missing[['Input_9.1', 'Input_9.2', 'Input_9.3']]

    data = pd.read_excel("data_train_BNN.xlsx")
    data = data[['Input_9_1','Input_9_2','Input_9_3']]
    df = data.sample(frac=1).reset_index(drop=True)
    print(df.head())
    #
    # scaler = StandardScaler()
    # df = scaler.fit_transform(df)

    X_MISSING = df.copy()  # This will have missing values
    X_TRUTH = df.copy()  # This will have no missing values for testing

    # Generate MCAR
    X_MASK = binary_sampler(1 - args.missingness, X_MISSING.shape[0], X_MISSING.shape[1])
    X_MISSING[X_MASK == 0] = np.nan

    datasize = X_MISSING.shape[0]
    missingness = args.missingness
    feature_dims = X_MISSING.shape[1]

    # Append indicator variables - One indicator per feature with missing values.
    missing_idxs = np.where(np.any(np.isnan(X_MISSING), axis=0))[0]

    seed_imputer_missforest = load_imputer("missforest")
    X_seed_imputer_missforest = seed_imputer_missforest.fit_transform(X_MISSING)

    seed_imputer_mean = load_imputer("mean")
    X_seed_imputer_mean = seed_imputer_mean.fit_transform(X_MISSING)

    seed_imputer_mice = load_imputer("mice")
    X_seed_imputer_mice = seed_imputer_mice.fit_transform(X_MISSING)

    seed_imputer_gain = load_imputer("gain")
    X_seed_imputer_gain = seed_imputer_gain.fit_transform(X_MISSING)

    seed_imputer_knn = load_imputer("knn")
    X_seed_imputer_knn = seed_imputer_knn.fit_transform(X_MISSING)

    from debug_mergecoce import imputationTDM
    X_seed_imputer_TDM = imputationTDM(X_MISSING, X_TRUTH)

    print("Mean Absolute Percentage Error")
    print(f"MAPE of MEAN: {calculate_mape(X_TRUTH, X_seed_imputer_mean)}")
    print(f"MAPE of KNN: {calculate_mape(X_TRUTH, X_seed_imputer_knn)}")
    print(f"MAPE of MICE: {calculate_mape(X_TRUTH, X_seed_imputer_mice)}")
    print(f"MAPE of GAIN: {calculate_mape(X_TRUTH, X_seed_imputer_gain)}")
    print(f"MAPE of MISSFOREST: {calculate_mape(X_TRUTH, X_seed_imputer_missforest)}")
    print(f"MAPE of TDM:  {calculate_mape(X_TRUTH, X_seed_imputer_TDM.numpy())}")

