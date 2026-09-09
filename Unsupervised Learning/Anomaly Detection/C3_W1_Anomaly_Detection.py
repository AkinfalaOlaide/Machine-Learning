"""Anomaly detection with a Gaussian model.

Fits a per-feature Gaussian to a training set of server measurements,
picks the probability threshold epsilon that maximises F1 on a labelled
validation set, and flags training examples below that threshold.

Run with:
    python3 C3_W1_Anomaly_Detection.py
"""

import numpy as np
import matplotlib.pyplot as plt

from utils import load_data, load_data_multi, multivariate_gaussian, visualize_fit
from public_tests import estimate_gaussian_test, select_threshold_test


def estimate_gaussian(X):
    """Return the mean and variance of every feature (column) in X.

    Args:
        X (ndarray): (m, n) data matrix

    Returns:
        mu  (ndarray): (n,) mean of each feature
        var (ndarray): (n,) variance of each feature
    """
    m, n = X.shape
    mu = np.sum(X, axis=0) / m
    var = np.sum((X - mu) ** 2, axis=0) / m
    return mu, var


def select_threshold(y_val, p_val):
    """Sweep candidate thresholds and return the one with the best F1.

    An example is predicted anomalous when p(x) < epsilon.

    Args:
        y_val (ndarray): ground-truth labels (1 = anomaly, 0 = normal)
        p_val (ndarray): Gaussian density of each validation example

    Returns:
        best_epsilon (float): chosen threshold
        best_F1      (float): F1 score at that threshold
    """
    best_epsilon = 0
    best_F1 = 0

    step_size = (max(p_val) - min(p_val)) / 1000

    for epsilon in np.arange(min(p_val), max(p_val), step_size):
        predictions = p_val < epsilon

        tp = np.sum((predictions == 1) & (y_val == 1))
        fp = np.sum((predictions == 1) & (y_val == 0))
        fn = np.sum((predictions == 0) & (y_val == 1))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        F1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        if F1 > best_F1:
            best_F1 = F1
            best_epsilon = epsilon

    return best_epsilon, best_F1


def run_2d_dataset():
    """Part 1: two features (latency, throughput), with plots."""
    X_train, X_val, y_val = load_data()

    print("The first 5 elements of X_train are:\n", X_train[:5])
    print("The first 5 elements of X_val are\n", X_val[:5])
    print("The first 5 elements of y_val are\n", y_val[:5])
    print("The shape of X_train is:", X_train.shape)
    print("The shape of X_val is:", X_val.shape)
    print("The shape of y_val is: ", y_val.shape)

    plt.scatter(X_train[:, 0], X_train[:, 1], marker="x", c="b")
    plt.title("The first dataset")
    plt.ylabel("Throughput (mb/s)")
    plt.xlabel("Latency (ms)")
    plt.axis([0, 30, 0, 30])
    plt.show()

    mu, var = estimate_gaussian(X_train)
    print("Mean of each feature:", mu)
    print("Variance of each feature:", var)
    estimate_gaussian_test(estimate_gaussian)

    p = multivariate_gaussian(X_train, mu, var)
    visualize_fit(X_train, mu, var)
    plt.show()

    p_val = multivariate_gaussian(X_val, mu, var)
    epsilon, F1 = select_threshold(y_val, p_val)
    print("Best epsilon found using cross-validation: %e" % epsilon)
    print("Best F1 on Cross Validation Set: %f" % F1)
    select_threshold_test(select_threshold)

    outliers = p < epsilon
    visualize_fit(X_train, mu, var)
    plt.plot(
        X_train[outliers, 0], X_train[outliers, 1], "ro",
        markersize=10, markerfacecolor="none", markeredgewidth=2,
    )
    plt.show()


def run_high_dimensional_dataset():
    """Part 2: eleven features, no plots."""
    X_train_high, X_val_high, y_val_high = load_data_multi()

    print("The shape of X_train_high is:", X_train_high.shape)
    print("The shape of X_val_high is:", X_val_high.shape)
    print("The shape of y_val_high is: ", y_val_high.shape)

    mu_high, var_high = estimate_gaussian(X_train_high)
    p_high = multivariate_gaussian(X_train_high, mu_high, var_high)
    p_val_high = multivariate_gaussian(X_val_high, mu_high, var_high)
    epsilon_high, F1_high = select_threshold(y_val_high, p_val_high)

    print("Best epsilon found using cross-validation: %e" % epsilon_high)
    print("Best F1 on Cross Validation Set:  %f" % F1_high)
    print("# Anomalies found: %d" % np.sum(p_high < epsilon_high))


def main():
    run_2d_dataset()
    run_high_dimensional_dataset()


if __name__ == "__main__":
    main()
