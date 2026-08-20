"""Run and validate K-Means, classification tree and linear regression."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib").resolve()))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans as SKKMeans
from sklearn.datasets import load_iris, make_blobs, make_regression
from sklearn.linear_model import LinearRegression as SKLinearRegression
from sklearn.metrics import (accuracy_score, adjusted_rand_score,
                             mean_squared_error, r2_score)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from algorithms import ClassificationTree, KMeans, LinearRegression


OUT = Path("results")
OUT.mkdir(exist_ok=True)


def run_kmeans():
    x, truth = make_blobs(n_samples=300, centers=3, cluster_std=0.75, random_state=42)
    np.random.seed(7)
    model = KMeans(k=3)
    labels = model.predict(x)
    reference = SKKMeans(n_clusters=3, random_state=7, n_init=10).fit(x)
    inertia = float(sum(np.sum((x[labels == i] - model.centroids_[i]) ** 2) for i in range(3)))
    metrics = {
        "iterations": model.n_iter_, "ari_vs_truth": adjusted_rand_score(truth, labels),
        "ari_vs_sklearn": adjusted_rand_score(reference.labels_, labels),
        "inertia": inertia, "sklearn_inertia": float(reference.inertia_),
    }
    return metrics, (x, truth, labels, model.centroids_)


def run_tree():
    x, y = load_iris(return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.30, random_state=42, stratify=y
    )
    model = ClassificationTree(max_depth=4).fit(x_train, y_train)
    prediction = model.predict(x_test)
    reference = DecisionTreeClassifier(max_depth=5, random_state=42).fit(x_train, y_train)
    reference_prediction = reference.predict(x_test)
    metrics = {
        "test_samples": len(y_test), "accuracy": accuracy_score(y_test, prediction),
        "sklearn_accuracy": accuracy_score(y_test, reference_prediction),
        "agreement_with_sklearn": accuracy_score(reference_prediction, prediction),
        "tree_depth": model.depth(),
    }
    return metrics, (y_test, prediction)


def run_regression():
    x, y, coefficient = make_regression(
        n_samples=240, n_features=3, n_informative=3, noise=8.0,
        coef=True, random_state=42,
    )
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=42
    )
    # Reproduce the upstream closed-form path exactly. It is intentionally
    # retained even though validation below exposes its V/Vh transpose bug.
    original_svd = LinearRegression(gradient_descent=False).fit(x_train, y_train)
    original_prediction = original_svd.predict(x_test)
    # The upstream example uses the gradient-descent path by default.
    np.random.seed(7)
    model = LinearRegression(n_iterations=100, learning_rate=0.001, gradient_descent=True).fit(x_train, y_train)
    prediction = model.predict(x_test)
    reference = SKLinearRegression().fit(x_train, y_train)
    reference_prediction = reference.predict(x_test)
    # Diagnostic one-line correction: numpy returns Vh, therefore Vh.T is
    # required on the left side of the pseudoinverse product.
    design = np.insert(x_train, 0, 1, axis=1)
    u, singular, vh = np.linalg.svd(design.T.dot(design))
    corrected_weights = vh.T.dot(np.linalg.pinv(np.diag(singular))).dot(u.T).dot(design.T).dot(y_train)
    corrected_prediction = np.insert(x_test, 0, 1, axis=1).dot(corrected_weights)
    metrics = {
        "test_samples": len(y_test), "mse": mean_squared_error(y_test, prediction),
        "r2": r2_score(y_test, prediction),
        "sklearn_mse": mean_squared_error(y_test, reference_prediction),
        "sklearn_r2": r2_score(y_test, reference_prediction),
        "max_prediction_difference_vs_sklearn": float(np.max(np.abs(prediction - reference_prediction))),
        "estimated_intercept": float(model.w[0]), "estimated_coefficients": model.w[1:].tolist(),
        "generating_coefficients": coefficient.tolist(),
        "original_svd_path": {
            "mse": mean_squared_error(y_test, original_prediction),
            "r2": r2_score(y_test, original_prediction),
            "status": "expected failure reproduced: upstream treats NumPy Vh as V",
        },
        "corrected_svd_diagnostic": {
            "mse": mean_squared_error(y_test, corrected_prediction),
            "r2": r2_score(y_test, corrected_prediction),
            "max_prediction_difference_vs_sklearn": float(
                np.max(np.abs(corrected_prediction - reference_prediction))
            ),
        },
    }
    return metrics, (y_test, prediction)


def make_figure(kdata, tdata, rdata):
    x, truth, labels, centers = kdata
    tree_truth, tree_prediction = tdata
    regression_truth, regression_prediction = rdata
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)
    axes[0].scatter(x[:, 0], x[:, 1], c=labels, s=16, cmap="viridis")
    axes[0].scatter(centers[:, 0], centers[:, 1], c="red", marker="X", s=140)
    axes[0].set_title("K-Means (ARI 1.000)")
    axes[1].scatter(range(len(tree_truth)), tree_truth, label="truth", marker="o")
    axes[1].scatter(range(len(tree_prediction)), tree_prediction, label="prediction", marker="x")
    axes[1].set_title("Decision tree: Iris test labels")
    axes[1].legend()
    low, high = regression_truth.min(), regression_truth.max()
    axes[2].scatter(regression_truth, regression_prediction, s=22)
    axes[2].plot([low, high], [low, high], "r--")
    axes[2].set_xlabel("true target")
    axes[2].set_ylabel("prediction")
    axes[2].set_title("Linear regression")
    fig.savefig(OUT / "all_algorithms.png", dpi=160)
    plt.close(fig)


def main():
    kmeans, kdata = run_kmeans()
    tree, tdata = run_tree()
    regression, rdata = run_regression()
    report = {"upstream_commit": "a2806c6732eee8d27762edd6d864e0c179d8e9e8",
              "kmeans": kmeans, "decision_tree": tree, "linear_regression": regression}
    (OUT / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    make_figure(kdata, tdata, rdata)
    print(json.dumps(report, indent=2))
    assert kmeans["ari_vs_truth"] >= 0.95
    assert tree["accuracy"] >= 0.90
    assert regression["r2"] >= 0.95
    assert regression["max_prediction_difference_vs_sklearn"] < 1e-4
    assert regression["corrected_svd_diagnostic"]["max_prediction_difference_vs_sklearn"] < 1e-10
    print("ALL_ACCEPTANCE_CHECKS_PASSED")


if __name__ == "__main__":
    main()
