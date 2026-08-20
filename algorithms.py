"""Standalone reproductions extracted from ML-From-Scratch at a2806c6."""

from __future__ import annotations

import math
import numpy as np


def euclidean_distance(x1, x2):
    return float(np.sqrt(sum(np.power(x1[i] - x2[i], 2) for i in range(len(x1)))))


class KMeans:
    def __init__(self, k=2, max_iterations=500):
        self.k, self.max_iterations = k, max_iterations
        self.centroids_, self.n_iter_ = None, 0

    def _init_random_centroids(self, x):
        centroids = np.zeros((self.k, x.shape[1]))
        for i in range(self.k):
            centroids[i] = x[np.random.choice(range(len(x)))]
        return centroids

    def _create_clusters(self, centroids, x):
        clusters = [[] for _ in range(self.k)]
        for sample_i, sample in enumerate(x):
            distances = [euclidean_distance(sample, centroid) for centroid in centroids]
            clusters[int(np.argmin(distances))].append(sample_i)
        return clusters

    def predict(self, x):
        centroids = self._init_random_centroids(x)
        for iteration in range(1, self.max_iterations + 1):
            clusters = self._create_clusters(centroids, x)
            previous = centroids
            centroids = np.zeros_like(centroids)
            for i, cluster in enumerate(clusters):
                centroids[i] = np.mean(x[cluster], axis=0)
            self.n_iter_ = iteration
            if not (centroids - previous).any():
                break
        self.centroids_ = centroids
        labels = np.zeros(len(x), dtype=int)
        for cluster_i, cluster in enumerate(clusters):
            labels[cluster] = cluster_i
        return labels


class DecisionNode:
    def __init__(self, feature_i=None, threshold=None, value=None,
                 true_branch=None, false_branch=None):
        self.feature_i, self.threshold, self.value = feature_i, threshold, value
        self.true_branch, self.false_branch = true_branch, false_branch


def entropy(y):
    labels, counts = np.unique(y, return_counts=True)
    probabilities = counts / counts.sum()
    return float(-np.sum(probabilities * np.log2(probabilities)))


class ClassificationTree:
    """Faithful classification subset of the upstream DecisionTree implementation."""

    def __init__(self, min_samples_split=2, min_impurity=1e-7, max_depth=float("inf")):
        self.min_samples_split = min_samples_split
        self.min_impurity = min_impurity
        self.max_depth = max_depth
        self.root = None

    def fit(self, x, y):
        self.root = self._build_tree(x, y)
        return self

    def _build_tree(self, x, y, current_depth=0):
        largest_impurity, best_criteria, best_sets = 0.0, None, None
        y_col = np.expand_dims(y, axis=1) if y.ndim == 1 else y
        xy = np.concatenate((x, y_col), axis=1)
        n_samples, n_features = x.shape
        if n_samples >= self.min_samples_split and current_depth <= self.max_depth:
            for feature_i in range(n_features):
                for threshold in np.unique(x[:, feature_i]):
                    mask = xy[:, feature_i] >= threshold
                    xy1, xy2 = xy[mask], xy[~mask]
                    if len(xy1) and len(xy2):
                        y1, y2 = xy1[:, n_features:], xy2[:, n_features:]
                        p = len(y1) / len(y_col)
                        gain = entropy(y_col) - p * entropy(y1) - (1 - p) * entropy(y2)
                        if gain > largest_impurity:
                            largest_impurity = gain
                            best_criteria = (feature_i, threshold)
                            best_sets = (xy1[:, :n_features], y1, xy2[:, :n_features], y2)
        if largest_impurity > self.min_impurity:
            left_x, left_y, right_x, right_y = best_sets
            return DecisionNode(
                feature_i=best_criteria[0], threshold=best_criteria[1],
                true_branch=self._build_tree(left_x, left_y, current_depth + 1),
                false_branch=self._build_tree(right_x, right_y, current_depth + 1),
            )
        values, counts = np.unique(y_col, return_counts=True)
        return DecisionNode(value=values[np.argmax(counts)])

    def predict_value(self, sample, tree=None):
        tree = self.root if tree is None else tree
        if tree.value is not None:
            return tree.value
        branch = tree.true_branch if sample[tree.feature_i] >= tree.threshold else tree.false_branch
        return self.predict_value(sample, branch)

    def predict(self, x):
        return np.asarray([self.predict_value(sample) for sample in x])

    def depth(self, node=None):
        node = self.root if node is None else node
        return 0 if node.value is not None else 1 + max(self.depth(node.true_branch), self.depth(node.false_branch))


class LinearRegression:
    """Upstream linear regression with gradient-descent and SVD paths."""

    def __init__(self, n_iterations=100, learning_rate=0.001, gradient_descent=True):
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.gradient_descent = gradient_descent
        self.w = None
        self.training_errors = []

    def fit(self, x, y):
        design = np.insert(x, 0, 1, axis=1)
        if not self.gradient_descent:
            u, singular, v = np.linalg.svd(design.T.dot(design))
            inverse = v.dot(np.linalg.pinv(np.diag(singular))).dot(u.T)
            self.w = inverse.dot(design.T).dot(y)
            return self
        limit = 1 / math.sqrt(design.shape[1])
        self.w = np.random.uniform(-limit, limit, design.shape[1])
        for _ in range(self.n_iterations):
            prediction = design.dot(self.w)
            self.training_errors.append(float(np.mean(0.5 * (y - prediction) ** 2)))
            gradient = -(y - prediction).dot(design)
            self.w -= self.learning_rate * gradient
        return self

    def predict(self, x):
        return np.insert(x, 0, 1, axis=1).dot(self.w)
