#!/usr/bin/env python3
"""
handwritten_digit_clustering.py
--------------------------------

This script demonstrates unsupervised learning using k-means clustering on the classic
handwritten digits dataset from scikit-learn. It reduces the dimensionality of the data
with Principal Component Analysis (PCA) for faster and more effective clustering, trains
a k-means model to find 10 clusters (one for each digit), and then evaluates the
clusters by mapping them back to the original digit labels. The script reports the
clustering accuracy, a detailed classification report, and the silhouette score to
quantify the quality of the clustering.

Usage:
    python handwritten_digit_clustering.py

Requirements:
    This script depends on scikit-learn and numpy. Install the dependencies using
    `pip install -r requirements.txt` from the root of this repository.
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report, silhouette_score


def main() -> None:
    """Load data, perform clustering, and report evaluation metrics."""
    # Load the handwritten digits dataset
    digits = load_digits()
    data = digits.data
    true_labels = digits.target

    # Reduce dimensionality with PCA to speed up clustering and mitigate noise.
    # We keep enough components to capture the majority of variance (e.g., 10).
    pca = PCA(n_components=10, random_state=42)
    data_reduced = pca.fit_transform(data)

    # Fit k-means clustering with 10 clusters (one for each digit)
    kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(data_reduced)

    # Evaluate clustering by assigning each cluster the most frequent true label
    cluster_to_label: dict[int, int] = {}
    for cluster_id in range(10):
        # Select points that belong to this cluster
        mask = cluster_labels == cluster_id
        if np.sum(mask) == 0:
            # Empty cluster (unlikely but handled for completeness)
            cluster_to_label[cluster_id] = -1
            continue
        # Determine the most frequent true label in this cluster
        unique_labels, counts = np.unique(true_labels[mask], return_counts=True)
        dominant_label = unique_labels[np.argmax(counts)]
        cluster_to_label[cluster_id] = int(dominant_label)

    # Map cluster assignments to predicted labels
    predicted_labels = np.array([cluster_to_label[cluster] for cluster in cluster_labels])

    # Compute accuracy by comparing predicted labels to true labels
    accuracy = accuracy_score(true_labels, predicted_labels)

    # Generate a classification report for more detailed metrics
    report = classification_report(true_labels, predicted_labels, digits=3)

    # Compute the silhouette score to measure cluster cohesion and separation
    sil_score = silhouette_score(data_reduced, cluster_labels)

    # Display results
    print("K-Means Clustering on Handwritten Digits")
    print("--------------------------------------")
    print(f"Clustering accuracy: {accuracy * 100:.2f}%")
    print(f"Silhouette score: {sil_score:.3f}")
    print()
    print("Classification report based on cluster assignments:")
    print(report)


if __name__ == "__main__":
    main()
