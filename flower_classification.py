"""
iris_classification.py
======================

This script provides a concise example of how to perform
classification on the famous Iris flower dataset using
two common machine learning algorithms: Logistic Regression
and Random Forest. It loads the dataset from scikit‑learn,
splits it into training and test sets, trains the models,
and prints evaluation metrics such as accuracy and a
classification report.

Run this script from the command line with:

    python iris_classification.py

Requirements:
    - Python 3.x
    - scikit‑learn
    - pandas

The script is designed to be simple and easy to understand,
serving both as a demonstration of basic machine learning
workflow and as a starting point for more advanced
experiments.
"""

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score


def main() -> None:
    """Load the Iris dataset, train two models, and print evaluation metrics."""
    # Load the dataset
    data = load_iris()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Initialize models
    lr = LogisticRegression(max_iter=200, n_jobs=-1, multi_class="auto")
    rf = RandomForestClassifier(n_estimators=200, random_state=42)

    # Train models
    lr.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    # Predictions
    lr_pred = lr.predict(X_test)
    rf_pred = rf.predict(X_test)

    # Evaluation
    print("=== Logistic Regression ===")
    print(f"Accuracy: {accuracy_score(y_test, lr_pred):.4f}")
    print(classification_report(y_test, lr_pred, target_names=data.target_names))

    print("=== Random Forest ===")
    print(f"Accuracy: {accuracy_score(y_test, rf_pred):.4f}")
    print(classification_report(y_test, rf_pred, target_names=data.target_names))


if __name__ == "__main__":
    main()
