"""
digits_classification.py
========================

This script demonstrates a simple approach to image
classification on the handwritten digits dataset provided by
scikit‑learn. It uses a logistic regression model to
recognize digits (0–9) from 8x8 grayscale pixel images. The
script loads the data, splits it into training and test sets,
scales the features, trains the model, and prints out the
accuracy and detailed classification report.

Run this script from the command line with:

    python digits_classification.py

Requirements:
    - Python 3.x
    - scikit‑learn

This example offers a quick introduction to multiclass
classification and can be extended with more advanced
algorithms or additional preprocessing techniques.
"""

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score


def main() -> None:
    """Load the digits dataset, train a logistic regression model, and report metrics."""
    # Load the digits dataset
    digits = load_digits()
    X = digits.data
    y = digits.target

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize and train the model
    model = LogisticRegression(max_iter=1000, n_jobs=-1, multi_class="auto")
    model.fit(X_train_scaled, y_train)

    # Make predictions
    y_pred = model.predict(X_test_scaled)

    # Evaluate
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    main()
