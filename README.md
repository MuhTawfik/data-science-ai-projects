# Data Science and AI Projects

This repository showcases a small collection of concise yet powerful examples of data science and machine learning implemented in Python. Each script is self -contained and demonstrates a complete workflow—from loading data, preprocessing, model training, and evaluation—with clear documentation in the source code.

## Projects

| File | Description |
| --- | --- |
| **flower_classification.py** | A tutorial-style script that trains and evaluates Logistic Regression and Random Forest classifiers on the classic Iris flower dataset. It reports accuracy and a detailed classification report for each model. |
| **digits_classification.py** | A compact example of multiclass classification on the handwritten digits dataset using Logistic Regression. The script scales features, trains the model, and prints out accuracy and the classification report. |
| **handwritten_digit_clustering.py** | An unsupervised learning example that applies k-means clustering with PCA on the handwritten digits dataset. It assigns clusters to labels, reports clustering accuracy, silhouette score, and a classification report. |

## Getting Started

To run these examples locally, follow these steps:

1. **Clone this repository** (or download the files).

2. **Install the required packages**. The dependencies are listed in the `requirements.txt` file. You can install them using pip:

   ```bash
   pip install -r requirements.txt
   ```

3. **Run a script** by executing it with Python. For example:

   ```bash
   python flower_classification.py
   python digits_classification.py
   python handwritten_digit_clustering.py
   ```

Each script will print out evaluation metrics to the console.

## About

These projects are meant to be clear and approachable examples of typical machine learning workflows. They are ideal for educational purposes or as a starting point for more advanced projects. Feel free to contribute with additional examples or enhancements!
