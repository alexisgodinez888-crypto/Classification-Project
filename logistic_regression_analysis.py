import pandas as pd
import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss
)
import matplotlib.pyplot as plt
from pathlib import Path


# -----------------------------
# Load Data
# -----------------------------

data = load_breast_cancer()

X = pd.DataFrame(
    data.data,
    columns=data.feature_names
)

X = X[["mean radius", "mean texture"]]

# 1 = malignant
# 0 = benign
y = (data.target == 0).astype(int)

# -----------------------------
# Split Indices
# -----------------------------

indices = range(len(X))

dev_idx, test_idx = train_test_split(
    list(indices),
    test_size=0.20,
    stratify=y,
    random_state=42
)

train_idx, val_idx = train_test_split(
    dev_idx,
    test_size=0.25,
    stratify=y[dev_idx],
    random_state=42
)

# -----------------------------
# Data Partitions
# -----------------------------

X_train = X.iloc[train_idx]
X_val = X.iloc[val_idx]
X_test = X.iloc[test_idx]

y_train = y[train_idx]
y_val = y[val_idx]
y_test = y[test_idx]

# -----------------------------
# Standardize
# -----------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# Add Intercept
# -----------------------------

X_train_aug = np.column_stack(
    (np.ones(len(X_train_scaled)), X_train_scaled)
)
X_val_aug = np.column_stack(
    (np.ones(len(X_val_scaled)), X_val_scaled)
)

X_test_aug = np.column_stack(
    (np.ones(len(X_test_scaled)), X_test_scaled)
)
# -----------------------------
# Functions
# -----------------------------

def sigmoid(z):
    return np.exp(-np.logaddexp(0, -z))

def compute_loss(X, y, beta):
    z = X @ beta
    return np.mean(
        np.logaddexp(0, z) - y * z
    )

# -----------------------------
# Initialization
# -----------------------------

beta = np.zeros(X_train_aug.shape[1])

learning_rate = 0.1
epochs = 20000

loss_history = []

initial_loss = compute_loss(
    X_train_aug,
    y_train,
    beta
)

loss_history.append(initial_loss)

print("Initial Loss:", initial_loss)

# ----------------------------------
# sklearn Logistic Regression
# ----------------------------------

sk_model = LogisticRegression(
    C=float("inf"),
    solver="lbfgs",
    max_iter=10000,
    tol=1e-9
)

sk_model.fit(
    X_train_scaled,
    y_train
)

print("\nSKLEARN COEFFICIENTS")

print("Intercept:", sk_model.intercept_[0])

print("Mean Radius:", sk_model.coef_[0][0])

print("Mean Texture:", sk_model.coef_[0][1])

# ----------------------------------
# Dummy Baseline
# ----------------------------------

baseline = DummyClassifier(
    strategy="prior"
)

baseline.fit(
    X_train_scaled,
    y_train
)

print("\nBaseline trained successfully")

# -----------------------------
# Gradient Descent
# -----------------------------

for epoch in range(epochs):

    z = X_train_aug @ beta

    p = sigmoid(z)

    gradient = (
        X_train_aug.T @ (p - y_train)
    ) / len(y_train)

    beta -= learning_rate * gradient

    loss = compute_loss(
        X_train_aug,
        y_train,
        beta
    )

    loss_history.append(loss)

print("Final Loss:", loss_history[-1])

print("\nCoefficients")

print("Intercept:", beta[0])
print("Mean Radius:", beta[1])

# -----------------------------
# Training Loss Plot
# -----------------------------

reports_dir = Path("reports")
reports_dir.mkdir(exist_ok=True)

plt.figure(figsize=(8, 5))

plt.plot(loss_history)

plt.title("Training Loss vs Epoch")

plt.xlabel("Epoch")

plt.ylabel("Binary Cross Entropy Loss")

plt.grid(True)

plt.tight_layout()

plt.savefig(
    reports_dir / "training_loss.png"
)

plt.close()

print("Mean Texture:", beta[2])

# ----------------------------------
# Validation Probability Comparison
# ----------------------------------

numpy_val_prob = sigmoid(
    X_val_aug @ beta
)

sklearn_val_prob = sk_model.predict_proba(
    X_val_scaled
)[:, 1]

mean_probability_difference = np.mean(
    np.abs(
        numpy_val_prob - sklearn_val_prob
    )
)

print("\nMean Probability Difference:")
print(mean_probability_difference)

print("\nSaved training_loss.png")

# =====================================
# THRESHOLD TESTING
# =====================================

thresholds = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]

threshold_results = []

for threshold in thresholds:

    predictions = (
        numpy_val_prob >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_val,
        predictions,
        labels=[0, 1]
    ).ravel()

    accuracy = accuracy_score(
        y_val,
        predictions
    )

    precision = precision_score(
        y_val,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_val,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_val,
        predictions,
        zero_division=0
    )

    threshold_results.append([
        threshold,
        tn,
        fp,
        fn,
        tp,
        accuracy,
        precision,
        recall,
        f1
    ])

threshold_df = pd.DataFrame(
    threshold_results,
    columns=[
        "threshold",
        "tn",
        "fp",
        "fn",
        "tp",
        "accuracy",
        "precision",
        "recall",
        "f1"
    ]
)

threshold_df.to_csv(
    "reports/threshold_results.csv",
    index=False
)

print("\nThreshold Results")

print(threshold_df)
# =====================================
# SELECTED THRESHOLD
# =====================================

selected_threshold = 0.60

print("\nSelected Threshold:", selected_threshold)

# =====================================
# HELPER FUNCTION
# =====================================

def evaluate_model(
    split_name,
    model_name,
    policy_name,
    threshold,
    y_true,
    probabilities
):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    ).ravel()

    accuracy = accuracy_score(
        y_true,
        predictions
    )

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0
    )

    loss = log_loss(
        y_true,
        probabilities
    )

    return [
        split_name,
        model_name,
        policy_name,
        threshold,
        tn,
        fp,
        fn,
        tp,
        accuracy,
        precision,
        recall,
        f1,
        loss
    ]

# =====================================
# PROBABILITIES
# =====================================

numpy_val_prob = sigmoid(
    X_val_aug @ beta
)

numpy_test_prob = sigmoid(
    X_test_aug @ beta
)

sk_val_prob = sk_model.predict_proba(
    X_val_scaled
)[:, 1]

sk_test_prob = sk_model.predict_proba(
    X_test_scaled
)[:, 1]

baseline_val_prob = baseline.predict_proba(
    X_val_scaled
)[:, 1]

baseline_test_prob = baseline.predict_proba(
    X_test_scaled
)[:, 1]

# =====================================
# RESULTS TABLE
# =====================================

results = []

# Validation

results.append(
    evaluate_model(
        "validation",
        "baseline",
        "default",
        0.50,
        y_val,
        baseline_val_prob
    )
)

results.append(
    evaluate_model(
        "validation",
        "numpy",
        "default",
        0.50,
        y_val,
        numpy_val_prob
    )
)

results.append(
    evaluate_model(
        "validation",
        "numpy",
        "selected",
        selected_threshold,
        y_val,
        numpy_val_prob
    )
)

results.append(
    evaluate_model(
        "validation",
        "sklearn",
        "default",
        0.50,
        y_val,
        sk_val_prob
    )
)

results.append(
    evaluate_model(
        "validation",
        "sklearn",
        "selected",
        selected_threshold,
        y_val,
        sk_val_prob
    )
)

# Test

results.append(
    evaluate_model(
        "test",
        "baseline",
        "default",
        0.50,
        y_test,
        baseline_test_prob
    )
)

results.append(
    evaluate_model(
        "test",
        "numpy",
        "default",
        0.50,
        y_test,
        numpy_test_prob
    )
)

results.append(
    evaluate_model(
        "test",
        "numpy",
        "selected",
        selected_threshold,
        y_test,
        numpy_test_prob
    )
)

results.append(
    evaluate_model(
        "test",
        "sklearn",
        "default",
        0.50,
        y_test,
        sk_test_prob
    )
)

results.append(
    evaluate_model(
        "test",
        "sklearn",
        "selected",
        selected_threshold,
        y_test,
        sk_test_prob
    )
)

assignment_df = pd.DataFrame(
    results,
    columns=[
        "split",
        "model",
        "policy",
        "threshold",
        "tn",
        "fp",
        "fn",
        "tp",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "log_loss"
    ]
)

assignment_df.to_csv(
    "reports/assignment_results.csv",
    index=False
)

print("\nAssignment Results")
print(assignment_df)