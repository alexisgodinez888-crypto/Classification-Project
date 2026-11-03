Breast Mass Classification: Logistic Regression and Decision Thresholds
Project Overview

This project implements binary classification using Logistic Regression to predict whether a breast mass is malignant or benign using two features from the Wisconsin Diagnostic Breast Cancer Dataset.

The project compares:

A custom NumPy Logistic Regression implementation using Gradient Descent
A scikit-learn Logistic Regression model
A DummyClassifier baseline model

The project also investigates how changing the classification threshold affects precision, recall, and F1 score.

Research Question

Can a logistic regression model trained on two breast-mass measurements accurately classify tumors as malignant or benign, and how does changing the classification threshold affect performance?

Dataset

Dataset:

Wisconsin Diagnostic Breast Cancer Dataset

Loaded directly from: from sklearn.datasets import load_breast_cancer
No external dataset download is required.

Original Target:
0 = Malignant
1 = Benign

Assignment Recoding:
1 = Malignant
0 = Benign

Features Used:
mean radius
mean texture

Only these two features were used throughout the analysis.

Prerequisites

Install:
Python 3.10 or newer
Git

Python Download:
https://www.python.org/downloads/

Git Download:
https://git-scm.com/downloads

Verify Installation:

Windows PowerShell:
python --version
git --version

macOS Terminal:
python3 --version
git --version

Clone the Repository

git clone https://github.com/alexisgodinez888-crypto/Classification-Project.git

Move into the project:
Windows:
cd Classification-Project

macOS:
cd Classification-Project

Windows PowerShell Setup
Create Virtual Environment:
PowerShell
python -m venv .venv

Activate:
PowerShell
.\.venv\Scripts\Activate.ps1

If PowerShell blocks activation:

PowerShell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
 
Then:
PowerShell
.\.venv\Scripts\Activate.ps1

Install Dependencies:
PowerShell
pip install -r requirements.txt

Run Project:
PowerShell
python logistic_regression_analysis.py

macOS Setup
Create Virtual Environment:
Shell
python3 -m venv .venv

Activate:
Shell
source .venv/bin/activate

Install Dependencies:
Shell
pip install -r requirements.txt
 

Run Project:
Shell
python3 logistic_regression_analysis.py

Generated Outputs
Running the script creates:
reports/training_loss.png
reports/threshold_results.csv
reports/assignment_results.csv