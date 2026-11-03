Project Report: Logistic Regression Classification of Breast Masses

Student: Alexis Godinez
 Course: MSBA
 Project: Breast Mass Classification: Logistic Regression and Decision Thresholds

 Introduction
 This project investigates whether a logistic regression model can classify breast masses as malignant or benign using two predictor variables from the Wisconsin Diagnostic Breast Cancer Dataset. The project compares a custom NumPy implementation of logistic regression against a scikit-learn implementation and evaluates how changing the classification threshold affects model performance. The goal is to understand logistic regression, binary cross-entropy loss, gradient descent optimization, and classification metrics while maintaining a reproducible machine learning workflow.

Dataset Description

This project uses the Wisconsin Diagnostic Breast Cancer Dataset loaded directly from scikit-learn.

Python
from sklearn.datasets import load_breast_cancer
 

The original target labels were:
0 = Malignant
1 = Benign

The assignment required recoding the response variable as:
1 = Malignant
0 = Benign
 
The analysis used only two predictor variables:

Mean Radius
Mean Texture

These measurements describe characteristics of cell nuclei obtained from digitized images of breast-mass samples.

Data Preparation and Leakage Prevention
The dataset was partitioned into training, validation, and test subsets using stratified sampling to preserve class proportions.

The split process was:
80% Development
20% Test

The development data was later divided into:
75% Training
25% Validation

Final partition sizes:
Training = 341 observations
Validation = 114 observations
Test = 114 observations

Feature scaling was performed using StandardScaler. The scaler was fit exclusively on the training data and then applied to validation and test observations. This process prevents data leakage by ensuring that information from future observations is not incorporated into model training.

Logistic Regression Implementation
A logistic regression model was implemented from scratch using NumPy.

Model settings:
Learning Rate = 0.1
Epochs = 20,000

The optimization procedure used full-batch gradient descent and mean binary cross-entropy loss.
Training began with all coefficients initialized to zero.

Initial training loss:
0.693147

Final training loss:
0.246834
 

The substantial reduction in loss indicates successful convergence of the optimization process.
Cross-entropy loss is directly related to likelihood maximization. Minimizing cross-entropy increases the probability that the model assigns to the observed class labels. Consequently, minimizing cross-entropy is equivalent to maximizing the likelihood of the training data.

Coefficient Comparison

NumPy Logistic Regression
Coefficient	ValueIntercept	-0.727172
Mean Radius	3.656794
Mean Texture	1.009872

Scikit-Learn Logistic Regression
Coefficient	ValueIntercept	-0.727172
Mean Radius	3.656794
Mean Texture	1.009872

The coefficients produced by the custom implementation and the scikit-learn implementation were nearly identical.

Threshold Selection
Validation data was evaluated at the following thresholds:
0.20
0.30
0.40
0.50
0.60
0.70
0.80

The highest validation F1 score occurred at:
Threshold = 0.60
F1 = 0.831169

Therefore, 0.60 was selected as the final classification threshold.

Baseline Versus Logistic Regression
A DummyClassifier using the prior class distribution served as the baseline model.

Validation results:
Accuracy = 0.622807
Recall = 0.000000
F1 = 0.000000

Although the baseline achieved moderate accuracy, it failed to identify any malignant tumors. This demonstrates that accuracy alone is insufficient when evaluating classification models.
The logistic regression models substantially improved recall and F1 by correctly identifying malignant cases.

Test Results
NumPy Logistic Regression (Threshold = 0.50)
TN = 68
FP = 4
FN = 11
TP = 31
 
Accuracy = 0.868421
Precision = 0.885714
Recall = 0.738095
F1 = 0.805195

NumPy Logistic Regression (Threshold = 0.60)
TN = 69
FP = 3
FN = 14
TP = 28
 
Accuracy = 0.850877
Precision = 0.903226
Recall = 0.666667
F1 = 0.767123

Increasing the threshold improved precision but reduced recall. Although the selected threshold produced the highest validation F1 score, test F1 decreased relative to the default threshold. This outcome was reported without modification because threshold selection was performed solely using validation data.
Changing the classification threshold affects predicted labels but does not affect the probability estimates themselves. Therefore, log loss remains unchanged.