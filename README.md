# Drug Classification using Machine Learning and Neural Networks

## Project Description
This project applies machine learning models, such as the Artificial Neural Network (ANN) and the Decision Tree classifier, for drug classification based on patient characteristics. The dataset, drugdataset.csv, includes features such as age, sex, blood pressure, cholesterol levels, and sodium-to-potassium ratio, and the target variable is the prescribed drug (Drug).

The aim of this work is to emphasize the working of neural network and decision tree in case of multi-class classification while showing their various metrics comprising accuracy, precision, recall, and F1-score by using the results.

## Getting Started
The following instructions will help to set up and execute the project on local machine.

### Prerequisites
To run this project, we need the following installed:
**Python**
Required Python libraries:
  pandas
  numpy
  matplotlib
  scikit-learn

we can install the required libraries using the command:
#Load Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

File Contents
drug_classification.py
This Python file performs the following tasks:

Data Loading:

Reads the dataset (drugdataset.csv) using pandas.
Displays the first few rows and statistical summary of the data.

Data Exploration:
Identifies the unique classes of drugs in the dataset.

Data Preparation:
Splits the dataset into features (X) and target (y).
Uses train_test_split to divide the data into training (80%) and testing (20%) sets, stratifying by the target variable for balanced classes.
Scales the features using StandardScaler to standardize the dataset.

Artificial Neural Network (ANN):
Implements a Multi-Layer Perceptron (MLP) classifier with:
Three hidden layers of sizes 5, 4, and 5.
ReLU activation function and Adam optimizer.
A maximum of 10,000 iterations.
Trains the model on the training data.
Makes predictions on the test set.
Evaluates the model using a confusion matrix and classification report, showing precision, recall, and F1-scores for each drug class.

Decision Tree Classifier:
Implements a Decision Tree classifier with a default configuration.
Trains the model and evaluates it using the same metrics as the ANN for comparison.

Results and Outputs:
Displays the confusion matrix and classification report for both models.
Shows the accuracy of each model and their ability to classify the different drug classes.

Installing
Clone the repository to local machine:

Navigate to the project directory:
DATA1200-Drug-Classification
Place the drugdataset.csv file in the project directory.

Running the Code
Open a terminal or command prompt.
Run the Python script:
The output will display:

Confusion matrices for the ANN and Decision Tree models.
Classification reports with detailed metrics for both models.
Example Output:

Confusion Matrix for ANN:
[[ 5  0  0  0  0]
 [ 0  2  0  0  1]
 [ 0  0  3  0  0]
 [ 0  0  0 11  0]
 [ 0  0  0  1 17]]

Classification Report for ANN:
              precision    recall  f1-score   support

       drugA       1.00      1.00      1.00         5
       drugB       1.00      0.67      0.80         3
       drugC       1.00      1.00      1.00         3
       drugX       0.92      1.00      0.96        11
       drugY       0.94      0.94      0.94        18

    accuracy                           0.95        40
   macro avg       0.97      0.92      0.94        40
weighted avg       0.95      0.95      0.95        40

Running the Tests
The project evaluates the models' performance using the following metrics:

Confusion Matrix: Shows the number of correct and incorrect predictions for each class.
Classification Report: Includes precision, recall, and F1-scores for each drug class.
Testing is automatically performed when running the script. Results are displayed for both the ANN and Decision Tree classifiers.

Deployment
This project is not intended for deployment in live systems. It is a demonstration of machine learning techniques for academic purposes.

Built With
Python  - Programming language
Pandas - Data manipulation library
NumPy - Numerical computing library
Matplotlib - Data visualization library
scikit-learn - Machine learning library
Authors
[Harmeet Kaur]
Data Analytics Student, Durham College

Acknowledgments
Special thanks to:

Durham College instructors for their guidance.
scikit-learn library developers for their tools and documentation.
The creators of the dataset used in this project.



