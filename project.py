"""
Project Name: Automated Machine Learning
Author: Alireza Saharkhiz
Date: Sunday, 8 September 2024

Description:
This project demonstrates an automated machine learning pipeline using the LazyClassifier library, which helps in automatically testing various classifiers and selecting the best-performing model. The project includes preprocessing, model selection, training, prediction, and accuracy evaluation. 

The main function coordinates the entire process, while additional custom functions handle each step individually.

Modules Used:
- numpy
- pandas
- scikit-learn
- LazyPredict (LazyClassifier)
- inspect
- warnings

Structure:
- preprocess: Handles data scaling and train-test splitting
- best_models: Automatically selects the best model using LazyClassifier
- train_model: Trains the selected model on the training data
- predict: Uses the trained model to predict outcomes on the test data
- myAccuracy: Computes and prints the classification report and accuracy score
- main: Orchestrates the process from data loading to accuracy evaluation
"""

import numpy as np
import pandas as pd
import sklearn
from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from lazypredict.Supervised import LazyClassifier
import inspect
import warnings

# Suppress warnings to make the output cleaner
warnings.filterwarnings('ignore')

def preprocess(X, y):
    """
    Function: preprocess
    Description: 
    - Splits the input data (X, y) into training and testing sets.
    - Applies standardization (scaling) to ensure the data has zero mean and unit variance.
    
    Parameters:
    - X: Feature matrix (input data)
    - y: Target labels
    
    Returns:
    - X_train, X_test: Scaled training and testing features
    - y_train, y_test: Training and testing labels
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test

def best_models(X_train, X_test, y_train, y_test):
    """
    Function: best_models
    Description:
    - Uses LazyClassifier to automatically fit various classifiers and return a ranking of the best models.
    - Extracts the names of the top 10 classifiers based on performance.
    - Searches through scikit-learn's module for the corresponding class of the top-performing model.

    Parameters:
    - X_train, X_test: Scaled training and testing features
    - y_train, y_test: Training and testing labels

    Returns:
    - model_class: The best-performing model class found within scikit-learn
    """
    clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
    models, predictions = clf.fit(X_train, X_test, y_train, y_test)
    print(models.iloc[0:10, :])  # Display top 10 models for reference
    model_names = models.index[:10]  # Top 10 models by performance

    def find_sklearn_class(model_names):
        """
        Helper Function: find_sklearn_class
        Description:
        - Searches through scikit-learn modules to find a matching classifier class
        - Only returns the first match it finds

        Parameters:
        - model_names: List of top model names from LazyClassifier

        Returns:
        - obj: The first matched model class from scikit-learn
        """
        for model_name in model_names:
            for module_name in dir(sklearn):
                module = getattr(sklearn, module_name)
                if inspect.ismodule(module):
                    for name, obj in inspect.getmembers(module):
                        if inspect.isclass(obj) and name == model_name:
                            return obj
        return None

    model_class = find_sklearn_class(model_names)
    return model_class

def train_model(model_class, X_train, y_train):
    """
    Function: train_model
    Description:
    - Instantiates and trains the model class returned by best_models on the training data.
    
    Parameters:
    - model_class: The model class returned by best_models
    - X_train: Scaled training features
    - y_train: Training labels

    Returns:
    - model: The trained model
    """
    model = model_class()  # Instantiate the model
    model.fit(X_train, y_train)  # Train the model on the training data
    return model

def predict(model, X_test):
    """
    Function: predict
    Description:
    - Uses the trained model to predict the labels for the test dataset.
    
    Parameters:
    - model: The trained model
    - X_test: Scaled testing features

    Returns:
    - y_pred: Predicted labels for the test data
    """
    y_pred = model.predict(X_test)  # Model makes predictions
    return y_pred

def myAccuracy(y_pred, y_test):
    """
    Function: myAccuracy
    Description:
    - Evaluates the performance of the model by comparing predicted labels with actual test labels.
    - Prints a classification report and returns the accuracy score.
    
    Parameters:
    - y_pred: Predicted labels
    - y_test: Actual test labels

    Returns:
    - acc: Accuracy score of the model
    """
    report = classification_report(y_test, y_pred)  # Generate classification report
    print(report)  # Print the report
    acc = accuracy_score(y_test, y_pred)  # Calculate accuracy score
    return acc

def main():
    """
    Function: main
    Description:
    - Orchestrates the complete machine learning pipeline:
      1. Loads the dataset
      2. Preprocesses the data
      3. Identifies the best-performing model using LazyClassifier
      4. Trains the model
      5. Makes predictions on test data
      6. Evaluates the model's performance
    
    Steps:
    1. Load dataset (Iris dataset in this case)
    2. Preprocess the data
    3. Identify the best model
    4. Train the selected model
    5. Make predictions on test data
    6. Print accuracy and classification report
    """
    # Load the Iris dataset (for demonstration purposes)
    iris = load_iris()
    X, y = iris.data, iris.target

    # Step 1: Preprocess the dataset
    X_train, X_test, y_train, y_test = preprocess(X, y)

    # Step 2: Find the best model using LazyClassifier
    model_class = best_models(X_train, X_test, y_train, y_test)

    if model_class:
        # Step 3: Train the best model
        model = train_model(model_class, X_train, y_train)

        # Step 4: Make predictions on the test dataset
        y_pred = predict(model, X_test)

        # Step 5: Evaluate model performance
        accuracy = myAccuracy(y_pred, y_test)
        print(f"Accuracy: {accuracy}")
    else:
        print("No suitable model found.")

# Run the main function when the script is executed
if __name__ == "__main__":
    main()
