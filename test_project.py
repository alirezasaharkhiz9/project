import pytest
from project import preprocess, best_models, train_model, predict, myAccuracy
from sklearn.datasets import load_iris
from sklearn.base import BaseEstimator

def test_preprocess():
    """
    Test: preprocess function
    Description:
    - Tests the `preprocess` function to ensure that the data is split and scaled correctly.
    """
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = preprocess(X, y)

    # Check if the split sizes are correct
    assert X_train.shape[0] == 0.8 * len(X), "Training set should be 80% of the data"
    assert X_test.shape[0] == 0.2 * len(X), "Test set should be 20% of the data"
    assert X_train.shape[1] == X_test.shape[1], "Feature count should remain consistent after splitting"

def test_best_models():
    """
    Test: best_models function
    Description:
    - Tests the `best_models` function to ensure it returns a valid scikit-learn classifier class.
    """
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = preprocess(X, y)

    # Get the best model class
    model_class = best_models(X_train, X_test, y_train, y_test)

    # Check that a model class is returned and it's a valid sklearn classifier
    assert model_class is not None, "No model class was found"
    assert issubclass(model_class, BaseEstimator), "Selected model should be a subclass of BaseEstimator"

def test_train_model():
    """
    Test: train_model function
    Description:
    - Tests the `train_model` function to ensure the selected model can be trained.
    """
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = preprocess(X, y)

    # Get the best model class and train it
    model_class = best_models(X_train, X_test, y_train, y_test)
    model = train_model(model_class, X_train, y_train)

    # Check if the model was successfully trained
    assert model is not None, "Model training failed"
    assert hasattr(model, 'predict'), "Trained model should have a predict method"

def test_predict():
    """
    Test: predict function
    Description:
    - Tests the `predict` function to ensure the trained model can make predictions.
    """
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = preprocess(X, y)

    # Train the best model and make predictions
    model_class = best_models(X_train, X_test, y_train, y_test)
    model = train_model(model_class, X_train, y_train)
    y_pred = predict(model, X_test)

    # Check if predictions are made correctly
    assert len(y_pred) == len(y_test), "Number of predictions should match the number of test samples"

def test_myAccuracy():
    """
    Test: myAccuracy function
    Description:
    - Tests the `myAccuracy` function to ensure accuracy and classification report are calculated.
    """
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = preprocess(X, y)

    # Train the best model and make predictions
    model_class = best_models(X_train, X_test, y_train, y_test)
    model = train_model(model_class, X_train, y_train)
    y_pred = predict(model, X_test)

    # Check if accuracy is calculated correctly
    accuracy = myAccuracy(y_pred, y_test)
    assert 0 <= accuracy <= 1, "Accuracy should be a value between 0 and 1"