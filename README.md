# 🚀 Automated Machine Learning with LazyClassifier 🌟

Welcome to Automated Machine Learning with LazyClassifier, a powerful Python project designed to automate the process of selecting, training, and evaluating machine learning models—all with just a few lines of code! 🎉

This project simplifies the typical data science workflow by automatically testing and ranking several machine learning models, choosing the best one for your dataset, and providing detailed evaluation metrics. Forget about manual model selection—let the machine do the work for you!

## Video Demo:

<https://youtu.be/cc5cYdXIcvE?si=d6TQ1qRXYb29GExc>

## 💡 Project Overview

In this project, we leverage the LazyPredict library, specifically the LazyClassifier, to automate the process of:

-   Preprocessing your dataset (scaling and splitting).
-   Testing and Ranking Models from a wide variety of classifiers.
-   Training the best model found.
-   Predicting and Evaluating the model's performance with detailed metrics like accuracy and classification reports.

Whether you're a data science enthusiast, a machine learning practitioner, or a curious developer, this project will show you how quickly you can build a performant machine learning pipeline with minimal effort.

## 🛠️ Key Features

-   🔍 **Automated Model Selection**: Automatically test multiple classifiers and choose the best one based on performance.
-   🧠 **No Manual Tuning Required**: LazyClassifier evaluates models without needing manual hyperparameter tuning.
-   📊 **Detailed Evaluation**: Get classification reports and accuracy metrics to assess the model’s performance.
-   🚀 **Preprocessing Pipeline**: Standardize your dataset with built-in preprocessing.

## 📂 Project Structure

```         
.
├── project.py          # Main file that implements the automated machine learning pipeline
├── test_project.py     # Unit tests for the key functions in project.py
├── requirements.txt    # List of required libraries (LazyPredict, scikit-learn, etc.)
└── README.md           # This documentation file!
```

## 🔧 How It Works

1.  **Preprocessing the Dataset**\
    The `preprocess` function automatically scales and splits your data into training and testing sets, making sure the model has high-quality data for training.

2.  **Model Selection with LazyClassifier**\
    The LazyClassifier quickly evaluates multiple classifiers from scikit-learn, ranks them based on performance, and identifies the best model for your data.

3.  **Model Training**\
    Once the best model is selected, the `train_model` function trains it on your dataset, preparing it for making predictions.

4.  **Making Predictions & Evaluating Performance**\
    With the trained model, we use the `predict` function to classify the test set. The performance is then evaluated with the `myAccuracy` function, which provides a detailed classification report and accuracy score.

## 🎬 Example Workflow

``` bash
# Step 1: Clone the repository and navigate to the project folder
git clone https://github.com/yourusername/automated-ml-lazyclassifier.git
cd automated-ml-lazyclassifier

# Step 2: Install required dependencies
pip install -r requirements.txt

# Step 3: Run the project
python project.py

# Step 4: Run tests (optional, for developers)
pytest test_project.py
```

## 🚀 Demo

Here's how the project works on the famous Iris dataset:

-   The LazyClassifier tests 10+ classifiers in just seconds! ⚡
-   The top-performing model is automatically selected.
-   The model is trained on the training data.
-   Finally, it predicts the target for the test data, and the results are evaluated with accuracy and classification reports.

**Output Example:**

```         
LazyClassifier: Evaluating Classifiers...
              Accuracy  Balanced Accuracy  ROC AUC  F1 Score  Time Taken
RandomForest   0.9667           0.9649    0.9993    0.9655    0.0283
XGBoost        0.9500           0.9474    0.9987    0.9473    0.0521
AdaBoost       0.9333           0.9283    0.9967    0.9282    0.0391
...
Selected Model: RandomForestClassifier
```

**Classification Report:**

```         
               precision    recall  f1-score   support

    Iris-setosa       1.00      1.00      1.00        10
 Iris-versicolor       0.92      1.00      0.96        11
  Iris-virginica       1.00      0.92      0.96         9

    accuracy                           0.97        30
   macro avg       0.97      0.97      0.97        30
weighted avg       0.97      0.97      0.97        30
```

## 🧪 Unit Tests

Unit tests are provided in `test_project.py` to ensure each core function works as expected. These tests include:

-   Testing the data preprocessing.
-   Ensuring that the best model is selected correctly.
-   Verifying that model training and prediction works without errors.

## 🧠 Want to Learn More?

Interested in expanding the project? Here are a few ideas:

-   🏋️‍♂️ **Expand the Dataset**: Try using a different dataset from sklearn or your own.
-   🧪 **Advanced Testing**: Add more complex unit tests or performance evaluations.
-   🛠 **Model Tuning**: Implement hyperparameter tuning for the selected models.

## 💻 Requirements

To run the project, you'll need:

-   Python 3.x
-   Required libraries listed in requirements.txt (including LazyPredict and scikit-learn).

You can install the dependencies using:

``` bash
pip install -r requirements.txt
```

## 🤝 Contributions

Feel free to open issues or submit pull requests if you have ideas for improving this project. Let's make this project even better, together!

E-mail: [as.alirezasaharkhiz\@gmail.com](mailto:as.alirezasaharkhiz@gmail.com){.email} telegram: @alirezasaharkhiz
