Iris Flower Classification
Domain

Artificial Intelligence and Machine Learning

Task Overview

This project implements a classification model to predict the species of an Iris flower based on its sepal and petal measurements. The task uses the classic Iris dataset and applies a supervised machine learning approach.

The objective is to classify flowers into one of the following species:

Setosa

Versicolor

Virginica

Dataset

The Iris dataset is sourced from the scikit-learn library, which is originally derived from the UCI Machine Learning Repository.

Features used:

Sepal length (cm)

Sepal width (cm)

Petal length (cm)

Petal width (cm)

Target variable:

Species (encoded as 0, 1, and 2)

Methodology

The following steps were followed to complete the task:

Data Loading
The Iris dataset was loaded using scikit-learn and converted into a Pandas DataFrame for easier manipulation.

Exploratory Data Analysis (EDA)
Basic visualizations such as histograms and scatter plots were used to understand feature distributions and observe patterns among different species.

Train–Test Split
The dataset was split into training and testing sets using an 80:20 ratio to evaluate the model on unseen data.

Model Training
A Logistic Regression classifier was trained on the training dataset.

Model Evaluation
The model performance was evaluated using:

Accuracy score

Confusion matrix

Classification report (precision, recall, F1-score)

Model Used

Logistic Regression

This model was chosen due to its simplicity and effectiveness for multi-class classification problems.

Results

The trained model achieved high accuracy on the test dataset, demonstrating effective classification of iris species based on the given features.

Typical performance metrics include:

Accuracy close to or equal to 1.0

Correct classification across all three species

Technologies Used

Python 3.9

Pandas

NumPy

Matplotlib

scikit-learn

Project Structure
Task-1-Iris-Flower-Classification/
│
├── iris_classification.py
└── README.md

How to Run the Project

Install required dependencies:

pip install numpy pandas matplotlib scikit-learn


Run the Python script:

python iris_classification.py

Skills Gained

Supervised machine learning

Classification modeling

Data visualization

Train–test splitting

Model evaluation using standard metrics

Conclusion

This project demonstrates the application of a supervised machine learning classification algorithm to a well-known dataset. It provides hands-on experience with data preprocessing, model training, and performance evaluation in an AI and ML context.