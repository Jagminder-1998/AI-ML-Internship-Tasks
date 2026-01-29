Task 2 (Spam Mail Detector)

Create / open this file:

Task-2-Spam-Mail-Detector/README.md

Paste the following professional README (no emojis, submission-ready):
Spam Mail Detector
Domain

Artificial Intelligence and Machine Learning

Task Overview

This project implements a Spam Mail Detection system using machine learning techniques.
The objective is to classify text messages as spam or ham (non-spam) based on their content.

This task demonstrates the application of Natural Language Processing (NLP) and supervised learning for text classification.

Dataset

The SMS Spam Collection Dataset is used for this project.
It is a publicly available dataset containing labeled SMS messages.

Each data instance includes:

Message text

Label (spam or ham)

Methodology

The following steps were performed:

Data Loading
The dataset was loaded using Pandas and inspected to understand its structure and size.

Text Preprocessing

Converted text to lowercase

Removed unnecessary words using stopword filtering

Prepared text for numerical conversion

Feature Extraction
Text data was transformed into numerical features using TF-IDF Vectorization.

Train–Test Split
The dataset was split into training and testing sets using an 80:20 ratio.

Model Training
A Naive Bayes classifier was trained on the vectorized text data.

Model Evaluation
Model performance was evaluated using:

Accuracy score

Precision

Recall

F1-score

Model Used

Naive Bayes Classifier

Naive Bayes is well-suited for text classification problems and performs efficiently for spam detection.

Technologies Used

Python 3.9

Pandas

scikit-learn

Natural Language Processing techniques

Project Structure
Task-2-Spam-Mail-Detector/
│
├── spam_detector.py
└── README.md

How to Run the Project

Install required libraries:

python -m pip install pandas scikit-learn


Run the script:

python spam_detector.py

Skills Gained

Text preprocessing

Feature extraction using TF-IDF

Spam classification

Natural Language Processing fundamentals

Model evaluation

Conclusion

This project demonstrates how machine learning and NLP techniques can be applied to detect spam messages effectively. The trained model achieves high accuracy and shows reliable performance on unseen data.