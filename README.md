📧 Spam Email Prediction using Machine Learning

This project implements a Spam Email Classification System using Natural Language Processing (NLP) and Machine Learning.
It classifies emails as Spam (1) or Not Spam (0) using the Bernoulli Naive Bayes algorithm.

🚀 Features

Loads and preprocesses email text data

Converts text into numerical features using CountVectorizer

Trains a Bernoulli Naive Bayes classifier

Evaluates model using:

Confusion Matrix

Accuracy Score

Classification Report

Predicts spam for new custom messages

🧠 Algorithm Used

Bernoulli Naive Bayes

Suitable for binary feature classification (presence/absence of words)

📂 Dataset

File: emails.csv

Required columns:

text → email content

spam → target label (1 = spam, 0 = not spam)

🛠️ Technologies Used

Python

Pandas

NumPy

Scikit-learn

📊 Model Workflow

Load dataset

Clean text (convert to lowercase)

Convert text to vectors using CountVectorizer

Split data into train and test sets

Train Bernoulli Naive Bayes model

Evaluate model performance

Predict spam for new messages

📈 Evaluation Metrics

Confusion Matrix

Accuracy Score

Precision

Recall

F1-score

These metrics help assess how well the model distinguishes spam from legitimate emails.
Matplotlib

Seaborn
