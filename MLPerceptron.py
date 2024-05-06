#!/usr/bin/env python
# coding: utf-8

# In[12]:


import numpy as np
import pandas as pd

class Perceptron:
    def __init__(self, learning_rate=0.01, epochs=100):
        # Initialize the perceptron model with learning rate and number of epochs
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # Train the perceptron model using the input data X and labels y
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Iterate through epochs
        for _ in range(self.epochs):
            # Iterate through each sample in the training data
            for i in range(n_samples):
                # Compute the activation using the current weights and bias
                activation = np.dot(X[i], self.weights) + self.bias
                # Predict the label based on the current weights and bias
                y_pred = self.predict(X[i])
                # Update the weights and bias using the perceptron update rule
                self.weights += self.learning_rate * (y[i] - y_pred) * X[i]
                self.bias += self.learning_rate * (y[i] - y_pred)

    def predict(self, X):
        # Make predictions for the input data X
        return np.where(np.dot(X, self.weights) + self.bias >= 0, 1, 0)

# Preprocessing the data
def preprocess_data(df):
    # Convert categorical features to numerical values
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    # Fill missing values in 'Age' with median
    df['Age'].fillna(df['Age'].median(), inplace=True)
    # Fill missing values in 'Embarked' with mode
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    # Map 'Embarked' values to numerical values
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    # Select relevant features and standardize numerical features
    X = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
    X = (X - X.mean()) / X.std()  # Standardize features
    return X.values

# Load the data
train_df = pd.read_csv('./train.csv')
test_df = pd.read_csv('./test.csv')

# Preprocess the data
X_train = preprocess_data(train_df)
y_train = train_df['Survived'].values

X_test = preprocess_data(test_df)

# Train the Perceptron model
perceptron = Perceptron(learning_rate=0.04, epochs=100)
perceptron.fit(X_train, y_train)

# Training accuracy
y_train_pred = perceptron.predict(X_train)
training_accuracy = np.mean(y_train_pred == y_train)
print("Training Accuracy:", training_accuracy)

# Predictions for all passengers in the test dataset
test_predictions = perceptron.predict(X_test)
print("Perceptron Test Predictions: \n", test_predictions)


# In[ ]:




