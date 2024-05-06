#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

# Load the training and test datasets
train_df = pd.read_csv('./train.csv')
test_df = pd.read_csv('./test.csv')

# Extract the 'Survived' column from the training dataset and 'PassengerId' from the test dataset
Y_train = train_df['Survived']
Y_test_PassengerId = test_df['PassengerId'] # Save for submission

# Select features for training and testing
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
train_df = train_df[features] 
test_df = test_df[features] 

# Combine the training and test datasets for preprocessing
combined = [train_df, test_df] 

# Preprocess the combined datasets
for df in combined:     
    # Fill missing values
    df['Age'].fillna(df['Age'].mean(), inplace=True)
    df['Fare'].fillna(df['Fare'].mean(), inplace=True)
    df['Embarked'].fillna(value='S', inplace=True)    
    
    # Convert categorical features to numeric
    df['Sex'] = df['Sex'].replace(['female','male'],[0,1]).astype(int)
    df['Embarked'] = df['Embarked'].replace(['S','Q','C'],[1,2,3]).astype(int)
    
    # Perform feature scaling
    df.loc[ df['Fare'] <= 7.91, 'Fare'] = 0
    df.loc[(df['Fare'] > 7.91) & (df['Fare'] <= 14.454), 'Fare'] = 1
    df.loc[(df['Fare'] > 14.454) & (df['Fare'] <= 31), 'Fare']   = 2
    df.loc[(df['Fare'] > 31) & (df['Fare'] <= 99), 'Fare']   = 3
    df.loc[(df['Fare'] > 99) & (df['Fare'] <= 250), 'Fare']   = 4
    df.loc[ df['Fare'] > 250, 'Fare'] = 5
    df['Fare'] = df['Fare'].astype(int)

# Assert that there are no NaN values in the datasets
assert not train_df.isnull().values.any()
assert not test_df.isnull().values.any()

# Convert datasets to numpy arrays and transpose them for compatibility with the logistic regression model
X_train = np.array(train_df).T
Y_train = np.array(Y_train)
Y_train = Y_train.reshape(Y_train.shape[0], 1).T
X_test = np.array(test_df).T

# Define the sigmoid function
def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s

# Initialize weights and bias
def initialize_with_zeros(dim):
    w = np.zeros(shape=(dim, 1))
    b = 0
    assert(w.shape == (dim, 1))
    assert(isinstance(b, float) or isinstance(b, int))
    return w, b

# Perform forward and backward propagation
def propagate(w, b, X, Y):
    m = X.shape[1]
    A = sigmoid(np.dot(w.T, X) + b)
    cost = (- 1 / m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))
    dw = (1 / m) * np.dot(X,(A - Y).T)
    db = (1 / m) * np.sum(A - Y)
    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    grads = {"dw": dw, "db": db}
    return grads, cost

# Optimize weights and bias using gradient descent
def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    costs = []
    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)
        dw = grads["dw"]
        db = grads["db"]
        w = w - learning_rate * dw 
        b = b - learning_rate * db
        if i % 100 == 0:
            costs.append(cost)
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
    params = {"w": w, "b": b}
    grads = {"dw": dw, "db": db}
    return params, grads, costs

# Predict the output based on learned parameters
def predict(w, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0], 1)
    A = sigmoid(np.dot(w.T, X) + b)
    for i in range(A.shape[1]):
        Y_prediction[0, i] = 1 if A[0, i] >= 0.5 else 0
    assert(Y_prediction.shape == (1, m))
    return Y_prediction

# Build the logistic regression model
def model(X_train, Y_train, X_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    w, b = initialize_with_zeros(X_train.shape[0])
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost=False)
    w = parameters["w"]
    b = parameters["b"]
    Y_prediction_train = predict(w, b, X_train)    
    Y_prediction_test = predict(w, b, X_test)
    print("Train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    d = {"costs": costs,         
         "Y_prediction_train" : Y_prediction_train, 
         "Y_prediction_test": Y_prediction_test,
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    return d

# Train the logistic regression model and plot the learning curve
d = model(X_train, Y_train, X_test, num_iterations=50000, learning_rate=0.004, print_cost=True)
costs = np.squeeze(d['costs'])
plt.plot(costs)
plt.ylabel('Logistic Regression cost (Negative Log Likelihood Loss)')
plt.xlabel('iterations (thou)')
plt.title("Learning rate =" + str(d["learning_rate"]))
plt.show()

d = model(X_train, Y_train, X_test, num_iterations=50000, learning_rate=0.04, print_cost=True)
costs = np.squeeze(d['costs'])
plt.plot(costs)
plt.ylabel('Logistic Regression cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(d["learning_rate"]))
plt.show()


# ## 
