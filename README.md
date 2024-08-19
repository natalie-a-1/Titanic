# Logistic Regression and Perceptron Models

## Project Overview

This project involves implementing and comparing two machine learning algorithms, Logistic Regression and Perceptron, to predict the survival of passengers on the Titanic. The models are trained and tested using a dataset derived from the Titanic dataset available on Kaggle. The goal is to build predictive models to determine whether a passenger survived based on features such as class, gender, age, number of siblings/spouses aboard, number of parents/children aboard, fare, and port of embarkation.

## Project Structure

The project is divided into two main sections:
1. **Logistic Regression Model**
2. **Perceptron Model**

### 1. Logistic Regression Model

#### File
- `MLLogisticRegression.py`: Contains the implementation of the Logistic Regression model.

#### Steps
- **Data Loading**: The training and test datasets are loaded from CSV files.
- **Feature Selection**: Features such as `Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, and `Embarked` are selected for training and testing.
- **Data Preprocessing**: Missing values in `Age`, `Fare`, and `Embarked` are filled, categorical features (`Sex`, `Embarked`) are converted to numeric, and feature scaling is applied to `Fare`.
- **Model Implementation**: 
  - Sigmoid function is defined to apply the logistic function.
  - Weights and bias are initialized.
  - Forward and backward propagation are implemented to calculate the gradients and update weights using gradient descent.
  - The model is trained using the training dataset, and predictions are made on both the training and test datasets.
- **Model Evaluation**: The training accuracy is calculated, and the cost function is plotted to visualize the learning curve.
  
#### Results
- The Logistic Regression model was trained for 50,000 iterations with two different learning rates (0.004 and 0.04).
- The learning curves for different learning rates were plotted to observe the cost reduction over iterations.

### 2. Perceptron Model

#### File
- `MLPerceptron.py`: Contains the implementation of the Perceptron model.

#### Steps
- **Data Loading**: The same training and test datasets used in the Logistic Regression model are loaded.
- **Data Preprocessing**: 
  - Categorical features (`Sex`, `Embarked`) are converted to numerical values.
  - Missing values in `Age` and `Embarked` are filled with the median and mode, respectively.
  - Selected features are standardized.
- **Model Implementation**: 
  - A Perceptron class is defined, implementing the perceptron algorithm.
  - The model is trained using the training dataset over a specified number of epochs.
  - The model predicts the labels for the test dataset.
- **Model Evaluation**: Training accuracy is calculated, and predictions for the test dataset are output.

#### Results
- The Perceptron model was trained with a learning rate of 0.04 over 100 epochs.
- The training accuracy was calculated and displayed.

## How to Run

1. Ensure that the following dependencies are installed:
   - `numpy`
   - `pandas`
   - `matplotlib`

2. Place the `train.csv` and `test.csv` files in the same directory as the Python scripts.

3. To run the Logistic Regression model:
   ```bash
   python MLLogisticRegression.py
   ```
   This will train the model, display the training accuracy, and plot the learning curves.

4. To run the Perceptron model:
   ```bash
   python MLPerceptron.py
   ```
   This will train the model, display the training accuracy, and output predictions for the test dataset.

## Conclusion

This project demonstrates the implementation of two fundamental machine learning algorithms from scratch. The Logistic Regression model showed the ability to converge over time with a reduction in cost, while the Perceptron model provided a straightforward approach to binary classification. By comparing these models, we gain insights into the effectiveness of different algorithms for classification tasks.

## Contributors

- **[Natalie Hill]**
- **[Keon Moradi]**

---

This README file should now accurately reflect the correct file names for your project.
