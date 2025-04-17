# Loan Default Risk Prediction Model

This repository contains a machine learning pipeline for predicting loan default risk. The project leverages various data preprocessing techniques, exploratory data analysis (EDA), feature engineering, and machine learning models to build a robust prediction system.

---

## Table of Contents
1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
4. [Data Preprocessing](#data-preprocessing)
5. [Feature Engineering](#feature-engineering)
6. [Model Training and Evaluation](#model-training-and-evaluation)
7. [Model Deployment](#model-deployment)
8. [How to Run](#how-to-run)

---

## Introduction

The goal of this project is to predict the likelihood of a loan default based on various borrower and loan attributes. The pipeline includes data visualization, feature selection, and the use of multiple machine learning models to achieve high prediction accuracy.

---

## Dataset

The dataset used in this project is stored in a CSV file named `train.csv`. It contains the following features:
- `credit_score`
- `income`
- `loan_amount`
- `loan_term`
- `interest_rate`
- `debt_to_income_ratio`
- `employment_years`
- `savings_balance`
- `age`
- `default_risk_score` (target variable)

---

## Exploratory Data Analysis (EDA)

1. **Data Overview**:
   - The dataset is loaded using `pandas` and inspected using `df.head()`, `df.info()`, and `df.describe()`.

2. **Missing Values**:
   - Missing values are identified using `df.isnull().sum()`.

3. **Correlation Analysis**:
   - A heatmap is generated using `seaborn` to visualize correlations between features.

4. **Feature Distributions**:
   - Histograms are plotted for each feature to understand their distributions.

5. **Outlier Detection**:
   - Boxplots are used to detect outliers in the dataset.

6. **Feature Relationships**:
   - Scatter plots are created to analyze relationships between features and the target variable.

---

## Data Preprocessing

1. **Scaling**:
   - Continuous variables such as `loan_term`, `interest_rate`, and `age` are scaled using `StandardScaler`.

2. **Log Transformation**:
   - Skewed features like `credit_score`, `loan_amount`, and `savings_balance` are log-transformed to reduce skewness.

3. **Variance Inflation Factor (VIF)**:
   - Multicollinearity is checked using VIF, and redundant features are identified.

---

## Feature Engineering

1. **Recursive Feature Elimination (RFE)**:
   - RFE is used to select the top 6 most important features for the model.

2. **Feature Selection**:
   - Selected features are used for training the models.

---

## Model Training and Evaluation

1. **Linear Regression**:
   - A baseline model is built using `LinearRegression`.

2. **Lasso Regression**:
   - Lasso regression is used for feature importance analysis.

3. **Ridge Regression**:
   - Ridge regression is tuned using `GridSearchCV` to find the best hyperparameters.

4. **Random Forest**:
   - A Random Forest model is trained for better performance.

5. **XGBoost**:
   - XGBoost is used as a high-performance model.

6. **Model Comparison**:
   - Models are evaluated using metrics such as R², Mean Squared Error (MSE), and Mean Absolute Error (MAE).

---

## Model Deployment

1. **Model Saving**:
   - The trained Random Forest model is saved using `joblib` for future use.

2. **Deployment**:
   - The saved model can be loaded and used for predictions in production.

---

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/loan-default-risk.git
   cd loan-default-risk

Collecting workspace information```markdown
# Loan Default Risk Prediction Model

This repository contains a machine learning pipeline for predicting loan default risk. The project leverages various data preprocessing techniques, exploratory data analysis (EDA), feature engineering, and machine learning models to build a robust prediction system.

---

## Table of Contents
1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
4. [Data Preprocessing](#data-preprocessing)
5. [Feature Engineering](#feature-engineering)
6. [Model Training and Evaluation](#model-training-and-evaluation)
7. [Model Deployment](#model-deployment)
8. [How to Run](#how-to-run)

---

## Introduction

The goal of this project is to predict the likelihood of a loan default based on various borrower and loan attributes. The pipeline includes data visualization, feature selection, and the use of multiple machine learning models to achieve high prediction accuracy.

---

## Dataset

The dataset used in this project is stored in a CSV file named `train.csv`. It contains the following features:
- `credit_score`
- `income`
- `loan_amount`
- `loan_term`
- `interest_rate`
- `debt_to_income_ratio`
- `employment_years`
- `savings_balance`
- `age`
- `default_risk_score` (target variable)

---

## Exploratory Data Analysis (EDA)

1. **Data Overview**:
   - The dataset is loaded using `pandas` and inspected using `df.head()`, `df.info()`, and `df.describe()`.

2. **Missing Values**:
   - Missing values are identified using `df.isnull().sum()`.

3. **Correlation Analysis**:
   - A heatmap is generated using `seaborn` to visualize correlations between features.

4. **Feature Distributions**:
   - Histograms are plotted for each feature to understand their distributions.

5. **Outlier Detection**:
   - Boxplots are used to detect outliers in the dataset.

6. **Feature Relationships**:
   - Scatter plots are created to analyze relationships between features and the target variable.

---

## Data Preprocessing

1. **Scaling**:
   - Continuous variables such as `loan_term`, `interest_rate`, and `age` are scaled using `StandardScaler`.

2. **Log Transformation**:
   - Skewed features like `credit_score`, `loan_amount`, and `savings_balance` are log-transformed to reduce skewness.

3. **Variance Inflation Factor (VIF)**:
   - Multicollinearity is checked using VIF, and redundant features are identified.

---

## Feature Engineering

1. **Recursive Feature Elimination (RFE)**:
   - RFE is used to select the top 6 most important features for the model.

2. **Feature Selection**:
   - Selected features are used for training the models.

---

## Model Training and Evaluation

1. **Linear Regression**:
   - A baseline model is built using `LinearRegression`.

2. **Lasso Regression**:
   - Lasso regression is used for feature importance analysis.

3. **Ridge Regression**:
   - Ridge regression is tuned using `GridSearchCV` to find the best hyperparameters.

4. **Random Forest**:
   - A Random Forest model is trained for better performance.

5. **XGBoost**:
   - XGBoost is used as a high-performance model.

6. **Model Comparison**:
   - Models are evaluated using metrics such as R², Mean Squared Error (MSE), and Mean Absolute Error (MAE).

---

## Model Deployment

1. **Model Saving**:
   - The trained Random Forest model is saved using `joblib` for future use.

2. **Deployment**:
   - The saved model can be loaded and used for predictions in production.

---

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/loan-default-risk.git
   cd loan-default-risk
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Jupyter Notebook:
   ```bash
   jupyter notebook ML.ipynb
   ```

4. Follow the steps in the notebook to preprocess the data, train models, and evaluate performance.

---

## Conclusion

This project demonstrates a complete pipeline for predicting loan default risk using machine learning. The combination of EDA, feature engineering, and model evaluation ensures a robust and interpretable solution.

Feel free to contribute to this project by submitting issues or pull requests!
```
