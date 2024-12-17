# Comprehensive Churn Prediction Project

## Overview
This project is a comprehensive machine learning solution for predicting customer churn in a banking context. The code generates a synthetic dataset of bank customers, performs exploratory data analysis, and evaluates different machine learning models to identify the best predictor for customer churn.

## Features
- **Synthetic Data Generation**: Automatically creates a dataset of customer information including age, account balance, credit score, and churn status.
- **Feature Engineering**: Adds advanced features like balance-to-income ratio, transaction frequency, and customer value classification.
- **Exploratory Data Analysis (EDA)**: Includes visualizations such as churn distribution, churn by age group, feature correlation heatmaps, and account balance analysis.
- **Model Comparison**: Compares Random Forest, Logistic Regression, and Decision Tree models.
- **Hyperparameter Tuning**: Optimizes the Random Forest model using grid search.
- **Final Evaluation**: Evaluates the best model with metrics such as accuracy, classification report, ROC AUC score, and feature importance visualization.

## Requirements
The following Python libraries are required:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

Install them using the command:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## How It Works

### 1. Synthetic Data Generation
The `BankChurnPredictor` class generates a dataset with synthetic customer information, including:
- Age
- Account balance
- Credit score
- Years as a customer
- Number of transactions
- Total products
- Churn status

### 2. Feature Engineering
The data is enhanced with additional features:
- **Balance-to-Income Ratio**: Ratio of account balance to income.
- **Transaction Frequency**: Transactions per year as a customer.
- **High-Value Customer**: Identifies customers with above-median account balances.
- **Age Groups**: Categorizes customers into age brackets.

### 3. Exploratory Data Analysis (EDA)
EDA provides insights into:
- Churn distribution.
- Correlations between features.
- Patterns in customer demographics.

### 4. Model Training and Comparison
The project compares three models:
- Random Forest
- Logistic Regression
- Decision Tree

Each model is evaluated using cross-validation, and results are printed with mean accuracy and standard deviation.

### 5. Hyperparameter Tuning
Random Forest is tuned using grid search with parameters such as:
- Number of estimators.
- Maximum depth.
- Minimum samples split and leaf.

### 6. Final Model Evaluation
The best model is evaluated with:
- Accuracy and classification report.
- ROC AUC score.
- Feature importance visualization.

## Project Execution
To execute the project, run the following command:
```bash
python churn_prediction.py
```
The script:
1. Performs EDA.
2. Compares models.
3. Tunes the best model.
4. Evaluates the final model.

## Example Outputs
- **Visualizations**: Churn distribution, feature correlations, and feature importance.
- **Metrics**: Accuracy, ROC AUC score, and detailed classification reports.

## Customization
You can customize the number of customers in the synthetic dataset by adjusting the `num_customers` parameter when initializing the `BankChurnPredictor` class:
```python
churn_predictor = BankChurnPredictor(num_customers=10000)
```

## Contributions
Feel free to fork the project and submit pull requests for improvements or additional features.

## License
This project is open-source and available under the MIT License.

---

For questions or feedback, contact [your_email@example.com].

