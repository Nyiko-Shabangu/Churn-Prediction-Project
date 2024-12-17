# Comprehensive Churn Prediction Project

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning Libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    classification_report, 
    accuracy_score, 
    confusion_matrix, 
    roc_auc_score
)

class BankChurnPredictor:
    def __init__(self, num_customers=2000):
        """
        Initialize the Churn Prediction Project
        
        Parameters:
        - num_customers: Number of synthetic customers to generate
        """
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Generate synthetic bank customer data
        self.bank_data = self.generate_bank_data(num_customers)
        
        # Prepare features and target
        self.prepare_data()
    
    def generate_bank_data(self, num_customers):
        """
        Create synthetic bank customer dataset
        
        Returns:
        - pandas DataFrame with customer information
        """
        data = {
            'customer_id': range(1, num_customers + 1),
            'age': np.random.randint(18, 70, num_customers),
            'account_balance': np.random.normal(50000, 20000, num_customers),
            'credit_score': np.random.normal(650, 100, num_customers),
            'years_as_customer': np.random.randint(0, 10, num_customers),
            'num_transactions': np.random.randint(0, 100, num_customers),
            'total_products': np.random.randint(1, 5, num_customers),
            'is_active': np.random.choice([0, 1], num_customers, p=[0.7, 0.3]),
            'churn': np.random.choice([0, 1], num_customers, p=[0.8, 0.2])
        }
        
        return pd.DataFrame(data)
    
    def feature_engineering(self):
        """
        Create advanced features for better prediction
        """
        df = self.bank_data.copy()
        
        # Derive new features
        df['balance_to_income_ratio'] = df['account_balance'] / (df['age'] * 1000)
        df['transaction_frequency'] = df['num_transactions'] / (df['years_as_customer'] + 1)
        df['is_high_value_customer'] = (df['account_balance'] > df['account_balance'].median()).astype(int)
        
        # Categorize age groups
        df['age_group'] = pd.cut(
            df['age'], 
            bins=[0, 25, 35, 45, 55, 100],
            labels=['18-25', '26-35', '36-45', '46-55', '55+']
        )
        
        return df
    
    def prepare_data(self):
        """
        Prepare data for machine learning
        """
        # Apply feature engineering
        self.engineered_data = self.feature_engineering()
        
        # Select features
        features = [
            'age', 'account_balance', 'credit_score', 
            'years_as_customer', 'num_transactions', 
            'total_products', 'is_active',
            'balance_to_income_ratio', 
            'transaction_frequency',
            'is_high_value_customer'
        ]
        
        # Prepare features and target
        X = self.engineered_data[features]
        y = self.engineered_data['churn']
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
    
    def exploratory_data_analysis(self):
        """
        Perform comprehensive data exploration and visualization
        """
        plt.figure(figsize=(16, 12))
        
        # Churn Distribution
        plt.subplot(2, 2, 1)
        self.bank_data['churn'].value_counts(normalize=True).plot(kind='pie', autopct='%1.1f%%')
        plt.title('Churn Distribution')
        
        # Churn by Age Group
        plt.subplot(2, 2, 2)
        churn_by_age = self.engineered_data.groupby('age_group')['churn'].mean()
        churn_by_age.plot(kind='bar')
        plt.title('Churn Rate by Age Group')
        plt.ylabel('Churn Probability')
        plt.xticks(rotation=45)
        
        # Correlation Heatmap
        plt.subplot(2, 2, 3)
        correlation_matrix = self.engineered_data.select_dtypes(include=[np.number]).corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Feature Correlation Heatmap')
        
        # Account Balance vs Churn
        plt.subplot(2, 2, 4)
        self.engineered_data.boxplot(column='account_balance', by='churn')
        plt.title('Account Balance Distribution by Churn')
        plt.suptitle('')
        
        plt.tight_layout()
        plt.show()
    
    def model_comparison(self):
        """
        Compare multiple machine learning models
        """
        models = {
            'Random Forest': RandomForestClassifier(random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42),
            'Decision Tree': DecisionTreeClassifier(random_state=42)
        }
        
        # Compare models using cross-validation
        results = {}
        for name, model in models.items():
            cv_scores = cross_val_score(model, self.X_train_scaled, self.y_train, cv=5)
            results[name] = {
                'mean_accuracy': cv_scores.mean(),
                'std_accuracy': cv_scores.std()
            }
        
        # Print results
        print("Model Comparison:")
        for name, metrics in results.items():
            print(f"{name}:")
            print(f"  Mean Accuracy: {metrics['mean_accuracy']:.4f}")
            print(f"  Standard Deviation: {metrics['std_accuracy']:.4f}")
        
        return models
    
    def tune_random_forest(self):
        """
        Perform hyperparameter tuning for Random Forest
        """
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(
            estimator=rf, 
            param_grid=param_grid, 
            cv=5, 
            n_jobs=-1, 
            verbose=0
        )
        
        grid_search.fit(self.X_train_scaled, self.y_train)
        
        print("\nBest Random Forest Parameters:")
        print(grid_search.best_params_)
        
        return grid_search.best_estimator_
    
    def evaluate_model(self, model):
        """
        Evaluate the final model's performance
        """
        # Predictions
        y_pred = model.predict(self.X_test_scaled)
        y_prob = model.predict_proba(self.X_test_scaled)[:, 1]
        
        # Performance Metrics
        print("\nModel Performance:")
        print("Accuracy Score:", accuracy_score(self.y_test, y_pred))
        print("\nDetailed Classification Report:")
        print(classification_report(self.y_test, y_pred))
        
        # ROC AUC Score
        print("\nROC AUC Score:", roc_auc_score(self.y_test, y_prob))
        
        # Feature Importance
        feature_names = self.X_train.columns
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=feature_importance)
        plt.title('Feature Importance for Churn Prediction')
        plt.tight_layout()
        plt.show()
        
        return feature_importance
    
    def run_project(self):
        """
        Execute the entire churn prediction project
        """
        print("Bank Customer Churn Prediction Project\n")
        
        # 1. Exploratory Data Analysis
        print("1. Exploratory Data Analysis")
        self.exploratory_data_analysis()
        
        # 2. Model Comparison
        print("\n2. Model Comparison")
        self.model_comparison()
        
        # 3. Hyperparameter Tuning
        print("\n3. Hyperparameter Tuning")
        best_model = self.tune_random_forest()
        
        # 4. Final Model Evaluation
        print("\n4. Final Model Evaluation")
        feature_importance = self.evaluate_model(best_model)
        
        return best_model, feature_importance

# Main Execution
def main():
    # Create and run the churn prediction project
    churn_predictor = BankChurnPredictor(num_customers=5000)
    best_model, feature_importance = churn_predictor.run_project()

if __name__ == "__main__":
    main()