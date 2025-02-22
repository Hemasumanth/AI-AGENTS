from transformers import ReactCodeAgent, HfApiEngine
from huggingface_hub import login
import pandas as pd

class MLTaskAutomation:
    def __init__(self):
        self.hf_token = ""
        self.model_name = "Qwen/Qwen2.5-Coder-32B-Instruct"
        self.task = """
        1. Load data from 'C:/Users/suman/OneDrive/Desktop/AI Agents/temp.csv'

        2. Perform automated supervised ML modeling:
            a. Check if target column is categorical:
                - If YES: Classification (try Logistic Regression, Random Forest, XGBoost)
                - If NO: Regression (try Linear Regression, Decision Tree Regressor, XGBoost Regressor)
            b. Auto-detect feature types (categorical/numerical)
            c. Handle preprocessing:
                - Encode categorical features (OneHotEncoding)
                - Scale numerical features (StandardScaler)
                - Handle missing values

        3. For classification models:
            a. Calculate confusion matrix
            b. Generate classification report (precision/recall/f1-score) as a single JSON object containing all metrics
            For regression models:
            a. Calculate RMSE, R-squared
            
        4. Store all evaluation metrics and confusion matrices in 
            'C:/Users/suman/OneDrive/Desktop/AI Agents/confusion.csv' with columns:
            - Model Name
            - Metric Type
            - Metric Values (JSON format)
            - Timestamp
            With only these Metric Type values:
            - For classification: 'confusion_matrix' and 'classification_report'
            - For regression: 'rmse' and 'r_squared'
            Where 'classification_report' contains a single JSON with all metrics (per class, macro avg, and weighted avg)

        5. Try all the given models

        6. Print status messages for each step (success or failure)
        """
        self.authorized_imports = [
            'pandas', 'openpyxl', 'sklearn.model_selection',
            'sklearn.preprocessing', 'sklearn.compose', 'sklearn.impute',
            'sklearn.linear_model', 'sklearn.ensemble', 'sklearn.tree',
            'sklearn.metrics', 'xgboost', 'joblib', 'datetime',
            'json', 'warnings', 'numpy','sklearn.pipeline'
        ]
        self.initialize_environment()

    def initialize_environment(self):
        login(token=self.hf_token)
        self.llm_engine = HfApiEngine(
            model=self.model_name,
            token=self.hf_token
        )
        self.code_agent = ReactCodeAgent(
            llm_engine=self.llm_engine,
            tools=[],
            add_base_tools=True,
            max_iterations=50,
            additional_authorized_imports=self.authorized_imports
        )

    def execute_task(self):
        generated_code = self.code_agent.run(self.task)
        print("Generated Code:\n", generated_code)
        return generated_code

if __name__ == "__main__":
    automation = MLTaskAutomation()
    automation.execute_task()