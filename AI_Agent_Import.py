from transformers import ReactCodeAgent, HfApiEngine

class ImportRecommenderAgent:
    def __init__(self, hf_token):
        self.engine = HfApiEngine(
            model="Qwen/Qwen2.5-Coder-32B-Instruct",
            token=hf_token
        )
    
    def get_required_imports(self, task_prompt):
        analysis_prompt = f"""Analyze this ML task and list ONLY required Python imports:
        {task_prompt}

        Consider these aspects:
        1. Data loading (pandas? openpyxl? numpy?)
        2. Model types (sklearn? xgboost? tensorflow?)
        3. Preprocessing (StandardScaler? OneHotEncoder?)
        4. Metrics (classification_report? mean_squared_error?)
        5. Visualization (matplotlib? seaborn?)
        6. Utilities (joblib? datetime?)

        Return ONLY a Python list of import strings, example:
        ['pandas', 'sklearn.preprocessing.StandardScaler', 'xgboost']
        """
        
        return eval(self.engine(analysis_prompt))

# Initialize agents
hf_token = ""
import_advisor = ImportRecommenderAgent(hf_token)
task = """
1. Load Excel data with mixed types
2. Build XGBoost classifier with feature scaling
3. Generate classification metrics and ROC curve
"""

required_imports = import_advisor.get_required_imports(task)
print("Detected required imports:", required_imports)

ml_agent = ReactCodeAgent(
    llm_engine=HfApiEngine(model="Qwen/Qwen2.5-Coder-32B-Instruct", token=hf_token),
    additional_authorized_imports=required_imports,
    add_base_tools=True,
    max_iterations=15
)

generated_code = ml_agent.run(task)