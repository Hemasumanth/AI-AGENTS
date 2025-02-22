from transformers import ReactCodeAgent, HfApiEngine
from huggingface_hub import login
from transformers import tool


class UnsupervisedMLAutomation:
    def __init__(self):
        self.hf_token = ""
        self.model_name = "Qwen/Qwen2.5-Coder-32B-Instruct"
        self.task = """
        1. Load data from 'C:/Users/suman/OneDrive/Desktop/AI Agents/temp.csv'

        2. Perform automated unsupervised ML analysis:
            a. Auto-detect feature types (categorical/numerical)
            b. Handle preprocessing:
                - Encode categorical features (OneHotEncoding)
                - Scale numerical features (StandardScaler)
                - Handle missing values
                - Dimensionality reduction if needed (PCA, t-SNE)
            
            3. Try different unsupervised techniques:
                - Clustering (K-Means, DBSCAN, Hierarchical)
                - Anomaly detection (Isolation Forest, Local Outlier Factor)
                - Dimensionality reduction (PCA, UMAP, t-SNE)
                - Association rule learning (Apriori, FP-Growth)
            
            4. Evaluate results:
                a. For clustering: silhouette score, calinski-harabasz index
                b. For anomaly detection: precision@k, anomaly score distribution
                c. For dimensionality reduction: variance explained
                d. For association rules: support/confidence metrics
            
            5. Store all results in 
                'C:/Users/suman/OneDrive/Desktop/AI Agents/unsupervised_results.csv' with columns:
                - Technique
                - Parameters
                - Evaluation Metrics (JSON)
                - Timestamp
        """
        self.authorized_imports = [
            'pandas', 'openpyxl', 'numpy',
            'sklearn.cluster', 'sklearn.decomposition',
            'sklearn.manifold', 'sklearn.neighbors',
            'sklearn.metrics', 'mlxtend.frequent_patterns',
            'umap', 'seaborn', 'matplotlib',
            'joblib', 'datetime', 'json'
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
            max_iterations=25, 
            additional_authorized_imports=self.authorized_imports
        )

    def execute_task(self):
        generated_code = self.code_agent.run(self.task)
        print("Generated Unsupervised Code:\n", generated_code)
        return generated_code

if __name__ == "__main__":
    unsup_automation = UnsupervisedMLAutomation()
    unsup_automation.execute_task()







