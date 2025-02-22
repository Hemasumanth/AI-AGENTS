from transformers import ReactCodeAgent, HfApiEngine
from huggingface_hub import login

class ExcelDataProcessor:
    def __init__(self):
        self.hf_token = ""
        self.model_name = "Qwen/Qwen2.5-Coder-32B-Instruct"
        self.task = """
        1. Load data from 'C:/Users/suman/OneDrive/Desktop/AI Agents/synthetic_data.xlsx'
        2. Return cleaned data as CSV also count number of 0 and 1 in purchased.
        3. Store it as C:/Users/suman/OneDrive/Desktop/AI Agents/temp.csv
        4. Print status messages for each step (success or failure)
        """
        self.authorized_imports = ['pandas', 'openpyxl']
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
            max_iterations=20,
            additional_authorized_imports=self.authorized_imports
        )

    def execute_processing(self):
        generated_code = self.code_agent.run(self.task)
        print("Generated Code:\n", generated_code)
        # Uncomment to execute
        # exec(generated_code)
        return generated_code

if __name__ == "__main__":
    processor = ExcelDataProcessor()
    processor.execute_processing()