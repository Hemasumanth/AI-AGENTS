from transformers import ReactCodeAgent, HfApiEngine
from huggingface_hub import login

class MetricsVisualizer:
    def __init__(self):
        self.hf_token = ""
        self.model_name = "Qwen/Qwen2.5-Coder-32B-Instruct"
        self.workflow_file = "C:/Users/suman/OneDrive/Desktop/AI Agents/workflow_metrics.json"
        self.confusion_file = "C:/Users/suman/OneDrive/Desktop/AI Agents/confusion.csv"
        self.task = f"""
        
        1. Use 'read_file_as_string' tool to get JSON string from '{self.workflow_file}', then parse with json.loads()

        2. Load CSV data from '{self.confusion_file}' using pandas.read_csv():
           - Try to read it into a DataFrame
           - If reading fails (file missing or invalid CSV), set data to an empty DataFrame and print a warning
        3. Create visualizations with error handling:
           - For workflow_metrics.json (if data is not empty):
             * Bar plot of total_duration for each phase:
               - Extract durations from 'phases' key, convert to minutes (assume timedelta strings like '0:00:00')
               - Handle missing 'phases' or 'total_duration' keys by skipping
               - Skip if no valid duration data
             * Pie chart of success/failure status for each phase:
               - Extract statuses from 'phases', count success vs failed
               - Skip if no status data
             * Bar plot of attempts vs failed_attempts:
               - Extract from 'phases', compare totals
               - Skip if no attempt data
           - For confusion.csv (if DataFrame is not empty):
             * Verify 'Metric Type', 'Model Name', and 'Metric Values' columns exist
             * For each unique value in 'Metric Type' (if column exists):
               - Create a bar plot comparing that metric across models
               - Use 'Model Name' for x-axis, 'Metric Values' for y-axis
               - Skip if data insufficient or columns missing
        4. Save all plots in 'C:/Users/suman/OneDrive/Desktop/AI Agents/':
           - Name workflow plots as 'workflow_[plot_type].png' (e.g., 'workflow_durations.png')
           - Name metric plots as 'metric_[metric_name].png' (lowercase, replace spaces with '_')
           - If saving fails, print a warning and continue
        5. Print status messages for each step (e.g., 'Loaded JSON', 'Failed to plot durations', etc.)
        """
        self.authorized_imports = ['io','pandas', 'matplotlib', 'matplotlib.pyplot', 'json','plotly','os','openpyxl']
        self.initialize_environment()

    def initialize_environment(self):
        """Initialize the Hugging Face environment and code agent"""
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

    def execute_visualization(self):
        """Generate and return visualization code"""
        generated_code = self.code_agent.run(self.task)
        print("Generated Visualization Code:\n", generated_code)
        return generated_code

if __name__ == "__main__":
    visualizer = MetricsVisualizer()
    visualizer.execute_visualization()