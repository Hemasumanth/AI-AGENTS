import os
import time
from datetime import datetime
import pandas as pd
import json
from dash import Dash, dcc, html, Input, Output
import plotly.express as px
from AI_Agent import ExcelDataProcessor
from AI_agent_ml import MLTaskAutomation

class MLWorkflowCoordinator:
    def __init__(self):
        self.metrics = {
            'total_start_time': None,
            'total_duration': None,
            'cleaning': {
                'attempts': 0,
                'failures': 0,
                'success': False,
                'durations': [],
                'file_check_attempts': 0,
                'duration': None
            },
            'ml': {
                'attempts': 0,
                'failures': 0,
                'success': False,
                'durations': [],
                'file_check_attempts': 0,
                'duration': None
            },
            'visualizations': []  
        }
        self.hf_token = ""
        self.clean_data_path = "C:/Users/suman/OneDrive/Desktop/AI Agents/temp.csv"
        self.results_path = "C:/Users/suman/OneDrive/Desktop/AI Agents/confusion.csv"
        self.workflow_file = "C:/Users/suman/OneDrive/Desktop/AI Agents/workflow_metrics.json"
        
        # Retry configuration
        self.max_cleaning_attempts = 3
        self.max_ml_attempts = 3
        self.file_check_retries = 5
        self.file_check_delay = 2
        self.step_retry_delay = 10
        
        # Initialize components
        self.data_processor = ExcelDataProcessor()
        self.ml_automation = MLTaskAutomation()

    def run_full_workflow(self):
        self.metrics['total_start_time'] = datetime.now()
        try:
            if not self._run_cleaning_phase():
                return False
            if not self._run_ml_phase():
                return False
            
            print("\nWorkflow completed successfully!")
            return True
            
        except Exception as e:
            print(f"\nCritical workflow failure: {str(e)}")
            return False
        finally:
            self.metrics['total_duration'] = datetime.now() - self.metrics['total_start_time']
            self._save_metrics_report()
            self._generate_dashboard()

    def _run_cleaning_phase(self):
        phase_metrics = self.metrics['cleaning']
        start_time = datetime.now()
        
        for attempt in range(1, self.max_cleaning_attempts + 1):
            print(f"\nAttempt {attempt}/{self.max_cleaning_attempts}")
            phase_metrics['attempts'] += 1
            try:
                self.data_processor.execute_processing()
                if self._check_file_exists(self.clean_data_path):
                    print("Data cleaning successful")
                    phase_metrics['success'] = True
                    phase_metrics['durations'].append(datetime.now() - start_time)
                    phase_metrics['duration'] = datetime.now() - start_time
                    return True
                
                print(f"Cleaning attempt {attempt} failed")
                phase_metrics['failures'] += 1
                if attempt < self.max_cleaning_attempts:
                    self._countdown(self.step_retry_delay)
                    
            except Exception as e:
                print(f"Error during cleaning attempt {attempt}: {str(e)}")
                phase_metrics['failures'] += 1
        
        phase_metrics['duration'] = datetime.now() - start_time
        return False

    def _run_ml_phase(self):
        phase_metrics = self.metrics['ml']
        start_time = datetime.now()
        
        for attempt in range(1, self.max_ml_attempts + 1):
            print(f"\nAttempt {attempt}/{self.max_ml_attempts}")
            phase_metrics['attempts'] += 1
            try:
                self.ml_automation.execute_task()
                if self._check_file_exists(self.results_path):
                    print("ML modeling successful")
                    phase_metrics['success'] = True
                    phase_metrics['durations'].append(datetime.now() - start_time)
                    phase_metrics['duration'] = datetime.now() - start_time
                    return True
                
                print(f"ML attempt {attempt} failed")
                phase_metrics['failures'] += 1
                if attempt < self.max_ml_attempts:
                    self._countdown(self.step_retry_delay)
                    
            except Exception as e:
                print(f"Error during ML attempt {attempt}: {str(e)}")
                phase_metrics['failures'] += 1
                
        print(f"\nFailed all {self.max_ml_attempts} ML attempts")
        phase_metrics['duration'] = datetime.now() - start_time
        return False

    def _check_file_exists(self, path):
        check_attempts = 0
        for i in range(1, self.file_check_retries + 1):
            check_attempts += 1
            if os.path.exists(path):
                return True
            print(f"Waiting for {os.path.basename(path)}... ({i}/{self.file_check_retries})")
            time.sleep(self.file_check_delay)
        
        if path == self.clean_data_path:
            self.metrics['cleaning']['file_check_attempts'] += check_attempts
        else:
            self.metrics['ml']['file_check_attempts'] += check_attempts
        return False

    def _countdown(self, seconds):
        start = time.time()
        for remaining in range(seconds, 0, -1):
            print(f"Retrying in {remaining}...", end="\r")
            time.sleep(1)
        print(" " * 30, end="\r")
        return time.time() - start

    def _generate_dashboard(self):
        """Generate a Dash dashboard for workflow_metrics.json"""
        print("\nGenerating dashboard for workflow_metrics.json...")
        app = Dash(__name__)

        try:
            # Load the JSON file
            with open(self.workflow_file, 'r') as f:
                workflow_data = json.load(f)

            phases = workflow_data.get('phases', {})
            if not phases:
                print("Warning: No 'phases' data found in workflow_metrics.json")
                return

            phases_df = pd.DataFrame(phases).T
            phases_df['duration_minutes'] = phases_df['total_duration'].apply(
                lambda x: pd.to_timedelta(x).total_seconds() / 60 if x else 0
            )

            # Layout of the dashboard
            app.layout = html.Div([
                html.H1("ML Workflow Metrics Dashboard"),
                
                # Total Duration Bar Chart
                dcc.Graph(
                    id='duration-bar',
                    figure=px.bar(
                        phases_df,
                        x=phases_df.index,
                        y='duration_minutes',
                        title='Phase Total Durations',
                        labels={'duration_minutes': 'Duration (Minutes)', 'index': 'Phase'}
                    )
                ),
                
                # Status Pie Chart
                dcc.Graph(
                    id='status-pie',
                    figure=px.pie(
                        phases_df,
                        names='status',
                        title='Phase Status Distribution'
                    )
                ),
                
                # Attempts vs Failed Bar Chart
                dcc.Graph(
                    id='attempts-bar',
                    figure=px.bar(
                        phases_df,
                        x=phases_df.index,
                        y=['total_attempts', 'failed_attempts'],
                        barmode='stack',  
                        title='Total and Failed Attempts per Phase', 
                        labels={'value': 'Number of Attempts', 'index': 'Phase', 'variable': 'Attempt Type'},  # Intuitive labels
                        color_discrete_map={'total_attempts': '#2ecc71', 'failed_attempts': '#e74c3c'},  # Green for total, Red for failed
                        text_auto=True  
                    ).update_traces(
                        textposition='inside',  
                        textfont_size=14  
                    ).update_layout(
                        legend_title_text='Attempt Type', 
                        bargap=0.2  
                    )
                ),
                
                # Metrics Table
                html.H3("Detailed Metrics"),
                html.Table([
                    html.Tr([html.Th(col) for col in phases_df.columns])] + [
                    html.Tr([html.Td(phases_df.iloc[i][col]) for col in phases_df.columns])
                    for i in range(len(phases_df))
                ])
            ])

            # Store dashboard indicator (URL will be printed when run)
            self.metrics['visualizations'] = ["Dashboard running at http://127.0.0.1:8050/"]
            print("Dashboard generated. Access it at http://127.0.0.1:8050/")

            # Run the Dash app
            app.run_server(debug=True, use_reloader=False)

        except Exception as e:
            print(f"Failed to generate dashboard: {str(e)}")
            self.metrics['visualizations'] = []

    def _save_metrics_report(self):
        report = {
            'total_duration': str(self.metrics['total_duration']),
            'phases': {
                'data_cleaning': {
                    'status': 'success' if self.metrics['cleaning']['success'] else 'failed',
                    'total_attempts': self.metrics['cleaning']['attempts'],
                    'failed_attempts': self.metrics['cleaning']['failures'],
                    'time_per_attempt': [str(d) for d in self.metrics['cleaning']['durations']],
                    'total_duration': str(self.metrics['cleaning']['duration'] or "0:00:00"),
                    'file_check_attempts': self.metrics['cleaning']['file_check_attempts']
                },
                'ml_modeling': {
                    'status': 'success' if self.metrics['ml']['success'] else 'failed',
                    'total_attempts': self.metrics['ml']['attempts'],
                    'failed_attempts': self.metrics['ml']['failures'],
                    'time_per_attempt': [str(d) for d in self.metrics['ml']['durations']],
                    'total_duration': str(self.metrics['ml']['duration'] or "0:00:00"),
                    'file_check_attempts': self.metrics['ml']['file_check_attempts']
                }
            },
            'visualizations': self.metrics['visualizations'],
            'timestamp': str(datetime.now())
        }

        print("\n=== Performance Metrics Report ===")
        print(f"Total workflow duration: {report['total_duration']}")
        print("\nData Cleaning Phase:")
        print(f"  Status: {report['phases']['data_cleaning']['status']}")
        print(f"  Attempts: {report['phases']['data_cleaning']['total_attempts']}")
        print(f"  Failures: {report['phases']['data_cleaning']['failed_attempts']}")
        print(f"  File checks: {report['phases']['data_cleaning']['file_check_attempts']}")
        print(f"  Total duration: {report['phases']['data_cleaning']['total_duration']}")

        print("\nML Modeling Phase:")
        print(f"  Status: {report['phases']['ml_modeling']['status']}")
        print(f"  Attempts: {report['phases']['ml_modeling']['total_attempts']}")
        print(f"  Failures: {report['phases']['ml_modeling']['failed_attempts']}")
        print(f"  File checks: {report['phases']['ml_modeling']['file_check_attempts']}")
        print(f"  Total duration: {report['phases']['ml_modeling']['total_duration']}")

        print("\nVisualizations:")
        for viz in report['visualizations']:
            print(f"  {viz}")

        with open(self.workflow_file, 'w') as f:
            json.dump(report, f, indent=2)

        return report

def main():
    output_dir = "C:/Users/suman/OneDrive/Desktop/AI Agents"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print("Starting ML Workflow Coordinator...")
    coordinator = MLWorkflowCoordinator()
    
    success = coordinator.run_full_workflow()
    
    print("\nFinal Status:")
    print(f"Workflow {'succeeded' if success else 'failed'}")
    print(f"Metrics saved to: C:/Users/suman/OneDrive/Desktop/AI Agents/workflow_metrics.json")
    print("Dashboard running at: http://127.0.0.1:8050/ ")

if __name__ == "__main__":
    main()