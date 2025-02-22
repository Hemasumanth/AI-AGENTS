# AI Agents Framework

The AI Agents Framework is a comprehensive Python-based toolkit for developing and managing intelligent agent systems. This repository provides modular components for creating AI-driven workflows, machine learning integration, and performance analytics in agent-based systems.

## Features

**Core Agent System**  
The `AI_Agent.py` module contains fundamental classes for agent instantiation, message passing, and decision-making processes. This core system supports reactive agent architectures.

**Machine Learning Integration**  
Two specialized modules enable ML capabilities:
- `AI_agent_ml.py`: Implements supervised learning patterns for predictive analytics
- `AI_agent_ml_unsupervised.py`: Provides clustering and anomaly detection features

**Workflow Automation**  
The `AI_agent_workflow.py` module offers a DAG-based workflow engine with automatic task sequencing and dependency resolution. This system can supports both sequential and parallel execution modes.

**Performance Monitoring**  
Real-time metrics collection and analysis through:
- `AI_Agent_Metrics.py`: Tracks 12 key performance indicators
- `workflow_metrics.json`: Stores historical performance data

- ## Workflow Execution

### Automated ML Pipeline
The `AI_agent_workflow.py` implements an intelligent pipeline that handles:
1. Data preprocessing 🧹
2. Problem type detection 🔍 
3. Model selection 🤖
4. Performance metrics storage 💾


### Key Functionality
**Automatic Problem Detection**  
The system analyzes dataset characteristics to determine:
- Regression 📈 (Continuous values)
- Classification 🏷️ (Categorical labels)

Note : Example outpus are given in the repo 
1. temp.csv : To store the cleaned data for ml modle
2. confusion.csv :  For Ml model metrics
3. Dashboard : Dashboard runs at http://127.0.0.1:8050/ for monitering.

**Install required dependencies** 
pip install -r requirements.txt

**Example Dashboard**  
![image](https://github.com/user-attachments/assets/06105ebd-680d-4eb9-9dca-11d595275970)
![image](https://github.com/user-attachments/assets/ae81197b-9e0f-4c12-92b1-f76b0351d3df)
![image](https://github.com/user-attachments/assets/950feab4-53aa-45a2-944e-ec9cf8e34b71)










