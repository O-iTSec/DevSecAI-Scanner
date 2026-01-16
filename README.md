# devsecai-scanner

AI-Driven Container Vulnerability Scanner: A Machine Learning Approach to Risk Classification
Abstract
This project presents an automated pipeline for assessing security risks in containerized environments using machine learning. The system integrates vulnerability scanning, quantitative feature extraction, statistical preprocessing, and probabilistic classification to evaluate Docker images according to their predicted security risk. The goal is to enhance DevSecOps workflows by automating detection of high-risk containers and supporting data-driven security policy enforcement.
1. Introduction
Containerized applications introduce unique security challenges stemming from outdated base images, dependency vulnerabilities, and configuration weaknesses. Traditional scanning tools generate raw vulnerability data but lack mechanisms to prioritize risks or contextualize severity.
This project addresses these limitations by applying machine learning to classify container images into Low, Medium, or High risk categories based on real vulnerability scan data collected from 100+ Docker images. This creates an intelligent risk-scoring model that augments traditional DevSecOps processes.
2. System Architecture
The system is structured as an end-to-end DevSecAI pipeline composed of several stages:
Image Collection – automated retrieval of official and verified community Docker images
Vulnerability Scanning – comprehensive Trivy scans producing structured JSON reports
Feature Engineering – extraction of vulnerability counts, severity ratios, and fixability metrics
Machine Learning Model Training – supervised risk classification using Random Forest, Logistic Regression, and XGBoost
Inference Engine – real-time prediction for new container images
DevSecOps Integration – security gating in CI/CD pipelines based on predicted risk
3. Data Acquisition and Preprocessing
A dataset was constructed from vulnerability scans of over 100 Docker images. Each JSON scan file was parsed to extract:
Total CVEs
Severity-based counts (Critical, High, Medium, Low, Unknown)
Number of fixable vulnerabilities
High-risk vulnerability ratio
Fixable ratio
Severity index (weighted severity score)
These metrics were compiled into a structured dataset (dataset.csv) suitable for supervised learning.
4. Machine Learning Methodology
4.1 Model Selection
The following supervised learning models were trained and compared:
Random Forest Classifier
Logistic Regression
XGBoost Classifier (optional, if installed)
These models were chosen for their strong performance on tabular data and interpretability in security contexts.
4.2 Labeling Strategy
Risk labels were derived using a composite risk score based on severity and fixability metrics:
Risk Level	Threshold
Low	< 0.08
Medium	0.08–0.18
High	> 0.18
To simulate real-world ambiguity and imperfect human labeling, a small percentage of label noise was optionally introduced.
4.3 Evaluation Metrics
Model evaluation used:
Accuracy
Precision
Recall
F1-score
Confusion Matrix
These metrics provided insight into classification performance and generalization.
5. Prediction Pipeline
A dedicated script (predict_risk.py) implements the inference pipeline:
Accepts a Trivy JSON scan
Extracts all required features
Loads the trained model, scaler, and encoder
Predicts the risk label and model confidence
Outputs class probabilities and vulnerability summary
This enables both on-demand and automated CI/CD risk evaluation.
6. DevSecOps Integration
The trained model can be integrated directly into CI/CD systems (GitHub Actions, Jenkins, GitLab CI). High-risk images can trigger:
Build failure
Pipeline halt
Security team notification
Automatic remediation workflows
This shifts security left by preventing unsafe containers from entering production environments.
7. Results
Key findings include:
High predictive performance on deterministic labels (up to 100%)
Realistic performance of 80–95% with noise-injected labels
Feature importance showed strong correlation with intuitive security drivers:
Critical & High CVE counts
High-risk ratio
Fixable ratio
These outcomes demonstrate that machine learning is effective for prioritizing vulnerabilities in container ecosystems.
8. Future Work
Opportunities for expansion include:
Integrating EPSS (Exploit Prediction Scoring System)
Linking vulnerabilities to MITRE ATT&CK techniques
SHAP-based explainability for model decisions
Automated retraining using new CVE data
Kubernetes admission controller integration
Streamlit dashboard for live risk visualization
9. Conclusion
This project demonstrates the feasibility of applying machine learning to container vulnerability management. By combining automated scanning, feature engineering, statistical learning, and DevSecOps integration, the system delivers intelligent risk classification and enables proactive security enforcement within modern cloud-native workflows.
