# Agentic Edge Intelligence for IoT Threat Detection 🛡️

Welcome to our project repository! This document tracks our progress and serves as a guide for all teammates coming on board to understand the system architecture, what has already been built, and how to jump in.

## 📖 Project Overview
This project focuses on building a **Multi-Agent IoT Security Framework** that operates at the network edge. Instead of sending all IoT network traffic to a centralized cloud component, this system deploys lightweight autonomous agents directly on edge devices to process streaming network data, detect anomalies, block known threats, and calculate complex risk vectors autonomously.

---

## 🚀 Progress: What Has Been Done So Far

We have successfully developed the core pipeline and implemented a fully functional distributed agent loop. The project currently simulates real-time data ingestion and runs the AI security validations instantaneously. 

### Phase 1: Core Framework & Data Pipeline 
- [x] **Project Environment & Directory Setup:** Established the folder structure (`agents/`, `core/`, `logs/`, `dataset/`) and `requirements.txt`.
- [x] **Data Streamer (`core/data_streamer.py`):** Instead of processing batch data, this streamer simulates a real IoT environment by reading instances line-by-line from the raw CICIDS dataset.
- [x] **Feature Processor (`core/feature_processor.py`):** Cleans invalid network packets (handling `NaN` or `inf` values) and standardizes numerical dimensions for the machine learning models.

### Phase 2: Agent Implementations
We have 5 core agents currently running sequentially on our edge simulated environment:
- [x] **Threat Intelligence Agent (`agents/threat_intel_agent.py`):** Built utilizing a highly memory-efficient **Bloom Filter**. This ensures $O(1)$ lookup times for checking IP addresses against known blocklists without bloating the edge device's RAM.
- [x] **Monitoring Agent (`agents/monitoring_agent.py`):** Intercepts data and invokes the feature processor to securely format raw payload information into processable structures.
- [x] **Anomaly Detection Agent (`agents/anomaly_agent.py`):** Powered by an **Isolation Forest** (Unsupervised ML) algorithm. It calculates mathematical variations in networking protocols and dynamically outputs an `anomaly_score`.
- [x] **Reasoning Agent ('Explainable AI') (`agents/reasoning_agent.py`):** The mastermind agent. It computes a combined Risk Score derived from the Anomaly Agent and Threat Intel Agent (e.g. `Risk = 0.7 * anomaly_score + 0.3 * threat_indicator`). It translates algorithmic decisions into human-readable text logs (e.g., "*Blocked IP due to abnormal port mapping AND presence on known signature list*").
- [x] **Mitigation Agent (`agents/mitigation_agent.py`):** Acts upon the final assigned Risk Category (LOW, MEDIUM, HIGH, CRITICAL) simulating firewall actions like "DROP", "ISOLATE", or "LOG".

### Phase 3: Orchestration and Testing
- [x] **Main Pipeline (`main.py`):** We stitched all modular agents into an operational loop. Data now successfully flows linearly through the entire multi-agent system.
- [x] **Performance Benchmarking (`evaluation.py`):** Successfully ran testing tools analyzing **Accuracy, Precision, Recall, and F1-Score** against normal traffic vs. actual malicious traffic, verifying our models operate properly at extremely high accuracies.

---

## 📁 Repository Structure
Our codebase is distributed as follows (check the specific directories to find the respective components):

```
project_directory/
│
├── core/                   # Utilities and global mechanics
│   ├── data_streamer.py
│   └── feature_processor.py
│
├── agents/                 # Autonomous modules
│   ├── threat_intel_agent.py
│   ├── monitoring_agent.py
│   ├── anomaly_agent.py
│   ├── reasoning_agent.py
│   └── mitigation_agent.py
│
├── dataset/                # Extracted CSV datasets (e.g. CICIDS)
├── logs/                   # System output 
│   └── reasoning_log.txt   # Explainable AI output log
│
├── main.py                 # Core running entrypoint
├── evaluation.py           # Metrics processing and accuracy tests
└── requirements.txt        # PIP modules and dependencies
```

---

## 🛠️ How To Run It (For Teammates)

1. Verify you have Python installed. It is recommended to use the generated `venv`.
2. First, install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. To run the continuous detection simulator and watch the AI make dynamic security decisions, run:
   ```bash
   python main.py
   ```
4. Check the `logs/reasoning_log.txt` to view the explainable justifications of what the framework decides to do with specific network packets.
5. To test mathematical accuracy and latency metrics:
   ```bash
   python evaluation.py
   ```

---

## 🎯 Next Steps / Ongoing Tasks
*(Teammates: Please update your tasks here!)*

- [ ] Further tuning the Machine Learning parameters for the Isolation Forest to reduce edge-case false positives.
- [ ] Connect the output Mitigation Agent into an actual system firewall wrapper (e.g., `iptables` simulator) instead of just string logging.
- [ ] Deploy and test this exact pipeline onto a Raspberry Pi to record physical CPU/RAM consumption graphs during active streaming.
