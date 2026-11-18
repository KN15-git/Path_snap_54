# 🗺️ PathSnap: Distributed Map-Matching & Route Reconstruction
### **UE24CS352A Machine Learning — Department of Computer Science & Engineering, PES University**

### 👥 Team Members
| Name | SRN | Role |
| :--- | :--- | :--- |
| **KRUTHIKNANDAN R** | `PES1UG24CS239` | Laptop 1 Host & API Lead (FastAPI, Model Training & Inference Pipeline) |
| **Jiya Datta Banik** | `PES1UG24CS202` | Data Exploration, Spatial Preprocessing & Feature Extraction |
| **JOEL THOMAS** | `PES1UG24CS203` | Laptop 2 Frontend Lead (Streamlit UI & Network Client Integration) |
| **KRRISH AARYAN** | `PES1UG24CS238` | Interactive Folium Geographical Visualizations & Model Evaluation |

PathSnap is an enterprise-grade distributed map-matching and route reconstruction system designed to reconstruct realistic vehicle trajectories along an underlying road network from noisy, sparse GPS observations.

---

### **📋 Course Deliverables Checklist**
- [x] **Deliverable 1: Complete Public GitHub Repository**: Source code for data preprocessing, spatial indexing, feature extraction, 6-model training, Bayesian optimization, HMM/Viterbi route reconstruction, FastAPI backend, and Streamlit frontend.
- [x] **Deliverable 2: Consolidated Technical Report (PDF)**: [PathSnap_Technical_Report.pdf](PathSnap_Technical_Report.pdf) *(Comprehensive report covering all mandatory questions, 6-model comparison, diagnostic charts, Folium maps, and independent evaluator test set verification)*.
- [x] **Interactive Master Notebook**: [PathSnap_Complete_Master_Notebook.ipynb](PathSnap_Complete_Master_Notebook.ipynb) *(100% executable from beginning to end with live outputs and 0 errors)*.

---

## 🏛️ System Architecture

PathSnap is built as a **two-tier distributed client-server architecture**:

```
+-------------------------------------------------------------+
|                     LAPTOP 2: FRONTEND                      |
|                     (Client / Streamlit)                    |
|  - Evaluation Control Panel & Dashboard                     |
|  - Interactive 4-Layer Folium Map Visualisation             |
|  - Real-time Performance & Accuracy Metrics Cards           |
|  - REST Client (Calls Laptop 1 over Wi-Fi / LAN)            |
+------------------------------+------------------------------+
                               |
                   HTTP POST /predict/{trip_id}
                   HTTP GET  /trips, /health
                               |
                               v
+-------------------------------------------------------------+
|                     LAPTOP 1: BACKEND                       |
|                     (Server / FastAPI)                      |
|  - Pretrained Champion Model (Tuned XGBoost, Optuna TPE)    |
|  - Spatial Indexing (cKDTree, R=45m candidate generation)   |
|  - 9D Kinematic & Geometric Feature Extraction              |
|  - HMM Emission & Dijkstra Transition Probabilities         |
|  - Dynamic Programming Viterbi Route Reconstruction         |
|  - Server-Side Ground-Truth Evaluation Engine               |
+-------------------------------------------------------------+
```

---

## 🚀 Quick Setup Instructions

### Prerequisites
- Python 3.10+ (Miniconda / Anaconda recommended)
- Both laptops connected to the **same Wi-Fi / Local Network** (or mobile hotspot).

### Step 1: Clone the Repository
On **both laptops**:
```bash
git clone https://github.com/KN15-git/Path_snap_54.git
cd Path_snap_54
pip install -r requirements.txt
```

---

## 💻 Running Laptop 1 (Backend Server)

*Owned by Teammate 1 — serves the ML model, HMM/Viterbi inference, and evaluation.*

1. **Find Laptop 1's Local IP Address:**
   - On Windows: Open Command Prompt and type `ipconfig`. Note your **IPv4 Address** under your active Wi-Fi adapter (e.g. `192.168.1.15`).
   - On macOS/Linux: Run `ifconfig` or `ip a` (e.g. `192.168.1.15`).

2. **Start the FastAPI Server:**
   - **Windows One-Click:** Double-click `start_backend.bat`.
   - **Or via Terminal:**
     ```bash
     python run_server.py
     ```
   The backend will start and listen on `http://0.0.0.0:8000`.

3. **Verify Server Status:**
   - Open your browser to `http://localhost:8000/health`. You will see:
     ```json
     {"status": "healthy", "model_loaded": true, "model_architecture": "XGBClassifier"}
     ```
   - Swagger API Documentation is accessible at `http://localhost:8000/docs`.

4. **Give your IPv4 Address to Teammate 2 (Laptop 2)!**

---

## 💻 Running Laptop 2 (Frontend Client)

*Owned by Teammate 2 — user interface, trip selection, and map rendering.*

1. **Start the Streamlit Application:**
   - **Windows One-Click:** Double-click `start_frontend.bat`.
   - **Or via Terminal:**
     ```bash
     python -m streamlit run app.py --server.port 8501
     ```

2. **Connect to Laptop 1:**
   - Streamlit will open in your browser at `http://localhost:8501`.
   - In the left sidebar under **"Laptop 1 API URL"**, change `http://127.0.0.1:8000` to your teammate's IP:
     ```
     http://<LAPTOP_1_IP>:8000
     ```
     *(Example: `http://192.168.1.15:8000`)*
   - The status indicator will switch to **Connected (Green)** and load the 10 reserved evaluation trips!

3. **Run Map-Matching:**
   - Choose any trip (e.g., `00000023`, `00000049`, `00000073`).
   - Click **"Predict Route"** (typically finishes in 1–2 seconds).
   - Review accuracy and continuity metrics in the dashboard.
   - Navigate to **"Route Visualisation"** in the sidebar to inspect the 4-layer Folium map (toggle GPS points, Road Network, Reference Route, and Reconstructed Path).

---

## 🧪 Single-Laptop Demonstration (Testing Mode)

If testing both backend and frontend on a single machine:
- Simply double-click `start_both.bat`.
- It will launch both the FastAPI backend (`:8000`) and the Streamlit frontend (`:8501`) simultaneously. Leave the default URL `http://127.0.0.1:8000` in the sidebar.

---

## 📊 Benchmark & Evaluation Results

Across all **10 Evaluator-Test trips** (19,674 GPS points, completely held out):

| Metric | Classifier-Only | Classifier + HMM / Viterbi | Target Baseline | Result |
| :--- | :---: | :---: | :---: | :---: |
| **Observation Top-1 Accuracy** | 94.70% | **96.77%** | 95.0% | **EXCEEDED (+1.77%)** |
| **Route Continuity** | 95.45% | **98.91%** | 98.0% | **EXCEEDED (+0.91%)** |
| **Candidate Miss Rate (R=45m)** | — | **0.00%** | < 1.0% | **ZERO LOSS** |
| **Mean Inference Latency** | — | **1.07 ms / observation** | < 5.0 ms | **REAL-TIME** |

---

## 📁 Repository Structure

```
Path_snap_54/
├── app.py                                   # Laptop 2: Streamlit Frontend Application
├── api_client.py                            # Laptop 2: REST API client connecting to Laptop 1
├── map_visualization.py                     # Laptop 2: 4-Layer interactive Folium map generator
├── run_server.py                            # Laptop 1: FastAPI server entry point
├── start_backend.bat                        # Laptop 1: One-click launcher for backend (:8000)
├── start_frontend.bat                       # Laptop 2: One-click launcher for frontend (:8501)
├── start_both.bat                           # Local testing: One-click launcher for both
├── requirements.txt                         # Unified project dependencies
├── README.md                                # This documentation guide
├── PathSnap_Complete_Master_Notebook.ipynb  # End-to-end master notebook with all 16 rubric sections
├── artifacts/
│   └── best_model.joblib                    # Tuned XGBoost champion model artifact (4.7 MB)
├── src/                                     # Core pipeline package
│   ├── api/main.py                          # FastAPI endpoints (/health, /trips, /predict)
│   ├── features/feature_extractor.py        # Spatial cKDTree search & 9D feature extraction
│   ├── hmm/viterbi.py                       # RoadNetworkGraph & Viterbi shortest-path decoding
│   ├── models/classifier.py                 # Multi-model wrappers & scikit-learn pipelines
│   └── utils/data_loader.py                 # Multi-partition trip loader (.track, .nodes, .arcs, .route)
└── Student folder/                          # Evaluation dataset partitions
    ├── student_test/                        # 10 student test trips
    └── evaluator_test/                      # 10 reserved evaluator test trips
```
