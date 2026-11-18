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
- [x] **Deliverable 2: Consolidated Technical Report (PDF)**: [PathSnap_Technical_Report.pdf](PathSnap_Technical_Report.pdf) *(Comprehensive 14-page report with cover page, mandatory question answers, 6-model comparison, diagnostic charts, Folium maps, and evaluator test set verification)*.
- [x] **GPS Path Detection Notebook**: [GPS_Path_Detection.ipynb](GPS_Path_Detection.ipynb) *(100% executable from beginning to end with live outputs and 0 errors)*.

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

## 📁 Repository Directory Structure

```text
Path_snap_54/
├── .env.example                               # Network configuration template (Laptop 1 IP)
├── .gitignore                                 # Clean exclusion rules for caches, envs & temp files
├── README.md                                  # Complete documentation, team & execution guide
├── requirements.txt                           # Pinned dependencies for backend and frontend
├── run_server.py                              # Backend FastAPI server entry point
├── app.py                                     # Frontend Streamlit UI application
├── api_client.py                              # REST client communicating with Laptop 1 API
├── map_visualization.py                       # Folium 4-layer map builder
├── GPS_Path_Detection.ipynb                   # Deliverable 1: 46-cell master notebook with live outputs
├── PathSnap_Technical_Report.pdf              # Deliverable 2: Official 14-page technical report PDF
├── PathSnap_Technical_Report.html             # Web-viewable HTML format of the technical report
│
├── artifacts/                                 # Serialized model weights & pipeline assets
│   └── best_model.joblib                      # Bayesian-tuned XGBoost champion model
│
├── Student folder/                            # Evaluation datasets
│   ├── student_test/                          # 10 student-testing benchmark trips
│   └── evaluator_test/                        # 10 held-out evaluator evaluation trips
│
├── src/                                       # Modular Python package
│   ├── __init__.py
│   ├── api/                                   # FastAPI REST microservice
│   │   ├── __init__.py
│   │   └── main.py                            # Endpoints: /health, /trips, /predict/{trip_id}
│   ├── features/                              # Spatial indexing & 9D feature engineering
│   │   ├── __init__.py
│   │   ├── candidate_generator.py             # cKDTree candidate generation with adaptive radius
│   │   └── feature_extractor.py               # Vector projections and 9D feature derivation
│   ├── hmm/                                   # Probabilistic sequence reconstruction
│   │   ├── __init__.py
│   │   └── viterbi.py                         # RoadNetworkGraph and HMMViterbiDecoder
│   ├── models/                                # Model training and experiment scripts
│   │   ├── __init__.py
│   │   ├── baseline_models.py                 # 6 model family definitions
│   │   ├── bayesian_tune.py                   # Optuna TPE hyperparameter optimizer
│   │   └── train_eval.py                      # GroupKFold training and evaluation pipeline
│   └── utils/                                 # Utilities and mathematical projections
│       ├── __init__.py
│       ├── data_loader.py                     # Atomic loader for .track, .nodes, .arcs, .route
│       └── geo.py                             # Equirectangular metric projection & Haversine math
│
├── outputs/                                   # Dynamically generated trajectory HTML maps
│
└── results/                                   # Experimental evidence and visual artifacts
    ├── bayesian_tuning_summary.csv            # Optuna tuning before/after comparison
    ├── metrics_summary.csv                    # 6-model cross-validation and test benchmark
    ├── evaluator_test_metrics_summary.csv     # 10 held-out evaluator trips performance
    ├── hmm_ablation_10trips.csv               # Classifier vs. HMM ablation table
    ├── figures/                               # 20+ diagnostic charts, training curves & UI logs
    │   ├── laptop1_cmd_api_response.jpg       # Live CMD screenshot of Laptop 1 backend
    │   ├── laptop2_ui_connected.jpg           # Live Streamlit screenshot showing connection
    │   ├── laptop2_ui_prediction_results.jpg  # Live Streamlit metrics card screenshot
    │   ├── laptop2_ui_folium_route_full.jpg   # Live Folium full route view
    │   ├── laptop2_ui_folium_route_zoom.jpg   # Live Folium zoom-in road matching view
    │   ├── trip_00000015_folium_map.jpg       # Cover page initial trajectory map
    │   └── master_*.png                       # Diagnostic plots and comparison charts
    └── maps/                                  # Interactive Folium HTML maps
        ├── trip_00000000_complete_4layer_map.html
        └── trip_00000001_eda_map.html
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
   - On Windows: Open Command Prompt and type `ipconfig`. Note your **IPv4 Address** under your active Wi-Fi adapter (e.g. `10.30.201.63`).
   - On macOS/Linux: Run `ifconfig` or `ip a`.

2. **Start the FastAPI Server:**
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
   ```bash
   streamlit run app.py --server.port 8501
   ```

2. **Connect to Laptop 1:**
   - Streamlit will open in your browser at `http://localhost:8501`.
   - In the left sidebar under **"Laptop 1 API URL"**, enter your teammate's IP:
     ```
     http://<LAPTOP_1_IP>:8000
     ```
     *(Example: `http://10.30.201.63:8000`)*
   - The status indicator will switch to **Connected (Green)** and load the 10 reserved evaluation trips!

3. **Run Map-Matching:**
   - Choose any trip (e.g., `00000023`, `00000015`, `00000049`).
   - Click **"Predict Route"** (typically finishes in 0.5–2 seconds).
   - Review accuracy and continuity metrics in the dashboard.
   - Navigate to **"Route Visualisation"** in the sidebar to inspect the 4-layer Folium map (toggle GPS points, Road Network, Reference Route, and Reconstructed Path).

---

## 🧪 Single-Laptop Demonstration (Testing Mode)

If testing both backend and frontend on a single machine:
- **Terminal 1:**
  ```bash
  python run_server.py
  ```
- **Terminal 2:**
  ```bash
  streamlit run app.py --server.port 8501
  ```
- Leave the default URL `http://127.0.0.1:8000` in the sidebar.

---

## 📊 Experimental Results Benchmark

| Model | Algorithm Family | CV Macro F1 | Student-Test Macro F1 | Top-1 Accuracy | Route Continuity | Log Loss | Inference Time |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | Linear & Probabilistic | 0.8742 | 0.8711 | 95.07% | 99.44% | 0.1419 | 0.0001 ms |
| **Gaussian Naive Bayes** | Probabilistic Generative | 0.8637 | 0.8611 | 94.89% | 99.40% | 0.3508 | 0.0002 ms |
| **Distance-Weighted KNN** | Instance-Based | 0.9066 | 0.9248 | 95.79% | 99.43% | 0.1124 | 0.0200 ms |
| **LightGBM Classifier** | Histogram Tree Boosting | 0.9228 | 0.9243 | 96.13% | 99.46% | 0.0925 | 0.0008 ms |
| **XGBoost (Baseline)** | Gradient Tree Boosting | 0.9264 | 0.9272 | 96.16% | 99.45% | 0.0920 | 0.0004 ms |
| **MLP Classifier** | Neural Networks | 0.9282 | 0.9413 | 96.25% | 99.47% | 0.0693 | 0.0003 ms |
| **Bayesian Tuned XGBoost** | **Champion Model** | **0.9316** | **0.9423** | **96.42%** | **99.47%** | **0.0688** | **0.0010 ms** |

---

## ⚖️ Academic Integrity & Independent Work

This project was conceived, implemented, trained, and documented independently by the team members listed above. All reported experimental metrics, figures, Folium visualizations, and two-laptop deployment logs represent genuine experimental results obtained from the dataset.
