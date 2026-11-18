import os
import time
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.utils.data_loader import load_trip, get_evaluation_trip_ids, list_trip_ids
from src.features.feature_extractor import extract_features_for_trip
from src.hmm.viterbi import RoadNetworkGraph, HMMViterbiDecoder, evaluate_route_predictions

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [PathSnap Laptop-1 API] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("pathsnap_api")

app = FastAPI(
    title="PathSnap Backend API (Laptop 1)",
    description="Distributed Map-Matching & Route Reconstruction Backend",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))

MODEL_PATH = os.path.join(PROJECT_ROOT, "artifacts", "best_model.joblib")
FALLBACK_PATHS = [
    os.path.join(PROJECT_ROOT, "artifacts", "xgboost_tuned.joblib"),
    os.path.join(PROJECT_ROOT, "artifacts", "xgboost.joblib"),
    os.path.join(PROJECT_ROOT, "artifacts", "lightgbm.joblib"),
    os.path.join(PROJECT_ROOT, "artifacts", "mlp_classifier.joblib"),
    os.path.join(PROJECT_ROOT, "artifacts", "logistic_regression.joblib")
]

loaded_model = None
loaded_model_path = None

def get_loaded_model():
    global loaded_model, loaded_model_path
    if loaded_model is not None:
        return loaded_model

    for path in [MODEL_PATH] + FALLBACK_PATHS:
        if os.path.exists(path):
            try:
                loaded_model = joblib.load(path)
                loaded_model_path = path
                logger.info(f"Loaded ML model artifact from: {path}")
                return loaded_model
            except Exception as e:
                logger.warning(f"Failed loading model at {path}: {e}")
    return None

class PredictRequest(BaseModel):
    trip_id: str

@app.on_event("startup")
def startup_event():
    model = get_loaded_model()
    if model is not None:
        logger.info("FastAPI Server initialized and ready for Laptop 2 requests.")
    else:
        logger.warning("FastAPI Server started, but no model artifact was found.")

@app.get("/")
def root():
    return {
        "service": "PathSnap Map-Matching API",
        "status": "online",
        "role": "Laptop 1 (Model + Inference + Server-Side Evaluation)",
        "docs_url": "/docs"
    }

@app.get("/health")
def health():
    model = get_loaded_model()
    model_loaded = model is not None
    model_type = type(model).__name__ if model_loaded else "None"
    return {
        "status": "healthy" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "model_architecture": model_type,
        "model_artifact": os.path.basename(loaded_model_path) if loaded_model_path else None,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/trips")
def get_trips(partition: Optional[str] = None):
    p = partition or ("evaluator_test" if list_trip_ids("evaluator_test") else "student_test")
    trips = list_trip_ids(p)
    logger.info(f"GET /trips ({p}) -> Returning {len(trips)} evaluation trips to Laptop 2 UI")
    return {
        "count": len(trips),
        "partition": p,
        "trips": trips
    }

def _run_trip_prediction(trip_id: str) -> Dict[str, Any]:
    t_start = time.time()
    logger.info(f"===> INCOMING PREDICTION REQUEST from Laptop 2 | Trip ID: {trip_id}")

    model = get_loaded_model()
    if model is None:
        logger.error("Prediction aborted: Model artifact not loaded.")
        raise HTTPException(status_code=503, detail="Model artifact is not loaded on Laptop 1.")

    try:
        trip = load_trip(trip_id)
    except FileNotFoundError:
        logger.error(f"Trip ID '{trip_id}' not found.")
        raise HTTPException(status_code=404, detail=f"Trip ID '{trip_id}' not found.")

    track_df = trip["track"]
    nodes = trip["nodes"]
    arcs = trip["arcs"]
    n_pts = len(track_df)

    logger.info(f"  [Step 1/4] Loaded Trip {trip_id}: {n_pts} GPS observations, {len(nodes)} nodes, {len(arcs)} road segments.")

    # Step 2: Candidate generation & 9D feature extraction
    res = extract_features_for_trip(trip, search_radius=45.0)
    X = res["X"]
    candidates = res["candidates_per_point"]
    true_arcs = res["true_arcs_per_point"]
    logger.info(f"  [Step 2/4] Extracted {len(X)} candidate segment pairs (Miss Rate: {res['miss_rate']*100:.2f}%).")

    # Step 3: Classifier probability prediction
    probs = model.predict_proba(X)[:, 1]
    probs_per_point = []
    offset = 0
    for cand_list in candidates:
        k = len(cand_list)
        probs_per_point.append(probs[offset:offset + k])
        offset += k

    # Step 4: HMM + Viterbi Route Reconstruction
    road_graph = RoadNetworkGraph(nodes, arcs, arc_lengths=res["spatial_index"].arc_lengths)
    decoder = HMMViterbiDecoder(road_graph)
    pred_arcs, unique_route_arcs = decoder.decode(candidates, probs_per_point, track_df)
    logger.info(f"  [Step 3/4] HMM/Viterbi reconstructed sequence across {len(unique_route_arcs)} road segments.")

    # Compute mean orthogonal distance error (meters) for predicted segments
    perp_errors = []
    for pt_idx, pred_arc in enumerate(pred_arcs):
        for c in candidates[pt_idx]:
            if c["arc_id"] == pred_arc:
                perp_errors.append(c["perp_dist"])
                break
    mean_dist_error = float(np.mean(perp_errors)) if perp_errors else 0.0

    # Server-side evaluation against ground-truth .route (never exposing reference labels to UI)
    if true_arcs is not None:
        eval_metrics = evaluate_route_predictions(pred_arcs, true_arcs, arcs)
    else:
        eval_metrics = {"top1_accuracy": 0.0, "route_continuity": 0.0, "matched_points": 0, "total_points": n_pts}

    total_duration_ms = (time.time() - t_start) * 1000.0
    total_duration_s = total_duration_ms / 1000.0

    # Format coordinates for Laptop 2 UI (map_visualization.py expects [lat, lon] order!)
    gps_points_lat_lon = [[float(row["lat"]), float(row["lon"])] for _, row in track_df.iterrows()]

    # Build continuous [lat, lon] polyline along reconstructed route segments
    predicted_route_lat_lon = []
    for arc_id in unique_route_arcs:
        u = int(arcs[arc_id, 0])
        v = int(arcs[arc_id, 1])
        u_pt = [float(nodes[u, 1]), float(nodes[u, 0])]
        v_pt = [float(nodes[v, 1]), float(nodes[v, 0])]
        if not predicted_route_lat_lon or predicted_route_lat_lon[-1] != u_pt:
            predicted_route_lat_lon.append(u_pt)
        predicted_route_lat_lon.append(v_pt)

    # Build relevant road network GeoJSON (corridor of candidate arcs near the trajectory)
    # GeoJSON standard uses [lon, lat] inside geometry.coordinates
    corridor_arc_ids = set()
    for cand_list in candidates:
        for c in cand_list[:6]:
            corridor_arc_ids.add(int(c["arc_id"]))
    for arc_id in unique_route_arcs:
        corridor_arc_ids.add(int(arc_id))

    geojson_features = []
    for arc_id in corridor_arc_ids:
        u = int(arcs[arc_id, 0])
        v = int(arcs[arc_id, 1])
        geojson_features.append({
            "type": "Feature",
            "properties": {"arc_id": arc_id},
            "geometry": {
                "type": "LineString",
                "coordinates": [
                    [float(nodes[u, 0]), float(nodes[u, 1])],
                    [float(nodes[v, 0]), float(nodes[v, 1])]
                ]
            }
        })

    road_network_geojson = {
        "type": "FeatureCollection",
        "features": geojson_features
    }

    metrics_payload = {
        "accuracy": f"{eval_metrics['top1_accuracy']:.2f}%",
        "prediction_time": f"{total_duration_s:.2f} s",
        "distance_error": f"{mean_dist_error:.2f} m",
        "route_continuity": f"{eval_metrics['route_continuity']:.2f}%",
        "top1_accuracy": eval_metrics["top1_accuracy"],
        "route_continuity_pct": eval_metrics["route_continuity"],
        "mean_orthogonal_error_m": round(mean_dist_error, 2),
        "matched_points": eval_metrics["matched_points"],
        "total_points": eval_metrics["total_points"],
        "inference_time_ms": round(total_duration_ms, 2)
    }

    logger.info(
        f"<=== [Step 4/4] COMPLETED Trip {trip_id} in {total_duration_s:.2f}s | "
        f"Accuracy: {metrics_payload['accuracy']} | Continuity: {metrics_payload['route_continuity']} | "
        f"Mean Dist Error: {metrics_payload['distance_error']}"
    )

    return {
        "trip_id": trip_id,
        "status": "success",
        "num_observations": n_pts,
        "gps_points": gps_points_lat_lon,
        "predicted_route": predicted_route_lat_lon,
        "road_network": road_network_geojson,
        "reference_route": [],  # Ground truth strictly kept server-side per evaluation integrity rules
        "predicted_arcs": pred_arcs,
        "reconstructed_route_arcs": unique_route_arcs,
        "raw_gps": track_df[["lon", "lat"]].values.tolist(),
        "predicted_path_coords": [[pt[1], pt[0]] for pt in predicted_route_lat_lon],
        "metrics": metrics_payload,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict/{trip_id}")
def predict_trip_by_path(trip_id: str):
    return _run_trip_prediction(trip_id)

@app.post("/predict")
def predict_trip_by_body(req: PredictRequest):
    return _run_trip_prediction(req.trip_id)
