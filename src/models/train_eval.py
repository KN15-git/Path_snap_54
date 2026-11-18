import os
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, log_loss, accuracy_score, precision_score, recall_score
from src.utils.data_loader import load_trip, list_trip_ids, get_evaluation_trip_ids
from src.features.feature_extractor import extract_features_for_trip, FEATURE_NAMES
from src.models.baseline_models import get_model_definitions
from src.hmm.viterbi import RoadNetworkGraph, HMMViterbiDecoder, evaluate_route_predictions

ARTIFACTS_DIR = r"N:\PES\Sem 5\ML\Project\artifacts"
RESULTS_DIR = r"N:\PES\Sem 5\ML\Project\results"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def build_dataset(trip_ids, partition="train", max_trips=20, search_radius=40.0):
    """
    Extracts and concatenates feature matrices and labels across trips.
    """
    selected_trips = trip_ids[:max_trips]
    print(f"Building dataset from {len(selected_trips)} trips in partition '{partition}'...", flush=True)

    all_X = []
    all_y = []
    all_groups = []
    trip_caches = {}

    for idx, tid in enumerate(selected_trips):
        trip = load_trip(tid)
        res = extract_features_for_trip(trip, search_radius=search_radius)
        if res["y"] is not None and len(res["y"]) > 0:
            all_X.append(res["X"])
            all_y.append(res["y"])
            all_groups.append(res["groups"])
            trip_caches[tid] = {
                "trip": trip,
                "res": res
            }
        print(f"  [{partition}] Processed trip {idx + 1}/{len(selected_trips)} (ID: {tid})", flush=True)

    X = np.vstack(all_X)
    y = np.concatenate(all_y)
    groups = np.concatenate(all_groups)
    print(f"Dataset complete: X shape={X.shape}, positive={np.sum(y)}, negative={len(y)-np.sum(y)}", flush=True)
    return X, y, groups, trip_caches

def evaluate_on_student_test(clf, test_trip_caches, max_eval_trips=3):
    """
    Evaluates classifier and HMM route reconstruction across student-test trips.
    Returns:
        mean_top1_acc, mean_f1, mean_logloss, mean_continuity, mean_infer_time_ms
    """
    top1_accuracies = []
    f1_scores = []
    log_losses = []
    continuities = []
    infer_times = []

    eval_items = list(test_trip_caches.items())[:max_eval_trips]

    for tid, data in eval_items:
        res = data["res"]
        trip = data["trip"]
        X_test = res["X"]
        y_test = res["y"]
        true_arcs = res["true_arcs_per_point"]
        candidates = res["candidates_per_point"]

        t0 = time.time()
        # Classifier probabilities
        probs = clf.predict_proba(X_test)[:, 1]
        infer_time_ms = (time.time() - t0) * 1000.0 / max(1, len(X_test))
        infer_times.append(infer_time_ms)

        # Macro F1
        preds = (probs >= 0.5).astype(int)
        f1 = f1_score(y_test, preds, average="macro", zero_division=0)
        f1_scores.append(f1)

        # Log loss
        try:
            ll = log_loss(y_test, probs, eps=1e-6)
            log_losses.append(ll)
        except Exception:
            pass

        # Split probabilities per GPS observation
        probs_per_point = []
        offset = 0
        for cand_list in candidates:
            k = len(cand_list)
            probs_per_point.append(probs[offset:offset + k])
            offset += k

        # HMM Viterbi decoding
        road_graph = RoadNetworkGraph(trip["nodes"], trip["arcs"])
        decoder = HMMViterbiDecoder(road_graph)
        pred_arcs, _ = decoder.decode(candidates, probs_per_point, trip["track"])

        # Sequence-level evaluation
        eval_res = evaluate_route_predictions(pred_arcs, true_arcs, trip["arcs"])
        top1_accuracies.append(eval_res["top1_accuracy"])
        continuities.append(eval_res["route_continuity"])

    return (
        np.mean(top1_accuracies),
        np.mean(f1_scores),
        np.mean(log_losses) if len(log_losses) > 0 else np.nan,
        np.mean(continuities),
        np.mean(infer_times)
    )

def run_experimentation():
    """
    Trains and compares all 6 baseline models using GroupKFold and student-test benchmarking.
    """
    train_trip_ids = list_trip_ids("train")
    test_trip_ids = get_evaluation_trip_ids()

    # Build train and student-test data (20 train trips gives ~160,000 samples)
    X_train, y_train, groups_train, _ = build_dataset(train_trip_ids, "train", max_trips=20)
    X_test, y_test, _, test_trip_caches = build_dataset(test_trip_ids, "student_test", max_trips=10)

    model_defs = get_model_definitions()
    results = []

    print("\n" + "="*80, flush=True)
    print("STARTING 6-MODEL BENCHMARK EXPERIMENTATION", flush=True)
    print("="*80, flush=True)

    for name, info in model_defs.items():
        print(f"\n--- Training Model: {name} ({info['family']}) ---", flush=True)
        model = info["model"]

        # Subsample for KNN to prevent O(N^2) latency
        if name == "KNN" and len(X_train) > 20000:
            np.random.seed(42)
            sub_idx = np.random.choice(len(X_train), size=20000, replace=False)
            X_tr_fit = X_train[sub_idx]
            y_tr_fit = y_train[sub_idx]
            groups_tr_fit = groups_train[sub_idx]
        else:
            X_tr_fit = X_train
            y_tr_fit = y_train
            groups_tr_fit = groups_train

        # 1. 3-Fold GroupKFold Cross-Validation on Train Data
        gkf = GroupKFold(n_splits=3)
        cv_f1_scores = []

        t_start = time.time()
        for fold, (train_idx, val_idx) in enumerate(gkf.split(X_tr_fit, y_tr_fit, groups_tr_fit)):
            X_tr, y_tr = X_tr_fit[train_idx], y_tr_fit[train_idx]
            X_val, y_val = X_tr_fit[val_idx], y_tr_fit[val_idx]

            model.fit(X_tr, y_tr)
            val_probs = model.predict_proba(X_val)[:, 1]
            val_preds = (val_probs >= 0.5).astype(int)
            score = f1_score(y_val, val_preds, average="macro", zero_division=0)
            cv_f1_scores.append(score)

        cv_f1 = np.mean(cv_f1_scores)
        train_duration = time.time() - t_start

        # 2. Refit on full training set
        model.fit(X_tr_fit, y_tr_fit)

        # 3. Save model artifact
        model_filename = os.path.join(ARTIFACTS_DIR, f"{name.lower().replace(' ', '_')}.joblib")
        joblib.dump(model, model_filename)

        # 4. Evaluate on Student-Testing Partition
        top1_acc, test_f1, logloss, continuity, infer_time_ms = evaluate_on_student_test(model, test_trip_caches)

        print(f"Results for {name}:", flush=True)
        print(f"  CV Macro F1: {cv_f1:.4f}", flush=True)
        print(f"  Student-Test Top-1 Accuracy: {top1_acc:.2f}%", flush=True)
        print(f"  Student-Test Route Continuity: {continuity:.2f}%", flush=True)
        print(f"  Student-Test Macro F1: {test_f1:.4f}", flush=True)
        print(f"  Student-Test Log Loss: {logloss:.4f}", flush=True)
        print(f"  Inference Latency: {infer_time_ms:.3f} ms/point", flush=True)
        print(f"  Training Time: {train_duration:.2f} s", flush=True)

        results.append({
            "Model Name": name,
            "Algorithm Family": info["family"],
            "Key Configuration": info["description"],
            "CV Macro F1-score": round(cv_f1, 4),
            "Top-1 Accuracy (%)": round(top1_acc, 2),
            "Route Continuity (%)": round(continuity, 2),
            "Student-Test Macro F1-score": round(test_f1, 4),
            "Log Loss": round(logloss, 4) if not np.isnan(logloss) else "N/A",
            "Inference Time (ms)": round(infer_time_ms, 3),
            "Training Time (s)": round(train_duration, 2)
        })

    # Save summary table
    df_results = pd.DataFrame(results)
    csv_path = os.path.join(RESULTS_DIR, "metrics_summary.csv")
    df_results.to_csv(csv_path, index=False)
    print("\n" + "="*80, flush=True)
    print("CONSOLIDATED MODEL COMPARISON TABLE SAVED TO:", csv_path, flush=True)
    print(df_results.to_string(index=False), flush=True)
    print("="*80, flush=True)

if __name__ == "__main__":
    run_experimentation()
