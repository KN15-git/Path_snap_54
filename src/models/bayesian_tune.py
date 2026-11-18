import os
import time
import joblib
import optuna
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score
from src.utils.data_loader import list_trip_ids, get_evaluation_trip_ids
from src.models.train_eval import build_dataset, evaluate_on_student_test, ARTIFACTS_DIR, RESULTS_DIR

optuna.logging.set_verbosity(optuna.logging.WARNING)

def run_bayesian_optimization(n_trials=25):
    """
    Executes Bayesian Hyperparameter Optimization on XGBoost using GroupKFold cross-validation.
    """
    print("\n" + "="*80)
    print("STARTING BAYESIAN HYPERPARAMETER OPTIMIZATION (OPTUNA)")
    print("="*80)

    train_trip_ids = list_trip_ids("train")
    test_trip_ids = get_evaluation_trip_ids()

    # Load datasets
    X_train, y_train, groups_train, _ = build_dataset(train_trip_ids, "train", max_trips=25)
    X_test, y_test, _, test_trip_caches = build_dataset(test_trip_ids, "student_test", max_trips=10)

    # 1. Baseline Untuned XGBoost evaluation
    print("\nEvaluating Untuned Baseline XGBoost...")
    untuned_model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.08,
        scale_pos_weight=7.0,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )
    t0 = time.time()
    untuned_model.fit(X_train, y_train)
    untuned_train_time = time.time() - t0
    untuned_top1, untuned_f1, untuned_ll, untuned_cont, untuned_infer = evaluate_on_student_test(untuned_model, test_trip_caches)

    # 2. Define Objective function for Optuna
    gkf = GroupKFold(n_splits=3)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 60, 180, step=20),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.20, log=True),
            "subsample": trial.suggest_float("subsample", 0.65, 0.95),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 0.95),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 5.0, 9.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
            "eval_metric": "logloss",
            "random_state": 42,
            "n_jobs": -1
        }

        fold_f1s = []
        for train_idx, val_idx in gkf.split(X_train, y_train, groups_train):
            X_tr, y_tr = X_train[train_idx], y_train[train_idx]
            X_val, y_val = X_train[val_idx], y_train[val_idx]

            clf = XGBClassifier(**params)
            clf.fit(X_tr, y_tr)
            preds = (clf.predict_proba(X_val)[:, 1] >= 0.5).astype(int)
            fold_f1s.append(f1_score(y_val, preds, average="macro", zero_division=0))

        return np.mean(fold_f1s)

    # 3. Optimize with TPE
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    print(f"Running {n_trials} Bayesian Optimization trials...")
    t_opt_start = time.time()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    opt_duration = time.time() - t_opt_start

    print(f"Bayesian Optimization finished in {opt_duration:.1f} s")
    print("Best Trial:")
    print(f"  Validation Macro F1: {study.best_value:.4f}")
    print("  Parameters:", study.best_params)

    # 4. Refit best model on full training data
    best_params = study.best_params.copy()
    best_params["eval_metric"] = "logloss"
    best_params["random_state"] = 42
    best_params["n_jobs"] = -1

    tuned_model = XGBClassifier(**best_params)
    t0 = time.time()
    tuned_model.fit(X_train, y_train)
    tuned_train_time = time.time() - t0

    # 5. Evaluate tuned model on student test
    tuned_top1, tuned_f1, tuned_ll, tuned_cont, tuned_infer = evaluate_on_student_test(tuned_model, test_trip_caches)

    # 6. Save best model artifact
    best_model_path = os.path.join(ARTIFACTS_DIR, "best_model.joblib")
    joblib.dump(tuned_model, best_model_path)
    print(f"Saved tuned champion model to: {best_model_path}")

    # 7. Produce comparison table
    comparison_data = [
        {
            "Configuration": "Untuned Baseline XGBoost",
            "CV Macro F1": "-",
            "Student-Test Top-1 Accuracy (%)": round(untuned_top1, 2),
            "Student-Test Macro F1": round(untuned_f1, 4),
            "Student-Test Log Loss": round(untuned_ll, 4),
            "Route Continuity (%)": round(untuned_cont, 2),
            "Inference Time (ms)": round(untuned_infer, 3),
            "Training Time (s)": round(untuned_train_time, 2)
        },
        {
            "Configuration": "Tuned XGBoost (Bayesian Optuna)",
            "CV Macro F1": round(study.best_value, 4),
            "Student-Test Top-1 Accuracy (%)": round(tuned_top1, 2),
            "Student-Test Macro F1": round(tuned_f1, 4),
            "Student-Test Log Loss": round(tuned_ll, 4),
            "Route Continuity (%)": round(tuned_cont, 2),
            "Inference Time (ms)": round(tuned_infer, 3),
            "Training Time (s)": round(tuned_train_time, 2)
        }
    ]

    df_comp = pd.DataFrame(comparison_data)
    comp_csv = os.path.join(RESULTS_DIR, "bayesian_tuning_summary.csv")
    df_comp.to_csv(comp_csv, index=False)

    print("\n" + "="*80)
    print("TUNED VS UNTUNED MODEL BENCHMARK:")
    print(df_comp.to_string(index=False))
    print("="*80)

if __name__ == "__main__":
    run_bayesian_optimization()
