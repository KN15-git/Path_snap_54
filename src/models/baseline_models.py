import time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def get_model_definitions():
    """
    Returns a dictionary of 6 models across distinct algorithm families,
    configured with their initial hyperparameters and descriptions.
    """
    models = {
        "Logistic Regression": {
            "family": "Linear and Probabilistic",
            "model": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(class_weight="balanced", max_iter=500, random_state=42))
            ]),
            "description": "Standardized linear model with balanced class weighting."
        },
        "Gaussian Naive Bayes": {
            "family": "Probabilistic",
            "model": GaussianNB(),
            "description": "Generative probabilistic classifier modeling Gaussian feature likelihoods."
        },
        "KNN": {
            "family": "Instance-based",
            "model": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", KNeighborsClassifier(n_neighbors=15, weights="distance", algorithm="kd_tree", n_jobs=-1))
            ]),
            "description": "Distance-weighted K-Nearest Neighbors measuring local feature similarity."
        },
        "XGBoost": {
            "family": "Boosting",
            "model": XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.08,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=7.0,  # accounts for ~1:7 to 1:8 class imbalance
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1
            ),
            "description": "Extreme gradient boosted decision trees optimized for tabular ranking."
        },
        "LightGBM": {
            "family": "Ensemble / Tree",
            "model": LGBMClassifier(
                n_estimators=100,
                max_depth=7,
                num_leaves=31,
                learning_rate=0.08,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=7.0,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            ),
            "description": "Histogram-based gradient boosted trees with leaf-wise splitting."
        },
        "MLP Classifier": {
            "family": "Neural Networks",
            "model": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200, early_stopping=True, random_state=42))
            ]),
            "description": "Multi-Layer Perceptron neural network with nonlinear ReLU activations."
        }
    }
    return models
