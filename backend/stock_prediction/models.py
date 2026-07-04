import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, ClassifierMixin


class MajorityClassBaseline(BaseEstimator, ClassifierMixin):
    # Predicts the most frequent class every time.
    # Establishes the floor, any model that can't beat this
    # isn't learning anything useful from the features.

    def fit(self, X, y):
        counts = np.bincount(y)
        self.majority_class_ = int(np.argmax(counts))
        return self

    def predict(self, X):
        return np.full(len(X), self.majority_class_, dtype=int)

    def predict_proba(self, X):
        n = len(X)
        proba = np.zeros((n, 2))
        proba[:, self.majority_class_] = 1.0
        return proba


def build_logistic_regression(random_state: int = 42) -> Pipeline:
    # Logistic regression with L2 regularization (C=0.1 is fairly strong).
    # class_weight="balanced" compensates for the slight class imbalance
    # between up and down days in stock return distributions.
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=0.1,
            max_iter=1000,
            solver="lbfgs",
            random_state=random_state,
            class_weight="balanced",
        )),
    ])


def build_random_forest(random_state: int = 42) -> RandomForestClassifier:
    # Shallow trees (max_depth=6) and high min_samples_leaf prevent overfitting
    # to noise in financial data, which is notoriously low signal-to-noise.
    # These are included alongside the DQN to show accuracy comparisons.
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=20,
        max_features="sqrt",
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )


def fit_feature_scaler(X_train: np.ndarray) -> "StandardScaler":
    # Scaler is fit only on training data to prevent leakage.
    # The DQN's features span very different ranges (RSI: 0–100, MA ratios: ~1.0,
    # ATR: ~0.01), standardizing ensures no single feature dominates the Q-network.
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


def chronological_split(
    feat_df: pd.DataFrame,
    feature_cols: list,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> dict:
    # Strict chronological split, no shuffling.
    # Shuffling would leak future data into training, producing overly
    # optimistic metrics that wouldn't hold up in live trading.
    n = len(feat_df)
    n_train = int(n * train_frac)
    n_val   = int(n * val_frac)

    train = feat_df.iloc[:n_train]
    val   = feat_df.iloc[n_train : n_train + n_val]
    test  = feat_df.iloc[n_train + n_val :]

    return {
        "X_train": train[feature_cols].values,
        "y_train": train["Target"].values,
        "X_val":   val[feature_cols].values,
        "y_val":   val["Target"].values,
        "X_test":  test[feature_cols].values,
        "y_test":  test["Target"].values,
        "dates_train": train.index,
        "dates_val":   val.index,
        "dates_test":  test.index,
        "test_df":     test,
    }


def rf_up_probability(rf_model, X: np.ndarray) -> np.ndarray:
    # Probability the next day closes up, per the random forest. Guards the
    # degenerate single-class case (a window where Target never varies).
    proba = rf_model.predict_proba(X)
    if proba.shape[1] == 1:
        return np.full(len(X), float(rf_model.classes_[0]))
    return proba[:, 1]


def rf_filtered_signals(dqn_signals: np.ndarray, rf_up_prob: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    # Use the RF as a directional confirmation filter on the DQN: keep a long
    # only when the RF also expects an up-move. The DQN and RF make different
    # kinds of errors, so gating entries on agreement cuts false positives.
    filtered = np.asarray(dqn_signals).copy()
    mask     = (filtered == 1) & (np.asarray(rf_up_prob) < threshold)
    filtered[mask] = 0
    return filtered


class MajorityVoteEnsemble:
    # Combines multiple classifiers by majority vote.
    # Useful for reducing variance when individual models disagree
    # if at least half predict "up", the ensemble predicts "up".

    def __init__(self, models: list):
        self.models = models

    def predict(self, X: np.ndarray) -> np.ndarray:
        votes = np.stack([m.predict(X) for m in self.models])
        return (votes.sum(axis=0) >= len(self.models) / 2).astype(int)


def train_all_models(X_train, y_train) -> dict:
    models = {
        "Baseline (Majority Class)": MajorityClassBaseline(),
        "Logistic Regression":       build_logistic_regression(),
        "Random Forest":             build_random_forest(),
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
        print(f"  Trained: {name}")
    return models
