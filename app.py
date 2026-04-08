"""
MindScan — Python Flask Backend
ML Model: Logistic Regression via scikit-learn
Database: SQLite
"""

import sqlite3
import json
import os
import numpy as np
from datetime import datetime
from flask import Flask, request, jsonify, render_template, g

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

app = Flask(__name__)
DB_PATH = os.path.join(os.path.dirname(__file__), "data", "mindscan.db")

# ─────────────────────────────────────────────
# DATABASE
# ─────────────────────────────────────────────

def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db

@app.teardown_appcontext
def close_db(exc):
    db = g.pop("db", None)
    if db:
        db.close()

def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS responses (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                answers     TEXT    NOT NULL,
                stress_pct  REAL    NOT NULL,
                level       TEXT    NOT NULL,
                created_at  TEXT    NOT NULL
            )
        """)
        conn.commit()


# ─────────────────────────────────────────────
# ML PIPELINE
# ─────────────────────────────────────────────

FEATURE_WEIGHTS = [1.2, 1.1, 1.3, 1.2, 1.4,
                   1.0, 1.1, 1.0, 1.2, 1.1,
                   1.0, 0.9, 1.0, 1.1, 1.3]

_model_cache = {}   # {"model": ..., "scaler": ..., "metrics": ...}


def generate_synthetic_data(n=780):
    """
    Generates balanced synthetic student stress data.
    Each class (0=Low, 1=Moderate, 2=High) gets n//3 samples.
    """
    rng = np.random.default_rng(seed=42)
    X, y = [], []
    per_class = n // 3

    for cls in range(3):
        for _ in range(per_class):
            row = []
            for _ in range(15):
                if cls == 0:
                    v = rng.integers(0, 2)        # 0-1
                elif cls == 1:
                    v = rng.integers(1, 4)        # 1-3
                else:
                    v = rng.integers(2, 5)        # 2-4
                row.append(int(v))
            X.append(row)
            y.append(cls)

    return np.array(X, dtype=float), np.array(y, dtype=int)


def load_user_data():
    """Load real user responses stored in SQLite."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            rows = conn.execute("SELECT answers, level FROM responses").fetchall()
        X, y = [], []
        label_map = {"low": 0, "moderate": 1, "high": 2}
        for row in rows:
            answers = json.loads(row[0])
            lbl = label_map.get(row[1], -1)
            if lbl != -1 and len(answers) == 15:
                X.append(answers)
                y.append(lbl)
        return np.array(X, dtype=float), np.array(y, dtype=int)
    except Exception:
        return np.empty((0, 15)), np.empty(0, dtype=int)


def train_model():
    """Full ML pipeline: generate + real data → clean → scale → train → evaluate."""
    X_syn, y_syn = generate_synthetic_data(780)
    X_real, y_real = load_user_data()

    if len(X_real) > 0:
        X_all = np.vstack([X_syn, X_real])
        y_all = np.concatenate([y_syn, y_real])
    else:
        X_all, y_all = X_syn, y_syn

    # Clean: drop rows with any value outside [0, 4]
    mask = np.all((X_all >= 0) & (X_all <= 4), axis=1)
    X_all, y_all = X_all[mask], y_all[mask]

    # Shuffle
    idx = np.random.permutation(len(X_all))
    X_all, y_all = X_all[idx], y_all[idx]

    # Scale features to [0, 1]
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_scaled = scaler.fit_transform(X_all)

    # Train/test split 80/20
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_all, test_size=0.2, random_state=42, stratify=y_all
    )

    # Logistic Regression (sklearn — production-grade)
    model = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        C=1.0,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy":  round(accuracy_score(y_test, y_pred) * 100, 1),
        "precision": round(precision_score(y_test, y_pred, average="macro", zero_division=0) * 100, 1),
        "recall":    round(recall_score(y_test, y_pred, average="macro", zero_division=0) * 100, 1),
        "f1":        round(f1_score(y_test, y_pred, average="macro", zero_division=0) * 100, 1),
        "dataset_size": int(len(X_all)),
    }

    _model_cache["model"]   = model
    _model_cache["scaler"]  = scaler
    _model_cache["metrics"] = metrics
    return model, scaler, metrics


def get_model():
    if "model" not in _model_cache:
        train_model()
    return _model_cache["model"], _model_cache["scaler"], _model_cache["metrics"]


def score_to_level(pct: float) -> str:
    if pct <= 33:
        return "low"
    elif pct <= 66:
        return "moderate"
    return "high"


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/metrics")
def api_metrics():
    """Return current model performance metrics."""
    _, _, metrics = get_model()
    return jsonify(metrics)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    POST { "answers": [0..4] × 15 }
    Returns prediction, stress_pct, level, metrics.
    """
    data = request.get_json(force=True)
    answers = data.get("answers", [])

    if len(answers) != 15 or not all(0 <= v <= 4 for v in answers):
        return jsonify({"error": "Invalid input: need 15 values in [0,4]"}), 400

    # Compute rule-based stress %
    raw = sum(answers[i] * FEATURE_WEIGHTS[i] for i in range(15))
    max_possible = 4 * sum(FEATURE_WEIGHTS)
    stress_pct = round((raw / max_possible) * 100, 1)
    level = score_to_level(stress_pct)

    # Save to DB
    db = get_db()
    db.execute(
        "INSERT INTO responses (answers, stress_pct, level, created_at) VALUES (?,?,?,?)",
        (json.dumps(answers), stress_pct, level, datetime.utcnow().isoformat())
    )
    db.commit()

    # Retrain with new data point included, then predict
    model, scaler, metrics = train_model()
    X_input = scaler.transform([answers])
    predicted_class = int(model.predict(X_input)[0])
    level_map = {0: "low", 1: "moderate", 2: "high"}
    ml_level = level_map[predicted_class]

    return jsonify({
        "stress_pct": stress_pct,
        "level":      ml_level,
        "metrics":    metrics,
    })


@app.route("/api/history")
def api_history():
    """Return last 50 anonymised responses for analytics."""
    db = get_db()
    rows = db.execute(
        "SELECT stress_pct, level, created_at FROM responses ORDER BY id DESC LIMIT 50"
    ).fetchall()
    return jsonify([dict(r) for r in rows])


# ─────────────────────────────────────────────
# STARTUP
# ─────────────────────────────────────────────

if __name__ == "__main__":
    init_db()
    train_model()       # warm up the model before first request
    print("\n  MindScan server running → http://127.0.0.1:5000\n")
    app.run(debug=True, port=5000)
