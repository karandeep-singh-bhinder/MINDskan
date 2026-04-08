# MindScan — Python ML Backend Setup

## Project Structure

```
mindscan/
├── app.py               ← Flask server + ML pipeline
├── requirements.txt     ← Python dependencies
├── templates/
│   └── index.html       ← Frontend (served by Flask)
├── data/
│   └── mindscan.db      ← SQLite database (auto-created on first run)
└── README.md
```

---

## 1. Install Python Dependencies

Make sure you have **Python 3.9+** installed, then run:

```bash
pip install -r requirements.txt
```

---

## 2. Run Locally

```bash
python app.py
```

Open your browser at → **http://127.0.0.1:5000**

The ML model trains automatically on startup. The SQLite database is created at `data/mindscan.db`.

---

## 3. API Endpoints

| Method | Endpoint        | Description                          |
|--------|-----------------|--------------------------------------|
| GET    | `/`             | Serves the frontend                  |
| POST   | `/api/predict`  | Accepts answers, returns prediction  |
| GET    | `/api/metrics`  | Returns current model metrics        |
| GET    | `/api/history`  | Returns last 50 anonymised responses |

### POST /api/predict — Example

**Request:**
```json
{ "answers": [2, 1, 3, 2, 4, 1, 2, 3, 2, 1, 0, 1, 2, 3, 4] }
```

**Response:**
```json
{
  "stress_pct": 54.2,
  "level": "moderate",
  "metrics": {
    "accuracy": 88.5,
    "precision": 87.2,
    "recall": 86.9,
    "f1": 87.0,
    "dataset_size": 781
  }
}
```

---

## 4. How the ML Pipeline Works

1. **Data Generation** — 780 balanced synthetic samples (PSS-10 based)
2. **Real Data Merge** — User responses from SQLite are appended
3. **Data Cleaning** — Drops rows with values outside [0, 4]
4. **Normalisation** — MinMaxScaler scales all features to [0, 1]
5. **Shuffle** — Random permutation to prevent ordering bias
6. **Train/Test Split** — 80% training, 20% testing (stratified)
7. **Training** — `LogisticRegression(solver='lbfgs', multi_class='ovr', max_iter=1000)`
8. **Evaluation** — Accuracy, Precision, Recall, F1 (macro average)
9. **Prediction** — New response is classified into Low / Moderate / High

---

## 5. Deploy to the Web

### Option A: Render (Free Tier)

1. Push this folder to a GitHub repo
2. Go to [render.com](https://render.com) → New → Web Service
3. Connect your GitHub repo
4. Set:
   - **Runtime:** Python 3
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app`
5. Click Deploy

Add `gunicorn` to `requirements.txt`:
```
gunicorn>=21.0
```

### Option B: Railway

1. Push to GitHub
2. Go to [railway.app](https://railway.app) → New Project → Deploy from GitHub
3. Railway auto-detects Flask — set start command to `python app.py`

### Option C: Vercel (Frontend only)

Vercel doesn't support Python backends. Use **Render** or **Railway** for the full-stack version, and **Vercel** only if you separate the frontend.

---

## 6. Using a Real Dataset (Kaggle PSS)

To use a real Kaggle dataset instead of synthetic data:

1. Download a CSV with columns like: `q1, q2, ..., q15, stress_level`
2. Replace `generate_synthetic_data()` in `app.py` with:

```python
import pandas as pd

def load_kaggle_data(path="data/pss_dataset.csv"):
    df = pd.read_csv(path)
    X = df[[f"q{i}" for i in range(1, 16)]].values.astype(float)
    y = df["stress_level"].values.astype(int)  # 0, 1, or 2
    return X, y
```

---

## 7. Add MongoDB (Optional)

Replace SQLite with MongoDB using `pymongo`:

```python
from pymongo import MongoClient

client = MongoClient("your-mongodb-uri")
db = client["mindscan"]
collection = db["responses"]

# Save response
collection.insert_one({"answers": answers, "level": level, "date": datetime.utcnow()})

# Load for training
docs = list(collection.find())
```

---

## Notes

- The model retrains after **every submission** — this means the first few users influence the model significantly. For production, consider retraining on a schedule (e.g., every 100 responses).
- The SQLite database persists between restarts. Delete `data/mindscan.db` to reset.
- This is an academic project. Do not use as a clinical tool.
