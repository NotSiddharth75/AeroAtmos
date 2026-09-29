# AeroAtmos
### AI-Powered Hyperlocal Weather Downscaling

**Team AERO VISION** · Smart India Hackathon — Problem Statement **SIH 26074**

---

## The Problem

IMD (India Meteorological Department) publishes weather forecasts at the **Block** level. But a farmer makes decisions — irrigate today, delay spraying, protect a harvest — at the **Panchayat** level. Rainfall and temperature can vary meaningfully within a single block due to soil type, elevation, and land use, so a block-wide forecast can genuinely mislead a farmer in a specific village.

## Our Solution

AeroAtmos takes a Block-level weather forecast and uses a trained machine learning model to **downscale** it into a Panchayat-level forecast, using each panchayat's:

- Geographic coordinates and elevation
- Soil type
- Land use (paddy, wheat/maize, vegetable, mixed agriculture, fallow)
- Historical average rainfall and temperature

The output is then turned into a **plain-language farmer advisory** (e.g. *"avoid irrigation today"*, *"delay pesticide application"*) — not just a number.

---

## What's in this repo

```
AeroAtmos/
├── backend/
│   ├── data_generator.py     # Generates the synthetic District→Block→Panchayat dataset
│   ├── train_model.py        # Trains the RandomForest downscaling model
│   ├── app.py                 # FastAPI backend serving live predictions
│   ├── requirements.txt
│   ├── panchayat_master.csv   # Generated: panchayat hierarchy + geo/soil/land-use data
│   ├── training_data.csv      # Generated: 12-month training examples
│   └── downscaling_model.joblib  # Generated: trained model
├── weather-down-live.html     # Interactive dashboard (map, charts, advisory) — talks to the backend
└── weather-down.html          # Standalone demo version (simulated data, no backend needed)
```

## Features

- **Interactive satellite map** — panchayats shown as color-coded markers by rainfall, temperature, humidity, or wind
- **12-month timeline** with play/scrub controls to see seasonal variation
- **Block forecast → AI panchayat forecast comparison** — the core value proposition, shown explicitly
- **Trend and comparison charts** — historical average vs. block forecast vs. AI prediction, and a ranked view of all panchayats in a block
- **Automated farmer advisory** generated from the predicted conditions
- **Alerts panel** flagging panchayats crossing heavy-rain or heat thresholds
- **CSV export** of a block's full 12-month forecast
- **Live FastAPI backend** with interactive Swagger docs at `/docs`

---

## Tech Stack

| Layer | Technology |
|---|---|
| ML Model | scikit-learn (RandomForest, multi-output regression) |
| Backend API | FastAPI + Uvicorn |
| Data | pandas, numpy |
| Frontend | HTML/CSS/JS, Leaflet.js (map), Chart.js (charts) |

---

## Getting Started

### 1. Set up the backend

```bash
cd backend
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
```

### 2. Generate data and train the model

```bash
python data_generator.py
python train_model.py
```

### 3. Start the API

```bash
uvicorn app:app --reload
```

The API is now live at `http://127.0.0.1:8000` — see interactive docs at `http://127.0.0.1:8000/docs`.

### 4. Open the dashboard

Open `weather-down-live.html` directly in your browser (double-click it) while the server from step 3 is running. It connects automatically to `http://127.0.0.1:8000`.

> Prefer a version with no backend setup? Open `weather-down.html` instead — it runs a client-side simulation of the same downscaling logic, no server required.

---

## API Endpoints

| Endpoint | Description |
|---|---|
| `GET /districts` | List all districts |
| `GET /blocks/{district}` | List blocks in a district |
| `GET /panchayats/{block}` | List panchayats in a block |
| `GET /forecast/{panchayat_id}?month=` | Downscaled forecast + advisory for one panchayat |
| `GET /forecast/block/{block}?month=` | Downscaled forecast for every panchayat in a block |

---

## Current Scope & Next Steps

This is a working prototype built to prove the downscaling pipeline end-to-end:

- **Data:** Currently synthetic (2 districts, 5 blocks, ~21 panchayats in Bihar) — generated with realistic seasonal and geographic patterns. Real deployment would use IMD's block forecast API and Census/SDMA panchayat boundaries.
- **Model:** RandomForest chosen for fast prototyping with zero extra native dependencies. Architecture supports swapping in XGBoost or spatial models (CNN/GNN) as a next step.
- **Delivery:** Dashboard is the demo interface; real-world deployment would extend to SMS/IVR/WhatsApp alerts for farmers without smartphone access.
- **Validation:** Model accuracy is currently measured against synthetic data; next step is validation against real ground-station observations.

---

## Team AERO VISION

Built for Smart India Hackathon 2026 — Problem Statement 26074.
