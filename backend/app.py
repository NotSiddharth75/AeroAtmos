"""
app.py — AeroAtmos backend (AI-Powered Hyperlocal Weather Downscaling)
FastAPI backend for AeroAtmos: Panchayat-level weather intelligence.

Endpoints
---------
GET  /districts
GET  /blocks/{district}
GET  /panchayats/{block}
GET  /forecast/{panchayat_id}?month=7
GET  /forecast/block/{block}?month=7      (all panchayats in a block, for the map)

Run:
    pip install -r requirements.txt
    python data_generator.py
    python train_model.py
    uvicorn app:app --reload
Then open http://127.0.0.1:8000/docs
"""

from datetime import datetime

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="AeroAtmos API — AI-Powered Hyperlocal Weather Downscaling", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                    allow_headers=["*"])

master = pd.read_csv("panchayat_master.csv")
model = joblib.load("downscaling_model.joblib")

NUMERIC = ["month", "block_forecast_rain_mm", "block_forecast_temp_c",
           "lat", "lon", "elevation_m", "hist_avg_rainfall_mm", "hist_avg_temp_c"]
CATEGORICAL = ["soil_type", "land_use"]


def fetch_block_forecast(block: str, month: int) -> dict:
    """
    Stand-in for the real IMD block-level API call. Replace this with a
    request to IMD's block forecast feed. Kept deterministic (seeded by
    block name + month) so the demo is repeatable.
    """
    seed = abs(hash((block, month))) % 1000
    season = 1.6 if month in (6, 7, 8, 9) else 0.6
    rain = round(10 + (seed % 20) * season, 1)
    temp = round(22 + (seed % 15) - (6 if month in (12, 1) else 0), 1)
    return {"block_forecast_rain_mm": rain, "block_forecast_temp_c": temp}


def advisory_for(rain: float, temp: float, humidity: float) -> list[str]:
    tips = []
    if rain >= 30:
        tips += ["Heavy rainfall expected — avoid irrigation today",
                 "Delay pesticide/fertilizer application",
                 "Check field drainage before evening"]
    elif rain >= 10:
        tips += ["Moderate rain expected — hold off on spraying",
                 "Good window to skip today's irrigation"]
    else:
        tips.append("Little to no rain expected")

    if temp >= 36 and rain < 10:
        tips += ["Hot, dry conditions — increase irrigation frequency",
                 "Avoid spraying during peak afternoon heat"]
    if humidity >= 85:
        tips.append("High humidity — monitor crops for fungal risk")
    return tips


def predict_panchayat(row: pd.Series, month: int) -> dict:
    block_fc = fetch_block_forecast(row.block, month)
    features = pd.DataFrame([{
        "month": month,
        "block_forecast_rain_mm": block_fc["block_forecast_rain_mm"],
        "block_forecast_temp_c": block_fc["block_forecast_temp_c"],
        "lat": row.lat, "lon": row.lon, "elevation_m": row.elevation_m,
        "hist_avg_rainfall_mm": row.hist_avg_rainfall_mm,
        "hist_avg_temp_c": row.hist_avg_temp_c,
        "soil_type": row.soil_type, "land_use": row.land_use,
    }])
    pred = model.predict(features)[0]
    rain, temp, humidity, wind = [round(float(v), 1) for v in pred]
    rain = max(rain, 0)
    return {
        "panchayat_id": row.panchayat_id,
        "panchayat": row.panchayat,
        "block": row.block,
        "district": row.district,
        "lat": float(row.lat),
        "lon": float(row.lon),
        "block_forecast_rain_mm": block_fc["block_forecast_rain_mm"],
        "rainfall_mm": rain,
        "temperature_c": temp,
        "humidity_pct": humidity,
        "wind_kmph": max(wind, 0),
        "rain_probability_pct": min(round(rain / 40 * 100 + 20, 0), 98),
        "hist_avg_rainfall_mm": row.hist_avg_rainfall_mm,
        "advisory": advisory_for(rain, temp, humidity),
    }


@app.get("/districts")
def list_districts():
    return sorted(master.district.unique().tolist())


@app.get("/blocks/{district}")
def list_blocks(district: str):
    blocks = master[master.district == district].block.unique().tolist()
    if not blocks:
        raise HTTPException(404, "District not found")
    return sorted(blocks)


@app.get("/panchayats/{block}")
def list_panchayats(block: str):
    rows = master[master.block == block]
    if rows.empty:
        raise HTTPException(404, "Block not found")
    return rows[["panchayat_id", "panchayat"]].to_dict("records")


@app.get("/forecast/{panchayat_id}")
def forecast(panchayat_id: str, month: int | None = None):
    month = month or datetime.now().month
    row = master[master.panchayat_id == panchayat_id]
    if row.empty:
        raise HTTPException(404, "Panchayat not found")
    return predict_panchayat(row.iloc[0], month)


@app.get("/forecast/block/{block}")
def forecast_block(block: str, month: int | None = None):
    month = month or datetime.now().month
    rows = master[master.block == block]
    if rows.empty:
        raise HTTPException(404, "Block not found")
    return [predict_panchayat(r, month) for _, r in rows.iterrows()]
