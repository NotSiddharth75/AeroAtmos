"""
data_generator.py
Generates a synthetic dataset that stands in for real IMD block-level
forecasts + panchayat-level geospatial/historical data, so you can train
and demo the downscaling model without waiting on real data access.

Replace this with real sources later:
- IMD block forecast: https://mausam.imd.gov.in/
- Panchayat boundaries / lat-lon: Bihar SDMA / Census 2011 shapefiles
- Soil & land-use: Bhuvan / NBSS-LUP
- Elevation: SRTM 30m DEM

Run:
    python data_generator.py
Produces: panchayat_master.csv, training_data.csv
"""

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

# ---- 1. District -> Block -> Panchayat hierarchy (sample, Bihar) ----
STRUCTURE = {
    "Patna": {
        "Masaurhi": ["Masaurhi Buzurg", "Dhanarua", "Chishi", "Sarai",
                     "Nadwan", "Bahadurpur", "Kishunpur", "Amanabad"],
        "Bikram":   ["Bikram Purab", "Bikram Paschim", "Lakhna", "Maranchi"],
        "Naubatpur":["Naubatpur Uttari", "Naubatpur Dakshini", "Deoria"],
    },
    "Gaya": {
        "Tikari":   ["Tikari Purab", "Tikari Paschim", "Amethi"],
        "Sherghati":["Sherghati Uttar", "Sherghati Dakshin", "Bhaluatand"],
    },
}

SOIL_TYPES = ["alluvial", "loamy", "clayey", "sandy_loam"]
LAND_USE = ["paddy", "wheat_maize", "vegetable", "mixed_agri", "fallow"]

rows = []
pid = 1
for district, blocks in STRUCTURE.items():
    base_lat = 25.3 + rng.uniform(-0.4, 0.4)
    base_lon = 84.9 + rng.uniform(-0.5, 0.5)
    for block, panchayats in blocks.items():
        block_lat = base_lat + rng.uniform(-0.15, 0.15)
        block_lon = base_lon + rng.uniform(-0.15, 0.15)
        for panch in panchayats:
            lat = block_lat + rng.uniform(-0.05, 0.05)
            lon = block_lon + rng.uniform(-0.05, 0.05)
            elevation = rng.uniform(45, 95)          # metres, Bihar plains
            soil = rng.choice(SOIL_TYPES)
            landuse = rng.choice(LAND_USE)
            # historical 10-yr seasonal averages
            hist_rain = rng.normal(20, 5)
            hist_temp = rng.normal(31, 2)
            rows.append({
                "panchayat_id": f"P{pid:04d}",
                "panchayat": panch,
                "block": block,
                "district": district,
                "lat": round(lat, 4),
                "lon": round(lon, 4),
                "elevation_m": round(elevation, 1),
                "soil_type": soil,
                "land_use": landuse,
                "hist_avg_rainfall_mm": round(max(hist_rain, 0), 1),
                "hist_avg_temp_c": round(hist_temp, 1),
            })
            pid += 1

master = pd.DataFrame(rows)
master.to_csv("panchayat_master.csv", index=False)
print(f"Wrote panchayat_master.csv with {len(master)} panchayats "
      f"across {master['block'].nunique()} blocks.")

# ---- 2. Synthetic training data: block forecast -> true panchayat outcome ----
# In production this label comes from AWS/ground station observations.
# Here we simulate a physically-plausible local deviation driven by
# elevation, soil, land-use and small spatial noise, so the model has a
# real (learnable) signal to downscale.
records = []
for _, p in master.iterrows():
    for month in range(1, 13):
        season_factor = 1.6 if month in (6, 7, 8, 9) else 0.5  # monsoon bump
        block_rain = max(rng.normal(18 * season_factor, 6), 0)
        block_temp = rng.normal(30 if month not in (12, 1) else 20, 3)

        soil_factor = {"clayey": 1.15, "loamy": 1.0,
                       "alluvial": 1.05, "sandy_loam": 0.85}[p.soil_type]
        elev_factor = 1 - (p.elevation_m - 70) * 0.002
        landuse_bump = {"paddy": 1.1, "wheat_maize": 0.95, "vegetable": 1.0,
                        "mixed_agri": 1.0, "fallow": 0.9}[p.land_use]

        true_rain = max(block_rain * soil_factor * elev_factor * landuse_bump
                         + rng.normal(0, 2), 0)
        true_temp = block_temp - (p.elevation_m - 70) * 0.01 + rng.normal(0, 0.5)
        true_humidity = np.clip(55 + season_factor * 10 + rng.normal(0, 5), 20, 98)
        true_wind = max(rng.normal(10, 3), 0)

        records.append({
            "panchayat_id": p.panchayat_id,
            "month": month,
            "block_forecast_rain_mm": round(block_rain, 1),
            "block_forecast_temp_c": round(block_temp, 1),
            "lat": p.lat, "lon": p.lon, "elevation_m": p.elevation_m,
            "soil_type": p.soil_type, "land_use": p.land_use,
            "hist_avg_rainfall_mm": p.hist_avg_rainfall_mm,
            "hist_avg_temp_c": p.hist_avg_temp_c,
            "true_rainfall_mm": round(true_rain, 1),
            "true_temp_c": round(true_temp, 1),
            "true_humidity_pct": round(true_humidity, 1),
            "true_wind_kmph": round(true_wind, 1),
        })

training = pd.DataFrame(records)
training.to_csv("training_data.csv", index=False)
print(f"Wrote training_data.csv with {len(training)} rows "
      f"({len(master)} panchayats x 12 months).")
