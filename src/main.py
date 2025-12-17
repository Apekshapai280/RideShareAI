# src/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

# Initialize API
app = FastAPI(title="AI Vehicle Matcher & Pricing Engine")

# Load Models (Global variables)
try:
    eta_model = joblib.load('models/eta_model.pkl')
    demand_model = joblib.load('models/demand_model.pkl')
    print("Models loaded successfully.")
except:
    print("Models not found. Run train_models.py first.")

# Request Schema
class RideRequest(BaseModel):
    origin_lat: float
    origin_lon: float
    dest_lat: float
    dest_lon: float
    user_preference: str = "balanced" # 'fastest', 'cheapest', 'balanced'

# Helper: Haversine Distance
def calculate_distance(lat1, lon1, lat2, lon2):
    # Radius of earth in km
    R = 6371 
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2) * np.sin(dlat/2) + \
        np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * \
        np.sin(dlon/2) * np.sin(dlon/2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    d = R * c
    return round(d, 2)

@app.post("/ride/quote")
def get_ride_quote(request: RideRequest):
    # 1. Feature Engineering
    current_time = datetime.now()
    hour = current_time.hour
    day = current_time.weekday()
    
    dist_km = calculate_distance(
        request.origin_lat, request.origin_lon, 
        request.dest_lat, request.dest_lon
    )
    
    if dist_km == 0:
        raise HTTPException(status_code=400, detail="Origin and Destination are the same.")

    # 2. Predict Demand/Surge
    # Input: [hour, day, lat, lon]
    demand_features = pd.DataFrame([[hour, day, request.origin_lat, request.origin_lon]], 
                                   columns=['hour', 'day_of_week', 'origin_lat', 'origin_lon'])
    surge_multiplier = float(demand_model.predict(demand_features)[0])
    # Cap surge between 1.0 and 3.0 for sanity
    surge_multiplier = max(1.0, min(3.0, surge_multiplier))

    # 3. Generate Options for different vehicles
    vehicles = ['economy', 'premium', 'van']
    v_map = {'economy': 0, 'premium': 1, 'van': 2}
    base_prices = {'economy': 50, 'premium': 80, 'van': 100}
    per_km_rate = 10
    per_min_rate = 2
    
    results = []
    
    for v in vehicles:
        # Prepare input for ETA model
        # Features: [trip_distance, hour, day, v_type_code]
        eta_input = pd.DataFrame([[dist_km, hour, day, v_map[v]]], 
                                 columns=['trip_distance', 'hour', 'day_of_week', 'v_type_code'])
        
        predicted_duration = float(eta_model.predict(eta_input)[0])
        
        # Calculate Dynamic Price
        # Formula: (Base + DistCost + TimeCost) * Surge
        base_cost = base_prices[v] + (dist_km * per_km_rate) + (predicted_duration * per_min_rate)
        final_price = round(base_cost * surge_multiplier, 2)
        
        results.append({
            "vehicle_type": v,
            "eta_minutes": round(predicted_duration, 1),
            "price": final_price,
            "surge_applied": round(surge_multiplier, 2),
            "distance_km": dist_km
        })
        
    # 4. Ranking Logic
    # Normalize scores to rank them
    # Preference Logic:
    # Cheapest: Sort by price
    # Fastest: Sort by ETA
    # Balanced: 0.7 * Price_Norm + 0.3 * ETA_Norm (lower score is better)
    
    if request.user_preference == 'cheapest':
        results.sort(key=lambda x: x['price'])
    elif request.user_preference == 'fastest':
        results.sort(key=lambda x: x['eta_minutes'])
    else: # balanced
        # Simple weighted sort
        results.sort(key=lambda x: (x['price'] * 0.7) + (x['eta_minutes'] * 2))

    return {
        "request_id": "req_" + str(int(datetime.timestamp(current_time))),
        "user_preference": request.user_preference,
        "recommendations": results
    }

@app.get("/")
def home():
    return {"status": "System Operational", "model_version": "1.0"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)