# src/train_models.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

def train_system():
    # 1. Load Data
    print("Loading data...")
    df = pd.read_csv('data/ride_data.csv')
    
    # --- MODEL 1: ETA PREDICTION ---
    print("\nTraining ETA Predictor...")
    # Features: Distance, Hour, Day, vehicle_type (encoded)
    # We map vehicle type to numbers
    v_map = {'economy': 0, 'premium': 1, 'van': 2}
    df['v_type_code'] = df['vehicle_type'].map(v_map)
    
    X_eta = df[['trip_distance', 'hour', 'day_of_week', 'v_type_code']]
    y_eta = df['trip_duration']
    
    X_train, X_test, y_train, y_test = train_test_split(X_eta, y_eta, test_size=0.2, random_state=42)
    
    eta_model = RandomForestRegressor(n_estimators=50, max_depth=10)
    eta_model.fit(X_train, y_train)
    
    # Evaluate
    preds = eta_model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    print(f"ETA Model MAE: {mae:.2f} minutes")
    
    # --- MODEL 2: DEMAND/SURGE PREDICTION ---
    print("\nTraining Demand/Surge Predictor...")
    # We predict the 'demand_surge' factor based on time and location context
    # Simplified: Using Hour, Day, and rough location (Lat/Lon)
    X_demand = df[['hour', 'day_of_week', 'origin_lat', 'origin_lon']]
    y_demand = df['demand_surge']
    
    X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X_demand, y_demand, test_size=0.2, random_state=42)
    
    demand_model = RandomForestRegressor(n_estimators=50, max_depth=5)
    demand_model.fit(X_train_d, y_train_d)
    
    preds_d = demand_model.predict(X_test_d)
    rmse = np.sqrt(mean_squared_error(y_test_d, preds_d))
    print(f"Demand Model RMSE: {rmse:.4f}")
    
    # 3. Save Models
    joblib.dump(eta_model, 'models/eta_model.pkl')
    joblib.dump(demand_model, 'models/demand_model.pkl')
    print("\nModels saved to 'models/' folder.")

if __name__ == "__main__":
    train_system()