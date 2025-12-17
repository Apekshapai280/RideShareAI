# src/generate_data.py
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

def generate_synthetic_data(num_samples=5000):
    print("Generating synthetic data...")
    
    # 1. Setup basic lists
    vehicle_types = ['economy', 'premium', 'van']
    base_fares = {'economy': 50, 'premium': 80, 'van': 100}
    
    data = []
    
    start_date = datetime.now()
    
    for _ in range(num_samples):
        # Simulate locations (approx lat/lon for a city like Bangalore/NYC)
        # Using a small grid approx 10x10km
        origin_lat = 12.97 + np.random.uniform(-0.05, 0.05)
        origin_lon = 77.59 + np.random.uniform(-0.05, 0.05)
        
        # Dest is origin + random shift
        dest_lat = origin_lat + np.random.uniform(-0.03, 0.03)
        dest_lon = origin_lon + np.random.uniform(-0.03, 0.03)
        
        # Calculate Mock Distance (Euclidean approx for speed)
        # 1 deg lat approx 111km
        dist_km = np.sqrt((dest_lat - origin_lat)**2 + (dest_lon - origin_lon)**2) * 111
        dist_km = round(max(0.5, dist_km), 2) # Min distance 0.5km
        
        # Time of day (0-23)
        hour = random.randint(0, 23)
        day_of_week = random.randint(0, 6)
        
        # Traffic Factor (Rush hour 8-10am, 5-8pm)
        traffic_multiplier = 1.0
        if (8 <= hour <= 10) or (17 <= hour <= 20):
            traffic_multiplier = np.random.uniform(1.2, 1.8)
        else:
            traffic_multiplier = np.random.uniform(0.9, 1.1)
            
        # Duration (Assume avg speed 30km/h -> 2 mins per km)
        base_duration_mins = dist_km * 2 
        actual_duration = round(base_duration_mins * traffic_multiplier, 2)
        
        # Vehicle & Price
        v_type = random.choice(vehicle_types)
        
        # Dynamic Pricing Logic (Simulated for ground truth)
        demand_factor = 1.0
        if traffic_multiplier > 1.2: # High traffic usually means high demand
            demand_factor = 1.5
            
        fare = round((base_fares[v_type] + (dist_km * 10) + (actual_duration * 2)) * demand_factor, 2)
        
        data.append([
            origin_lat, origin_lon, dest_lat, dest_lon, 
            hour, day_of_week, v_type, dist_km, actual_duration, fare, demand_factor
        ])
        
    columns = ['origin_lat', 'origin_lon', 'dest_lat', 'dest_lon', 
               'hour', 'day_of_week', 'vehicle_type', 'trip_distance', 
               'trip_duration', 'fare', 'demand_surge']
               
    df = pd.DataFrame(data, columns=columns)
    
    # Save to CSV
    output_path = 'data/ride_data.csv'
    df.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}. Shape: {df.shape}")

if __name__ == "__main__":
    generate_synthetic_data()