# src/generate_plots.py
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, r2_score

def create_evaluation_plots():
    print("Generating evaluation plots...")
    
    # Load data and models
    df = pd.read_csv('data/ride_data.csv')
    eta_model = joblib.load('models/eta_model.pkl')
    
    # Prepare Data for ETA Eval
    v_map = {'economy': 0, 'premium': 1, 'van': 2}
    df['v_type_code'] = df['vehicle_type'].map(v_map)
    X = df[['trip_distance', 'hour', 'day_of_week', 'v_type_code']]
    y_true = df['trip_duration']
    y_pred = eta_model.predict(X)
    
    # Plot 1: Actual vs Predicted ETA
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Actual Duration (mins)')
    plt.ylabel('Predicted Duration (mins)')
    plt.title(f'ETA Model Accuracy (R2: {r2_score(y_true, y_pred):.2f})')
    plt.savefig('models/eta_accuracy_plot.png')
    print("Saved models/eta_accuracy_plot.png")

    # Plot 2: Demand Surge Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['demand_surge'], bins=20, kde=True)
    plt.title('Distribution of Demand Surge Multipliers')
    plt.xlabel('Surge Multiplier')
    plt.savefig('models/demand_dist_plot.png')
    print("Saved models/demand_dist_plot.png")

if __name__ == "__main__":
    create_evaluation_plots()