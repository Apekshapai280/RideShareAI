
# AI-Driven Vehicle Matching & Dynamic Pricing System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-0.68%2B-green) ![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Model-orange)

## Overview
This project is an AI-powered backend system designed for a ride-hailing platform. It intelligently matches vehicles to ride requests by predicting Estimated Time of Arrival (ETA), forecasting demand surges, and calculating dynamic pricing in real-time.

The system utilizes **Machine Learning (Random Forest)** to learn from historical trip data and **FastAPI** to serve recommendations via a RESTful API.

##  Key Features
* **Synthetic Data Generation:** Simulates realistic urban traffic patterns, rush hours, and demand spikes.
* **ETA Prediction:** Uses ML to estimate trip duration based on distance, time of day, and traffic conditions.
* **Dynamic Pricing Engine:** Calculates surge multipliers (1.0x - 3.0x) based on predicted demand in specific locations.
* **Smart Vehicle Ranking:** Sorts available vehicles (Economy, Premium, Van) based on user preference:
    * *Cheapest* (Price optimized)
    * *Fastest* (Time optimized)
    * *Balanced* (Weighted score of price and time)

##  Project Structure
```

RideShareAI/
├── data/                   # Generated synthetic dataset
│   └── ride_data.csv
├── models/                 # Serialized ML models and plots
│   ├── eta_model.pkl
│   ├── demand_model.pkl
│   ├── eta_accuracy_plot.png
│   └── demand_dist_plot.png
├── src/                    # Source code
│   ├── generate_data.py    # Generates training data
│   ├── train_models.py     # Trains and saves ML models
│   ├── main.py             # FastAPI application
│   └── generate_plots.py   # Generates evaluation metrics
├── tests.py                # API Unit tests
├── requirements.txt        # Dependencies
└── README.md               # Project documentation

```

##  Installation & Setup

### 1. Clone the Repository

```bash
git clone [https://github.com/Apekshapai280/RideShareAI.git](https://github.com/Apekshapai280/RideShareAI.git)
cd RideShareAI

```

### 2. Set up Virtual Environment

```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate

```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
pip install matplotlib seaborn  # For plotting support

```

##  How to Run

Follow these steps in order to build and run the system:

**Step 1: Generate Data**
Creates a synthetic dataset with 5,000+ trip records.

```bash
python src/generate_data.py

```

**Step 2: Train AI Models**
Trains the ETA predictor (Random Forest) and Demand Forecasting model.

```bash
python src/train_models.py

```

**Step 3: Start the Server**
Launches the API on localhost port 8001.

```bash
python src/main.py

```

*Server will start at: `http://127.0.0.1:8001*`

**Step 4: Generate Evaluation Report (Optional)**
Creates visualization plots in the `models/` folder.

```
python src/generate_plots.py

```

## 🔌 API Usage

The system provides interactive documentation at `http://127.0.0.1:8001/docs`.

### Get Ride Quote

**Endpoint:** `POST /ride/quote`

**Example Request:**

```
{
  "origin_lat": 12.9716,
  "origin_lon": 77.5946,
  "dest_lat": 12.9352,
  "dest_lon": 77.6245,
  "user_preference": "balanced"
}

```

**Example Response:**

```
{
  "request_id": "req_1708156621",
  "user_preference": "balanced",
  "recommendations": [
    {
      "vehicle_type": "economy",
      "eta_minutes": 14.5,
      "price": 245.50,
      "surge_applied": 1.2
    },
    {
      "vehicle_type": "premium",
      "eta_minutes": 14.5,
      "price": 310.00,
      "surge_applied": 1.2
    }
  ]
}

```

##  Evaluation Metrics

* **ETA Model:** Evaluated using Mean Absolute Error (MAE) ~2.5 mins.
* **Demand Model:** Evaluated using RMSE on surge multiplier predictions.
* **Visuals:** Check `models/eta_accuracy_plot.png` for actual vs. predicted performance.


