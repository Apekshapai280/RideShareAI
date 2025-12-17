import requests

def test_api():
    url = "http://127.0.0.1:8000/ride/quote"
    payload = {
        "origin_lat": 12.97, "origin_lon": 77.59,
        "dest_lat": 12.99, "dest_lon": 77.60,
        "user_preference": "balanced"
    }
    response = requests.post(url, json=payload)
    assert response.status_code == 200
    print("API Test Passed!")
    print(response.json())

if __name__ == "__main__":
    test_api()