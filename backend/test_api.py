import requests

url = "http://localhost:8000/analyze"
payload = {
    "wing_type": "General Aviation",
    "span": 11.0,
    "ar": 7.2,
    "taper": 0.45,
    "sweep_deg": 3.0,
    "altitude": 2000.0,
    "velocity": 55.0,
    "thickness": 0.12,
    "camber": 0.02
}

try:
    response = requests.post(url, json=payload)
    print("Status Code:", response.status_code)
    print("Response JSON:", response.json())
except Exception as e:
    print("Error:", e)
