import requests
import json
import time
import random

url = "http://localhost:8000/predict"

# 11 input features corresponding to wine characteristics
# fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides,
# free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol
features_samples = [
    [7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4],
    [7.8, 0.88, 0.0, 2.6, 0.098, 25.0, 67.0, 0.9968, 3.2, 0.68, 9.8],
    [7.8, 0.76, 0.04, 2.3, 0.092, 15.0, 54.0, 0.997, 3.26, 0.65, 9.8],
    [11.2, 0.28, 0.56, 1.9, 0.075, 17.0, 60.0, 0.998, 3.16, 0.58, 9.8],
    [7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4],
    [7.4, 0.66, 0.0, 1.8, 0.075, 13.0, 40.0, 0.9978, 3.51, 0.56, 9.4]
]

print("Starting sending inference requests to generate metrics...")
for i in range(50):
    sample = random.choice(features_samples)
    
    # Introduce random perturbations to mimic real usage
    perturbed_sample = [val + random.uniform(-val * 0.05, val * 0.05) for val in sample]
    
    payload = {"features": perturbed_sample}
    
    try:
        response = requests.post(url, json=payload)
        print(f"Request {i+1} status code: {response.status_code}, prediction: {response.json().get('prediction')}")
    except Exception as e:
        print(f"Request {i+1} failed: {e}")
        
    time.sleep(random.uniform(0.1, 0.5))

print("Completed sending inference requests!")
