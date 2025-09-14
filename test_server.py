#!/usr/bin/env python3
"""
Test if the backend server is running
"""
import requests
import json

try:
    # Test health endpoint
    response = requests.get('http://localhost:5000/api/health')
    print(f"✅ Server is running! Status: {response.status_code}")
    print(f"Response: {response.json()}")
    
    # Test prediction endpoint
    test_data = {
        'mlScores': {
            'teamA': {'metrics': {'possession': 60, 'accuracy': 85}},
            'teamB': {'metrics': {'possession': 40, 'accuracy': 78}}
        }
    }
    
    pred_response = requests.post('http://localhost:5000/api/predict_outcome', 
                                 json=test_data)
    print(f"\n🤖 Prediction endpoint test: {pred_response.status_code}")
    if pred_response.status_code == 200:
        prediction = pred_response.json()
        print(f"Prediction: {prediction}")
    else:
        print(f"Error: {pred_response.text}")
        
except requests.exceptions.ConnectionError:
    print("❌ Server is not running. Start it with: python simple_web_app.py")
except Exception as e:
    print(f"❌ Error: {e}")
