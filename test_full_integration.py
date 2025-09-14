#!/usr/bin/env python3
"""
Test complete ML integration end-to-end
"""
import requests
import json

def test_complete_integration():
    print("🧪 Testing Complete ML Integration")
    print("=" * 50)
    
    try:
        # 1. Test server health
        health = requests.get('http://localhost:5000/api/health')
        assert health.status_code == 200
        print("✅ Server health check passed")
        
        # 2. Test prediction endpoint directly
        test_data = {
            'mlScores': {
                'teamA': {
                    'name': 'Team A',
                    'score': 7.5,
                    'metrics': {'possession': 65, 'accuracy': 85, 'passes': 450}
                },
                'teamB': {
                    'name': 'Team B', 
                    'score': 6.2,
                    'metrics': {'possession': 35, 'accuracy': 78, 'passes': 320}
                }
            },
            'soccerAnalytics': {
                'total_distance_covered': 15000,
                'goals_detected': 2
            }
        }
        
        pred_response = requests.post('http://localhost:5000/api/predict_outcome', 
                                     json=test_data)
        assert pred_response.status_code == 200
        prediction = pred_response.json()
        
        print("✅ Prediction endpoint works")
        print(f"   Predicted: {prediction['predicted_outcome']}")
        print(f"   Confidence: {prediction['confidence']:.2f}")
        print(f"   Model: {prediction['model_used']}")
        
        # 3. Verify prediction format
        required_fields = ['home_win_probability', 'draw_probability', 'away_win_probability', 
                          'predicted_outcome', 'confidence', 'model_used']
        for field in required_fields:
            assert field in prediction, f"Missing field: {field}"
        print("✅ Prediction format is correct")
        
        # 4. Verify probabilities sum to ~1
        total_prob = prediction['home_win_probability'] + prediction['draw_probability'] + prediction['away_win_probability']
        assert abs(total_prob - 1.0) < 0.01, f"Probabilities don't sum to 1: {total_prob}"
        print("✅ Probabilities are valid")
        
        print("\n" + "=" * 50)
        print("🎉 All integration tests passed!")
        print("\n📋 Next Steps:")
        print("1. In your React app, upload a video")
        print("2. Start analysis (use 'Quick Analysis' for testing)")
        print("3. Look for the 'Match Prediction' tab in results")
        print("4. You should see predictions like:")
        print(f"   '{prediction['predicted_outcome']}' with {prediction['confidence']*100:.1f}% confidence")
        
        return True
        
    except AssertionError as e:
        print(f"❌ Test failed: {e}")
        return False
    except requests.exceptions.ConnectionError:
        print("❌ Server not running. Start with: python web_app.py")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == '__main__':
    test_complete_integration()
