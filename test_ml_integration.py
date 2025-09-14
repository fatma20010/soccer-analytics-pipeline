#!/usr/bin/env python3
"""
Test script for ML prediction integration
Tests the new ML predictor functionality without requiring the actual model file
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from soccer_analytics.ml_predictor import MatchOutcomePredictor

def test_ml_predictor():
    """Test the ML predictor with mock data"""
    print("🧪 Testing ML Predictor Integration")
    print("=" * 50)
    
    # Initialize predictor
    predictor = MatchOutcomePredictor()
    print(f"✅ Predictor initialized (model loaded: {predictor.is_loaded})")
    
    # Create mock analysis data similar to what the pipeline produces
    mock_analysis_data = {
        'mlScores': {
            'teamA': {
                'name': 'Home Team',
                'score': 7.5,
                'metrics': {
                    'possession': 65,
                    'accuracy': 85,
                    'passes': 450
                }
            },
            'teamB': {
                'name': 'Away Team', 
                'score': 6.2,
                'metrics': {
                    'possession': 35,
                    'accuracy': 78,
                    'passes': 320
                }
            }
        },
        'soccerAnalytics': {
            'total_distance_covered': 15000,
            'average_speeds': {'1': 2.5, '2': 3.1, '3': 2.8, '4': 2.9},
            'max_speeds': {'1': 8.5, '2': 9.2, '3': 7.8, '4': 8.9},
            'ball_touches': {'1': 45, '2': 23, '3': 38, '4': 19},
            'goals_detected': 2
        }
    }
    
    # Test feature extraction
    print("\n🔍 Testing feature extraction...")
    features = predictor.extract_features_from_analysis(mock_analysis_data)
    print(f"✅ Features extracted: shape {features.shape}")
    print(f"   Feature sample: {features[0][:5]}...")
    
    # Test prediction
    print("\n🎯 Testing match outcome prediction...")
    prediction = predictor.predict_match_outcome(mock_analysis_data)
    
    print(f"✅ Prediction generated successfully!")
    print(f"   Predicted outcome: {prediction['predicted_outcome']}")
    print(f"   Home win probability: {prediction['home_win_probability']:.3f}")
    print(f"   Draw probability: {prediction['draw_probability']:.3f}")
    print(f"   Away win probability: {prediction['away_win_probability']:.3f}")
    print(f"   Confidence: {prediction['confidence']:.3f}")
    print(f"   Model used: {prediction['model_used']}")
    
    # Test with empty data
    print("\n🔄 Testing with empty data...")
    empty_prediction = predictor.predict_match_outcome({})
    print(f"✅ Empty data handled: {empty_prediction['predicted_outcome']}")
    
    print("\n" + "=" * 50)
    print("🎉 All tests passed! ML predictor is ready for integration.")
    
    return True

if __name__ == '__main__':
    try:
        test_ml_predictor()
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
