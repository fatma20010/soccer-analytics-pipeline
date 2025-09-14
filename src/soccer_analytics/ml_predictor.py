"""
ML Predictor Module for Soccer Match Outcome Prediction

This module integrates a neural network model to predict match outcomes
(Home Win, Draw, Away Win) based on video analysis insights.
"""

import os
import pickle
import warnings
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging

# Suppress sklearn version warnings
warnings.filterwarnings('ignore', category=UserWarning)

class MatchOutcomePredictor:
    """
    Predicts match outcomes using neural network model and video insights
    """
    
    def __init__(self, model_path: str = '../neural_net_model.pkl'):
        self.model_path = model_path
        self.model = None
        self.is_loaded = False
        self.feature_names = None
        self.logger = logging.getLogger(__name__)
        
        # Expected feature names for the model (based on common soccer analytics)
        self.expected_features = [
            'possession_home', 'possession_away',
            'passes_home', 'passes_away', 
            'pass_accuracy_home', 'pass_accuracy_away',
            'avg_speed_home', 'avg_speed_away',
            'max_speed_home', 'max_speed_away',
            'total_distance_home', 'total_distance_away',
            'ball_touches_home', 'ball_touches_away',
            'goals_home', 'goals_away'
        ]
        
        # Class mapping for predictions
        self.outcome_classes = ['Home Win', 'Draw', 'Away Win']
        
        self._load_model()
    
    def _load_model(self) -> bool:
        """
        Load the neural network model with error handling
        """
        try:
            if not os.path.exists(self.model_path):
                self.logger.warning(f"Model file not found: {self.model_path}")
                return False
            
            # Try multiple loading methods
            loading_methods = [
                self._load_with_pickle,
                self._load_with_joblib,
                self._load_with_tolerance
            ]
            
            for method in loading_methods:
                try:
                    if method():
                        self.is_loaded = True
                        self.logger.info("✅ Neural network model loaded successfully!")
                        return True
                except Exception as e:
                    self.logger.debug(f"Loading method {method.__name__} failed: {e}")
                    continue
            
            self.logger.warning("❌ Could not load neural network model - using fallback predictions")
            return False
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            return False
    
    def _load_with_pickle(self) -> bool:
        """Load using standard pickle"""
        with open(self.model_path, 'rb') as f:
            self.model = pickle.load(f)
        return True
    
    def _load_with_joblib(self) -> bool:
        """Load using joblib"""
        import joblib
        self.model = joblib.load(self.model_path)
        return True
    
    def _load_with_tolerance(self) -> bool:
        """Load with version tolerance"""
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
        return True
    
    def extract_features_from_analysis(self, analysis_data: Dict[str, Any]) -> np.ndarray:
        """
        Extract features from video analysis data for ML prediction
        
        Args:
            analysis_data: Dictionary containing analysis results from soccer pipeline
            
        Returns:
            Feature array ready for model prediction
        """
        try:
            # Initialize feature dictionary
            features = {}
            
            # Extract team metrics if available
            if 'soccerAnalytics' in analysis_data:
                soccer_data = analysis_data['soccerAnalytics']
                
                # Ball touches (home vs away)
                ball_touches = soccer_data.get('ball_touches', {})
                total_touches = sum(ball_touches.values()) if ball_touches else 1
                touches_home = sum(v for k, v in ball_touches.items() if int(k) % 2 == 0) if ball_touches else 0
                touches_away = total_touches - touches_home
                
                features['ball_touches_home'] = touches_home
                features['ball_touches_away'] = touches_away
                
                # Calculate possession percentages
                if total_touches > 0:
                    features['possession_home'] = (touches_home / total_touches) * 100
                    features['possession_away'] = (touches_away / total_touches) * 100
                else:
                    features['possession_home'] = 50.0
                    features['possession_away'] = 50.0
                
                # Speed metrics
                avg_speeds = soccer_data.get('average_speeds', {})
                max_speeds = soccer_data.get('max_speeds', {})
                
                if avg_speeds:
                    speeds_home = [v for k, v in avg_speeds.items() if int(k) % 2 == 0]
                    speeds_away = [v for k, v in avg_speeds.items() if int(k) % 2 == 1]
                    features['avg_speed_home'] = np.mean(speeds_home) if speeds_home else 0.0
                    features['avg_speed_away'] = np.mean(speeds_away) if speeds_away else 0.0
                else:
                    features['avg_speed_home'] = 0.0
                    features['avg_speed_away'] = 0.0
                
                if max_speeds:
                    max_home = [v for k, v in max_speeds.items() if int(k) % 2 == 0]
                    max_away = [v for k, v in max_speeds.items() if int(k) % 2 == 1]
                    features['max_speed_home'] = max(max_home) if max_home else 0.0
                    features['max_speed_away'] = max(max_away) if max_away else 0.0
                else:
                    features['max_speed_home'] = 0.0
                    features['max_speed_away'] = 0.0
                
                # Distance covered
                total_distance = soccer_data.get('total_distance_covered', 0)
                features['total_distance_home'] = total_distance * 0.5  # Rough split
                features['total_distance_away'] = total_distance * 0.5
                
                # Goals detected
                features['goals_home'] = soccer_data.get('goals_detected', 0) * 0.5
                features['goals_away'] = soccer_data.get('goals_detected', 0) * 0.5
            
            # Extract from mlScores if available
            if 'mlScores' in analysis_data:
                ml_scores = analysis_data['mlScores']
                
                # Team metrics
                team_a = ml_scores.get('teamA', {})
                team_b = ml_scores.get('teamB', {})
                
                team_a_metrics = team_a.get('metrics', {})
                team_b_metrics = team_b.get('metrics', {})
                
                # Override with more accurate data if available
                if 'possession' in team_a_metrics:
                    features['possession_home'] = team_a_metrics['possession']
                    features['possession_away'] = team_b_metrics.get('possession', 100 - team_a_metrics['possession'])
                
                if 'passes' in team_a_metrics:
                    features['passes_home'] = team_a_metrics['passes']
                    features['passes_away'] = team_b_metrics.get('passes', 0)
                
                if 'accuracy' in team_a_metrics:
                    features['pass_accuracy_home'] = team_a_metrics['accuracy']
                    features['pass_accuracy_away'] = team_b_metrics.get('accuracy', 0)
            
            # Fill in missing features with defaults
            for feature_name in self.expected_features:
                if feature_name not in features:
                    features[feature_name] = 0.0
            
            # Create feature array in expected order
            feature_array = np.array([features[name] for name in self.expected_features])
            return feature_array.reshape(1, -1)
            
        except Exception as e:
            self.logger.error(f"Error extracting features: {e}")
            # Return default feature array
            return np.zeros((1, len(self.expected_features)))
    
    def predict_match_outcome(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict match outcome probabilities
        
        Args:
            analysis_data: Video analysis results
            
        Returns:
            Dictionary with prediction probabilities and metadata
        """
        try:
            # Extract features from analysis
            features = self.extract_features_from_analysis(analysis_data)
            
            if self.is_loaded and self.model is not None:
                # Use actual model prediction
                try:
                    if hasattr(self.model, 'predict_proba'):
                        probabilities = self.model.predict_proba(features)[0]
                    elif hasattr(self.model, 'predict'):
                        # If only predict is available, convert to probabilities
                        prediction = self.model.predict(features)[0]
                        probabilities = np.zeros(3)
                        probabilities[int(prediction)] = 1.0
                    else:
                        raise AttributeError("Model has no predict method")
                    
                    # Ensure we have 3 classes
                    if len(probabilities) != 3:
                        probabilities = np.array([0.33, 0.34, 0.33])  # Default balanced
                    
                except Exception as e:
                    self.logger.error(f"Model prediction failed: {e}")
                    probabilities = self._fallback_prediction(features)
            else:
                # Use fallback prediction
                probabilities = self._fallback_prediction(features)
            
            # Ensure probabilities sum to 1
            probabilities = probabilities / np.sum(probabilities)
            
            # Create result dictionary
            result = {
                'home_win_probability': float(probabilities[0]),
                'draw_probability': float(probabilities[1]),
                'away_win_probability': float(probabilities[2]),
                'predicted_outcome': self.outcome_classes[np.argmax(probabilities)],
                'confidence': float(np.max(probabilities)),
                'model_used': 'neural_network' if self.is_loaded else 'fallback',
                'features_used': len(self.expected_features)
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in match outcome prediction: {e}")
            return self._default_prediction()
    
    def _fallback_prediction(self, features: np.ndarray) -> np.ndarray:
        """
        Fallback prediction logic based on video analysis features
        """
        try:
            # Simple heuristic based on possession and performance metrics
            feature_dict = dict(zip(self.expected_features, features[0]))
            
            # Calculate home team advantage
            home_score = 0.0
            away_score = 0.0
            
            # Possession factor
            possession_diff = feature_dict['possession_home'] - feature_dict['possession_away']
            home_score += possession_diff * 0.01
            
            # Pass accuracy factor
            accuracy_diff = feature_dict['pass_accuracy_home'] - feature_dict['pass_accuracy_away']
            home_score += accuracy_diff * 0.005
            
            # Speed factor (higher average speed might indicate better play)
            speed_diff = feature_dict['avg_speed_home'] - feature_dict['avg_speed_away']
            home_score += speed_diff * 0.1
            
            # Goals factor
            goal_diff = feature_dict['goals_home'] - feature_dict['goals_away']
            home_score += goal_diff * 0.3
            
            # Convert to probabilities
            if home_score > 0.1:
                return np.array([0.5, 0.25, 0.25])  # Home favored
            elif home_score < -0.1:
                return np.array([0.25, 0.25, 0.5])  # Away favored
            else:
                return np.array([0.3, 0.4, 0.3])    # Draw favored
                
        except Exception:
            return np.array([0.33, 0.34, 0.33])  # Default balanced
    
    def _default_prediction(self) -> Dict[str, Any]:
        """
        Default prediction when everything fails
        """
        return {
            'home_win_probability': 0.33,
            'draw_probability': 0.34,
            'away_win_probability': 0.33,
            'predicted_outcome': 'Draw',
            'confidence': 0.34,
            'model_used': 'default',
            'features_used': 0
        }
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance if available from the model
        """
        if not self.is_loaded or self.model is None:
            return {}
        
        try:
            if hasattr(self.model, 'feature_importances_'):
                importance = self.model.feature_importances_
                return dict(zip(self.expected_features, importance))
            elif hasattr(self.model, 'coef_'):
                # For linear models
                importance = np.abs(self.model.coef_[0]) if len(self.model.coef_.shape) > 1 else np.abs(self.model.coef_)
                return dict(zip(self.expected_features, importance))
            else:
                return {}
        except Exception as e:
            self.logger.error(f"Error getting feature importance: {e}")
            return {}
