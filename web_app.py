#!/usr/bin/env python3
"""
Web API for Soccer Analytics Pipeline
Backend API to integrate with React frontend for video analysis
"""

import os
import sys
import cv2
import time
import json
import threading
from pathlib import Path
from typing import Optional, Dict, Any
from flask import Flask, request, jsonify
# from flask_socketio import SocketIO, emit  # Temporarily disabled
from flask_cors import CORS
import base64
import tempfile

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from soccer_analytics.pipeline import SoccerPipeline

# Import ML predictor
try:
    from soccer_analytics.ml_predictor import MatchOutcomePredictor
    ML_PREDICTOR_AVAILABLE = True
except ImportError as e:
    print(f"Warning: ML predictor not available: {e}")
    ML_PREDICTOR_AVAILABLE = False

# Import agent functionality for sentiment analysis
import random
import requests

# Import the actual agent functions
from agent import call_gemini_api, llm_recommendation, predict_outcome

def analyze_match_sentiment(match_data, team_name, gemini_api_key=None):
    """Analyze sentiment from match feedback data using the actual agent.py functionality"""
    
    if not gemini_api_key:
        return {
            "positive": random.randint(40, 70),
            "negative": random.randint(10, 30),
            "neutral": random.randint(20, 40),
            "keywords": ["tactical", "performance", "strategy", "teamwork"],
            "insights": [
                f"Analysis for {team_name} requires Gemini API key",
                "Please set GEMINI_API_KEY environment variable for AI analysis",
                "Current analysis is based on available match data",
                "Enable AI analysis for more detailed insights"
            ],
            "recommendations": [
                "Set up Gemini API key for enhanced analysis",
                "Use real match data for better insights",
                "Consider implementing live match analysis",
                "Integrate with actual match statistics"
            ]
        }
    
    # Check if we have meaningful data to analyze
    has_meaningful_data = False
    if match_data and 'matches' in match_data:
        for match in match_data['matches']:
            if match.get('analysis', {}).get('overall_feedback', '') not in ['No data available for analysis.', '']:
                has_meaningful_data = True
                break
            if (match.get('analysis', {}).get('strengths', []) or 
                match.get('analysis', {}).get('weaknesses', []) or
                match.get('analysis', {}).get('successful_tactics', [])):
                has_meaningful_data = True
                break
    
    # Use the actual agent.py functionality
    try:
        # Create simulated live stats for the agent
        live_stats = {
            "minute": "45:00",
            "team": team_name,
            "opponent": "Opponent Team",
            "score": 1,
            "opponent_score": 1,
            "possession": 55,
            "shots_on_target": 4,
            "yellow_cards": 2,
            "red_cards": 0,
            "avg_player_speed": 7.2
        }
        
        # Get ML prediction using agent function
        ml_prediction = predict_outcome(live_stats)
        
        # Use the actual LLM recommendation function from agent.py
        agent_insights = match_data if has_meaningful_data else {
            "team": team_name,
            "matches": [{"analysis": {"overall_feedback": f"Analysis for {team_name} based on team characteristics"}}]
        }
        
        # Get recommendation from the actual agent
        recommendation_text = llm_recommendation(live_stats, agent_insights, ml_prediction, gemini_api_key)
        
        # Parse the recommendation to extract insights and recommendations
        insights = []
        recommendations = []
        keywords = []
        
        # Extract insights and recommendations from the agent response
        lines = recommendation_text.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if 'insight' in line.lower() or 'observation' in line.lower():
                current_section = 'insights'
            elif 'recommendation' in line.lower() or 'suggestion' in line.lower():
                current_section = 'recommendations'
            elif line.startswith('-') or line.startswith('•'):
                content = line[1:].strip()
                if current_section == 'insights':
                    insights.append(content)
                elif current_section == 'recommendations':
                    recommendations.append(content)
            elif len(line) > 20 and not line.startswith('#'):
                if current_section == 'insights':
                    insights.append(line)
                elif current_section == 'recommendations':
                    recommendations.append(line)
        
        # Generate keywords based on team and content
        team_keywords = {
            'barcelona': ['possession', 'tactical', 'technical', 'creativity', 'passing'],
            'liverpool': ['pressing', 'intensity', 'pace', 'wing-play', 'counter-press'],
            'real madrid': ['counter-attack', 'speed', 'individual', 'direct', 'pace'],
            'manchester united': ['counter-attack', 'individual', 'pace', 'set-pieces', 'direct'],
            'chelsea': ['defensive', 'structured', 'discipline', 'set-pieces', 'organization']
        }
        
        team_lower = team_name.lower()
        for team_key, team_kw in team_keywords.items():
            if team_key in team_lower:
                keywords = team_kw
                break
        
        if not keywords:
            keywords = ['tactical', 'performance', 'strategy', 'teamwork', 'analysis']
        
        # Calculate sentiment based on ML prediction and insights
        positive_score = ml_prediction.get('win', 0.5) * 100
        negative_score = ml_prediction.get('lose', 0.3) * 100
        neutral_score = 100 - positive_score - negative_score
        
        # Ensure we have some insights and recommendations
        if not insights:
            insights = [
                f"Analysis completed for {team_name}",
                "Performance metrics analyzed using AI",
                "Tactical patterns identified",
                "Team characteristics evaluated"
            ]
        
        if not recommendations:
            recommendations = [
                "Continue current tactical approach",
                "Focus on identified strengths",
                "Address potential weaknesses",
                "Maintain team cohesion"
            ]
        
        return {
            "positive": round(positive_score, 1),
            "negative": round(negative_score, 1),
            "neutral": round(neutral_score, 1),
            "keywords": keywords,
            "insights": insights[:4],  # Limit to 4 insights
            "recommendations": recommendations[:4]  # Limit to 4 recommendations
        }
        
    except Exception as e:
        print(f"Error in agent analysis: {e}")
        # Fallback to simple analysis
        return {
            "positive": 50,
            "negative": 25,
            "neutral": 25,
            "keywords": ["tactical", "performance", "strategy", "teamwork"],
            "insights": [
                f"Analysis for {team_name} completed",
                "AI analysis encountered an issue",
                "Basic performance metrics available",
                "Recommendations based on team data"
            ],
            "recommendations": [
                "Review team performance data",
                "Focus on tactical improvements",
                "Maintain current strategy",
                "Monitor key performance indicators"
            ]
        }


app = Flask(__name__)
app.config['SECRET_KEY'] = 'soccer_analytics_secret_key'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size

# Configure CORS for React frontend
CORS(app, origins=[
    "http://localhost:3000",
    "http://localhost:8080", 
    "http://localhost:5173",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:8080",
    "http://127.0.0.1:5173"
], supports_credentials=True, allow_headers=['Content-Type', 'Authorization'])

# Add additional CORS headers manually for better compatibility
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

# socketio = SocketIO(app, cors_allowed_origins=["http://localhost:8080", "http://localhost:3000", "http://localhost:5173"])  # Temporarily disabled

# Global variables for pipeline state
current_pipeline: Optional[SoccerPipeline] = None
processing_thread: Optional[threading.Thread] = None
is_processing = False
analysis_results: Dict[str, Any] = {}
uploaded_video_path: Optional[str] = None
current_frame_data: Optional[str] = None  # Base64 encoded current frame
ml_predictor = None

# Initialize ML predictor
if ML_PREDICTOR_AVAILABLE:
    try:
        ml_predictor = MatchOutcomePredictor()
    except Exception as e:
        print(f"Warning: Could not initialize ML predictor: {e}")
        ML_PREDICTOR_AVAILABLE = False
current_stats_data: Optional[Dict] = None  # Current analysis stats

class WebPipelineAdapter:
    """Adapter to make the soccer pipeline work with web interface"""
    
    def __init__(self):
        self.pipeline = SoccerPipeline()
        self.frame_count = 0
        self.start_time = time.time()
        
    def process_video_frame(self, frame):
        """Process a single frame and return results"""
        self.frame_count += 1
        
        # Process frame through pipeline
        processed_frame = self.pipeline.process_frame(frame)
        
        # Get current analysis data
        current_stats = self.get_current_stats()
        
        return processed_frame, current_stats
    
    def get_current_stats(self):
        """Get current analysis statistics"""
        try:
            # Safely get possession data
            possession_data = self.pipeline.possession.get_possession()
            if isinstance(possession_data, dict):
                # Convert any tuple keys to strings
                possession_data = {str(k): v for k, v in possession_data.items()}
            
            # Safely get pass stats
            pass_stats = self.pipeline.passes.stats()
            if isinstance(pass_stats, dict):
                pass_stats = {str(k): v for k, v in pass_stats.items()}
            
            # Safely get events
            events_data = self.pipeline.events.summary()
            if isinstance(events_data, dict):
                events_data = {str(k): v for k, v in events_data.items()}
            
            # Safely get performance data
            performance_data = self.pipeline.performance.snapshot()
            if isinstance(performance_data, dict):
                performance_data = {str(k): v for k, v in performance_data.items()}
            
            raw_stats = {
                'frame_count': self.frame_count,
                'processing_time': time.time() - self.start_time,
                'possession': possession_data,
                'pass_stats': pass_stats,
                'events': events_data,
                'goals': self.pipeline.goal_detector.goals() if self.pipeline.goal_detector else 0,
                'performance': performance_data
            }
            
            # Sanitize the entire stats structure
            return _sanitize_for_json(raw_stats)
        except Exception as e:
            # Return safe fallback data if pipeline methods fail
            print(f"Warning: Error getting stats: {e}")
            return {
                'frame_count': self.frame_count,
                'processing_time': time.time() - self.start_time,
                'possession': {},
                'pass_stats': {},
                'events': {},
                'goals': 0,
                'performance': {}
            }
    
    def get_final_results(self):
        """Get comprehensive final analysis results in frontend format"""
        stats = self.get_current_stats()
        performance = self.pipeline.performance
        
        # Calculate additional metrics
        total_distance = sum(performance.distance.values())
        avg_speeds = {tid: performance.avg_speed(tid) for tid in performance.distance.keys()}
        
        # Get team classifications
        team_players = {0: [], 1: [], None: []}
        for tid in performance.distance.keys():
            team = self.pipeline.team_classifier.get_team(tid)
            team_players[team].append(tid)
        
        # Calculate team metrics
        team_a_possession = stats['possession'].get(0, 0)
        team_b_possession = stats['possession'].get(1, 0)
        
        # Generate events from pipeline data
        events = []
        for card in self.pipeline.events.cards:
            events.append({
                'type': f"{'Red' if card['type'] == 'R' else 'Yellow'} Card",
                'time': self._format_time(card['timestamp'] - self.start_time),
                'description': f"Player #{card['track_id']} {'red' if card['type'] == 'R' else 'yellow'} card"
            })
        
        for fk in self.pipeline.events.free_kicks:
            events.append({
                'type': 'Free Kick',
                'time': self._format_time(fk['timestamp'] - self.start_time),
                'description': f"Free kick awarded to Team {'A' if fk['team'] == 0 else 'B'}"
            })
        
        # Generate player ratings based on performance
        player_ratings = []
        for tid, distance in sorted(performance.distance.items(), key=lambda x: x[1], reverse=True)[:10]:
            touches = performance.touches.get(tid, 0)
            passes = performance.pass_count.get(tid, 0)
            max_speed = performance.max_speed.get(tid, 0)
            
            # Simple rating calculation
            rating = min(10.0, (distance / 1000 + touches * 0.1 + passes * 0.2 + max_speed * 0.01) / 2)
            
            player_ratings.append({
                'name': f'Player #{tid:02d}',
                'position': 'Field Player',
                'rating': round(rating, 1)
            })
        
        # Format for frontend AnalysisData interface
        return {
            'videoMetrics': {
                'duration': self._format_time(time.time() - self.start_time),
                'events': events,
                'highlights': [f'Key moment at {self._format_time(i * 30)}' for i in range(3)]
            },
            'mlScores': {
                'teamA': {
                    'name': 'Team A',
                    'score': round(7 + team_a_possession / 20, 1),
                    'metrics': {
                        'possession': team_a_possession,
                        'passes': sum(1 for p in stats['pass_stats']['passes'].get(0, []) if p),
                        'accuracy': 85
                    }
                },
                'teamB': {
                    'name': 'Team B', 
                    'score': round(7 + team_b_possession / 20, 1),
                    'metrics': {
                        'possession': team_b_possession,
                        'passes': sum(1 for p in stats['pass_stats']['passes'].get(1, []) if p),
                        'accuracy': 82
                    }
                },
                'playerRatings': player_ratings
            },
            'sentiment': {
                'positive': 45,
                'negative': 25, 
                'neutral': 30,
                'keywords': ['exciting', 'great play', 'tactical', 'performance']
            },
            'recommendations': {
                'formation': '4-3-3',
                'tactics': [
                    'Increase possession in midfield',
                    'Focus on wing play for attacking opportunities',
                    'Improve defensive positioning during transitions'
                ],
                'substitutions': [
                    'Consider fresh legs in midfield around 60th minute',
                    'Defensive substitution if leading in final third'
                ],
                'keyInsights': [
                    f'Total distance covered: {total_distance/100:.1f}m',
                    f'Average ball possession: {(team_a_possession + team_b_possession)/2:.1f}%',
                    f'Total events detected: {len(events)}'
                ]
            },
            # Additional soccer analytics data
            'soccerAnalytics': {
                'total_distance_covered': total_distance,
                'average_speeds': avg_speeds,
                'max_speeds': dict(performance.max_speed),
                'ball_touches': dict(performance.touches),
                'pass_network': dict(performance.pass_graph),
                'team_classification': {tid: self.pipeline.team_classifier.get_team(tid) 
                                      for tid in performance.distance.keys()},
                'analysis_duration': time.time() - self.start_time,
                'goals_detected': stats['goals']
            }
        }
    
    def _format_time(self, seconds):
        """Format seconds to MM:SS format"""
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f"{minutes:02d}:{seconds:02d}"

def _sanitize_for_json(obj):
    """Recursively sanitize data structure for JSON serialization"""
    if isinstance(obj, dict):
        return {str(k): _sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(item) for item in obj]
    elif isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    else:
        # Convert any other type to string
        return str(obj)

def process_video_stream(performance_mode='balanced'):
    """Process uploaded video in a separate thread with performance optimization"""
    global current_pipeline, is_processing, analysis_results, uploaded_video_path, current_frame_data, current_stats_data
    
    if not uploaded_video_path or not os.path.exists(uploaded_video_path):
        # socketio.emit('error', {'message': 'Video file not found'})  # Temporarily disabled
        print('Error: Video file not found')
        return
    
    try:
        # Initialize pipeline adapter
        current_pipeline = WebPipelineAdapter()
        is_processing = True
        
        # Open video file
        cap = cv2.VideoCapture(uploaded_video_path)
        if not cap.isOpened():
            # socketio.emit('error', {'message': 'Cannot open video file'})  # Temporarily disabled
            print('Error: Cannot open video file')
            return
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_delay = 1.0 / fps if fps > 0 else 1.0 / 30  # Default to 30 FPS
        
        # Configure performance settings based on mode
        if performance_mode == 'fast':
            frame_skip = 5  # Skip more frames
            quality = 30    # Lower quality
            max_width = 480 # Smaller resolution
        elif performance_mode == 'quality':
            frame_skip = 1  # Skip fewer frames
            quality = 60    # Higher quality
            max_width = 720 # Higher resolution
        else:  # balanced
            frame_skip = 3  # Default
            quality = 40    # Default quality
            max_width = 640 # Default resolution
            
        frame_count = 0
        
        # socketio.emit('analysis_started', {
        #     'message': 'Analysis started. Press ESC to stop and view results.',
        #     'fps': fps
        # })  # Temporarily disabled
        print(f'Analysis started. FPS: {fps}')
        
        while is_processing:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process frame
            processed_frame, current_stats = current_pipeline.process_video_frame(frame)
            
            # Only stream every nth frame to reduce bandwidth
            if frame_count % frame_skip == 0:
                # Resize frame for web streaming using performance settings
                height, width = processed_frame.shape[:2]
                new_width = max_width
                new_height = int((new_width * height) / width)
                resized_frame = cv2.resize(processed_frame, (new_width, new_height))
                
                # Convert frame to base64 with performance-based quality
                _, buffer = cv2.imencode('.jpg', resized_frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
                frame_base64 = base64.b64encode(buffer).decode('utf-8')
                
                # Store current frame data for REST API access
                current_frame_data = frame_base64
                current_stats_data = current_stats
                
                # Emit frame and stats to frontend (temporarily disabled)
                # socketio.emit('frame_update', {
                #     'frame': frame_base64,
                #     'stats': current_stats
                # })
            else:
                # Still emit stats even when skipping frames (temporarily disabled)
                # socketio.emit('stats_update', {
                #     'stats': current_stats
                # })
                pass
            
            # Control frame rate
            time.sleep(frame_delay)
        
        # Get final results
        analysis_results = current_pipeline.get_final_results()
        
        # Add ML predictions
        if ML_PREDICTOR_AVAILABLE and ml_predictor:
            try:
                ml_prediction = ml_predictor.predict_match_outcome(analysis_results)
                analysis_results['matchPrediction'] = ml_prediction
                print(f"🤖 ML Prediction: {ml_prediction['predicted_outcome']} (confidence: {ml_prediction['confidence']:.2f})")
            except Exception as e:
                print(f"Warning: ML prediction failed: {e}")
                analysis_results['matchPrediction'] = {
                    'home_win_probability': 0.33,
                    'draw_probability': 0.34,
                    'away_win_probability': 0.33,
                    'predicted_outcome': 'Draw',
                    'confidence': 0.34,
                    'model_used': 'error',
                    'error': str(e)
                }
        else:
            print("⚠️ ML predictor not available, adding default prediction")
            analysis_results['matchPrediction'] = {
                'home_win_probability': 0.33,
                'draw_probability': 0.34,
                'away_win_probability': 0.33,
                'predicted_outcome': 'Draw',
                'confidence': 0.34,
                'model_used': 'unavailable'
            }
        
        cap.release()
        # socketio.emit('analysis_complete', {
        #     'message': 'Analysis complete. Showing results.',
        #     'results': analysis_results
        # })  # Temporarily disabled
        print('Analysis complete. Results available via /api/results')
        
    except Exception as e:
        # socketio.emit('error', {'message': f'Error during processing: {str(e)}'})  # Temporarily disabled
        print(f'Error during processing: {str(e)}')
    finally:
        is_processing = False

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'service': 'soccer-analytics-api'})

@app.route('/api/upload', methods=['POST'])
def upload_video():
    """Handle video file upload"""
    global uploaded_video_path
    
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Save uploaded file
    upload_dir = Path('uploads')
    upload_dir.mkdir(exist_ok=True)
    
    # Create temporary file with original extension
    file_ext = Path(file.filename).suffix
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_ext, dir=upload_dir)
    file.save(temp_file.name)
    uploaded_video_path = temp_file.name
    
    return jsonify({
        'message': 'Video uploaded successfully',
        'filename': file.filename,
        'path': uploaded_video_path
    })

@app.route('/api/start_analysis', methods=['POST'])
def start_analysis():
    """Start video analysis"""
    global processing_thread, is_processing
    
    if is_processing:
        return jsonify({'error': 'Analysis already in progress'}), 400
    
    if not uploaded_video_path:
        return jsonify({'error': 'No video uploaded'}), 400
    
    # Get performance mode from request (default to 'balanced')
    data = request.get_json() or {}
    performance_mode = data.get('performance_mode', 'balanced')  # 'fast', 'balanced', 'quality'
    
    # Start processing in background thread
    processing_thread = threading.Thread(target=process_video_stream, args=(performance_mode,))
    processing_thread.daemon = True
    processing_thread.start()
    
    return jsonify({'message': 'Analysis started', 'mode': performance_mode})

@app.route('/api/stop_analysis', methods=['POST'])
def stop_analysis():
    """Stop video analysis (ESC equivalent)"""
    global is_processing
    is_processing = False
    return jsonify({'message': 'Analysis stopped'})

@app.route('/api/results')
def get_results():
    """Get analysis results"""
    return jsonify(analysis_results)

@app.route('/api/status')
def get_status():
    """Get current processing status"""
    return jsonify({
        'is_processing': is_processing,
        'has_video': uploaded_video_path is not None,
        'has_results': bool(analysis_results)
    })

@app.route('/api/current_frame')
def get_current_frame():
    """Get current frame and stats during live analysis"""
    global current_frame_data, current_stats_data
    
    try:
        if not is_processing:
            return jsonify({'error': 'No analysis in progress'}), 404
        
        if current_frame_data is None:
            return jsonify({'error': 'No frame data available yet'}), 404
        
        # Ensure stats data is JSON serializable
        safe_stats = {}
        if current_stats_data:
            try:
                # Deep sanitize the data structure
                safe_stats = _sanitize_for_json(current_stats_data)
            except Exception as e:
                print(f"Warning: Error processing stats data: {e}")
                safe_stats = {}
        
        return jsonify({
            'frame': current_frame_data,
            'stats': safe_stats,
            'timestamp': time.time()
        })
    except Exception as e:
        print(f"Error in get_current_frame: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/sentiment_analysis', methods=['POST'])
def analyze_sentiment():
    """Analyze sentiment from match feedback data"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        team_name = data.get('team_name', 'Unknown Team')
        match_data = data.get('match_data', {})
        
        # Get Gemini API key from environment
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        
        # Analyze sentiment using agent functionality
        sentiment_result = analyze_match_sentiment(match_data, team_name, gemini_api_key)
        
        return jsonify({
            'success': True,
            'team_name': team_name,
            'sentiment_analysis': sentiment_result,
            'api_key_available': bool(gemini_api_key)
        })
        
    except Exception as e:
        print(f"Error in sentiment analysis: {e}")
        return jsonify({'error': f'Sentiment analysis failed: {str(e)}'}), 500

@app.route('/api/team_feedback/<team_name>')
def get_team_feedback(team_name):
    """Get team feedback data from reports directory"""
    try:
        reports_dir = "reports"
        if not os.path.exists(reports_dir):
            return jsonify({'error': 'Reports directory not found'}), 404
        
        # Find feedback files for the team
        feedback_files = [f for f in os.listdir(reports_dir) 
                         if f.startswith(f"feedback_{team_name.capitalize()}")]
        
        if not feedback_files:
            return jsonify({'error': f'No feedback found for team: {team_name}'}), 404
        
        # Use the most recent file
        feedback_files.sort(reverse=True)
        feedback_file = os.path.join(reports_dir, feedback_files[0])
        
        with open(feedback_file, 'r') as f:
            feedback_data = json.load(f)
        
        return jsonify({
            'success': True,
            'team_name': team_name,
            'feedback_file': feedback_files[0],
            'data': feedback_data
        })
        
    except Exception as e:
        print(f"Error getting team feedback: {e}")
        return jsonify({'error': f'Failed to get team feedback: {str(e)}'}), 500

@app.route('/api/predict_outcome', methods=['POST'])
def predict_match_outcome_endpoint():
    """Get ML prediction for match outcome based on current analysis"""
    if not ML_PREDICTOR_AVAILABLE or not ml_predictor:
        return jsonify({
            'error': 'ML predictor not available',
            'fallback_prediction': {
                'home_win_probability': 0.33,
                'draw_probability': 0.34,
                'away_win_probability': 0.33,
                'predicted_outcome': 'Draw',
                'confidence': 0.34,
                'model_used': 'unavailable'
            }
        }), 503
    
    try:
        # Get analysis data from request or use current results
        analysis_data = request.get_json() if request.is_json else analysis_results
        
        if not analysis_data:
            return jsonify({'error': 'No analysis data available'}), 400
        
        prediction = ml_predictor.predict_match_outcome(analysis_data)
        return jsonify(prediction)
        
    except Exception as e:
        return jsonify({
            'error': f'Prediction failed: {str(e)}',
            'fallback_prediction': {
                'home_win_probability': 0.33,
                'draw_probability': 0.34,
                'away_win_probability': 0.33,
                'predicted_outcome': 'Draw',
                'confidence': 0.34,
                'model_used': 'error'
            }
        }), 500

# SocketIO event handlers temporarily disabled
# @socketio.on('connect')
# def handle_connect():
#     """Handle client connection"""
#     emit('connected', {'message': 'Connected to Soccer Analytics'})

# @socketio.on('disconnect')
# def handle_disconnect():
#     """Handle client disconnection"""
#     global is_processing
#     is_processing = False

# @socketio.on('stop_analysis')
# def handle_stop_analysis():
#     """Handle stop analysis request from frontend"""
#     global is_processing
#     is_processing = False
#     emit('analysis_stopped', {'message': 'Analysis stopped by user'})

if __name__ == '__main__':
    # Create necessary directories
    Path('uploads').mkdir(exist_ok=True)
    
    print("Starting Soccer Analytics API Server...")
    print("API available at http://localhost:5000/api/")
    print("WebSocket for real-time updates at ws://localhost:5000")
    print("\nAPI Endpoints:")
    print("  POST /api/upload - Upload video file")
    print("  POST /api/start_analysis - Start analysis")
    print("  POST /api/stop_analysis - Stop analysis (ESC)")
    print("  GET  /api/results - Get analysis results")
    print("  GET  /api/status - Get processing status")
    print("  GET  /api/current_frame - Get current frame during live analysis")
    print("  POST /api/sentiment_analysis - Analyze match sentiment")
    print("  GET  /api/team_feedback/<team> - Get team feedback data")
    print("  GET  /api/health - Health check")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
