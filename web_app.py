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
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import base64
import tempfile

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from soccer_analytics.pipeline import SoccerPipeline

app = Flask(__name__)
app.config['SECRET_KEY'] = 'soccer_analytics_secret_key'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
CORS(app)  # Enable CORS for React frontend
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global variables for pipeline state
current_pipeline: Optional[SoccerPipeline] = None
processing_thread: Optional[threading.Thread] = None
is_processing = False
analysis_results: Dict[str, Any] = {}
uploaded_video_path: Optional[str] = None

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
        return {
            'frame_count': self.frame_count,
            'processing_time': time.time() - self.start_time,
            'possession': self.pipeline.possession.get_possession(),
            'pass_stats': self.pipeline.passes.stats(),
            'events': self.pipeline.events.summary(),
            'goals': self.pipeline.goal_detector.goals() if self.pipeline.goal_detector else 0,
            'performance': self.pipeline.performance.snapshot()
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

def process_video_stream():
    """Process uploaded video in a separate thread"""
    global current_pipeline, is_processing, analysis_results, uploaded_video_path
    
    if not uploaded_video_path or not os.path.exists(uploaded_video_path):
        socketio.emit('error', {'message': 'Video file not found'})
        return
    
    try:
        # Initialize pipeline adapter
        current_pipeline = WebPipelineAdapter()
        is_processing = True
        
        # Open video file
        cap = cv2.VideoCapture(uploaded_video_path)
        if not cap.isOpened():
            socketio.emit('error', {'message': 'Cannot open video file'})
            return
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_delay = 1.0 / fps if fps > 0 else 1.0 / 30  # Default to 30 FPS
        
        socketio.emit('analysis_started', {
            'message': 'Analysis started. Press ESC to stop and view results.',
            'fps': fps
        })
        
        while is_processing:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            processed_frame, current_stats = current_pipeline.process_video_frame(frame)
            
            # Convert frame to base64 for web transmission
            _, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Emit frame and stats to frontend
            socketio.emit('frame_update', {
                'frame': frame_base64,
                'stats': current_stats
            })
            
            # Control frame rate
            time.sleep(frame_delay)
        
        # Get final results
        analysis_results = current_pipeline.get_final_results()
        
        cap.release()
        socketio.emit('analysis_complete', {
            'message': 'Analysis complete. Showing results.',
            'results': analysis_results
        })
        
    except Exception as e:
        socketio.emit('error', {'message': f'Error during processing: {str(e)}'})
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
    
    # Start processing in background thread
    processing_thread = threading.Thread(target=process_video_stream)
    processing_thread.daemon = True
    processing_thread.start()
    
    return jsonify({'message': 'Analysis started'})

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

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    emit('connected', {'message': 'Connected to Soccer Analytics'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    global is_processing
    is_processing = False

@socketio.on('stop_analysis')
def handle_stop_analysis():
    """Handle stop analysis request from frontend"""
    global is_processing
    is_processing = False
    emit('analysis_stopped', {'message': 'Analysis stopped by user'})

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
    
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
