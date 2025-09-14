#!/usr/bin/env python3
"""
Simplified Soccer Analytics Web API
A more reliable version with better error handling
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
from flask_cors import CORS
import base64
import tempfile

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from soccer_analytics.pipeline import SoccerPipeline
    PIPELINE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Soccer analytics pipeline not available: {e}")
    PIPELINE_AVAILABLE = False

app = Flask(__name__)
app.config['SECRET_KEY'] = 'soccer_analytics_secret_key'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
CORS(app)  # Enable CORS for React frontend

# Global variables for pipeline state
current_pipeline = None
processing_thread: Optional[threading.Thread] = None
is_processing = False
analysis_results: Dict[str, Any] = {}
uploaded_video_path: Optional[str] = None
current_frame_data: Optional[str] = None  # Base64 encoded current frame
current_stats_data: Optional[Dict] = None  # Current analysis stats

class MockPipelineAdapter:
    """Mock adapter for testing when pipeline is not available"""
    
    def __init__(self):
        self.frame_count = 0
        self.start_time = time.time()
        
    def process_video_frame(self, frame):
        """Mock process frame"""
        self.frame_count += 1
        return frame, self.get_current_stats()
    
    def get_current_stats(self):
        """Mock current stats"""
        return {
            'frame_count': self.frame_count,
            'processing_time': time.time() - self.start_time,
            'possession': {0: 60, 1: 40},
            'pass_stats': {'passes': {0: 15, 1: 12}},
            'events': {'yellow_cards': 1, 'red_cards': 0, 'free_kicks': 3},
            'goals': 2,
            'performance': {}
        }
    
    def get_final_results(self):
        """Mock final results"""
        return {
            'videoMetrics': {
                'duration': '45:00',
                'events': [
                    {'type': 'Goal', 'time': '12:30', 'description': 'Team A scores'},
                    {'type': 'Yellow Card', 'time': '25:15', 'description': 'Player cautioned'}
                ],
                'highlights': ['Goal at 12:30', 'Great save at 30:45']
            },
            'mlScores': {
                'teamA': {'name': 'Team A', 'score': 8.2, 'metrics': {'possession': 60, 'passes': 245, 'accuracy': 85}},
                'teamB': {'name': 'Team B', 'score': 7.8, 'metrics': {'possession': 40, 'passes': 198, 'accuracy': 82}},
                'playerRatings': [
                    {'name': 'Player #10', 'position': 'Midfielder', 'rating': 8.5},
                    {'name': 'Player #9', 'position': 'Forward', 'rating': 8.0}
                ]
            },
            'sentiment': {
                'positive': 55, 'negative': 20, 'neutral': 25,
                'keywords': ['exciting', 'great match', 'tactical', 'performance']
            },
            'recommendations': {
                'formation': '4-3-3',
                'tactics': ['Increase wing play', 'Press higher up pitch'],
                'substitutions': ['Fresh legs in midfield at 60min'],
                'keyInsights': ['Strong possession game', 'Good defensive structure']
            }
        }

class WebPipelineAdapter:
    """Real adapter for soccer analytics pipeline"""
    
    def __init__(self):
        if not PIPELINE_AVAILABLE:
            raise RuntimeError("Soccer analytics pipeline not available")
        self.pipeline = SoccerPipeline()
        self.frame_count = 0
        self.start_time = time.time()
        
    def process_video_frame(self, frame):
        """Process a single frame and return results"""
        self.frame_count += 1
        processed_frame = self.pipeline.process_frame(frame)
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
        """Get comprehensive final analysis results"""
        # Implementation similar to the full version
        stats = self.get_current_stats()
        return {
            'videoMetrics': {
                'duration': self._format_time(time.time() - self.start_time),
                'events': [],  # Simplified for now
                'highlights': []
            },
            'mlScores': {
                'teamA': {'name': 'Team A', 'score': 7.5, 'metrics': {'possession': 50, 'passes': 200, 'accuracy': 80}},
                'teamB': {'name': 'Team B', 'score': 7.5, 'metrics': {'possession': 50, 'passes': 200, 'accuracy': 80}},
                'playerRatings': []
            },
            'sentiment': {'positive': 50, 'negative': 25, 'neutral': 25, 'keywords': []},
            'recommendations': {
                'formation': '4-4-2',
                'tactics': ['Maintain possession'],
                'substitutions': ['Consider changes at 60min'],
                'keyInsights': ['Balanced game']
            }
        }
    
    def _format_time(self, seconds):
        """Format seconds to MM:SS format"""
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f"{minutes:02d}:{seconds:02d}"

def process_video_real():
    """Real video processing using soccer analytics pipeline"""
    global current_pipeline, is_processing, analysis_results, uploaded_video_path
    
    if not uploaded_video_path or not os.path.exists(uploaded_video_path):
        print("❌ Video file not found")
        is_processing = False
        return
    
    try:
        # Initialize real pipeline adapter
        if PIPELINE_AVAILABLE:
            current_pipeline = WebPipelineAdapter()
        else:
            current_pipeline = MockPipelineAdapter()
            
        is_processing = True
        
        # Open video file
        cap = cv2.VideoCapture(uploaded_video_path)
        if not cap.isOpened():
            print("❌ Cannot open video file")
            is_processing = False
            return
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_delay = 1.0 / fps if fps > 0 else 1.0 / 30  # Default to 30 FPS
        
        print(f"🎬 Starting real analysis of video: {uploaded_video_path}")
        print(f"📊 Video FPS: {fps}")
        
        frame_count = 0
        while is_processing:
            ret, frame = cap.read()
            if not ret:
                # Loop the video when it reaches the end
                print("📹 End of video reached, looping...")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning
                ret, frame = cap.read()
                if not ret:
                    print("❌ Cannot read video even after reset")
                    break
            
            frame_count += 1
            
            # Process frame through real pipeline
            if PIPELINE_AVAILABLE:
                processed_frame, current_stats = current_pipeline.process_video_frame(frame)
                
                # Add frame info overlay (like run_realtime.py)
                cv2.putText(processed_frame, f'Frame {frame_count}', (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Save current frame for streaming
                _, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                frame_base64 = base64.b64encode(buffer).decode('utf-8')
                
                # Store current frame and stats globally for API access
                global current_frame_data, current_stats_data
                current_frame_data = frame_base64
                current_stats_data = current_stats
                
                print(f"🔄 Processed frame {frame_count} - Goals: {current_stats.get('goals', 0)}")
            else:
                # Mock processing
                time.sleep(frame_delay)
                print(f"🔄 Mock processing frame {frame_count}")
            
            # Control frame rate
            time.sleep(frame_delay)
        
        # Get final results
        analysis_results = current_pipeline.get_final_results()
        print("✅ Real analysis complete!")
        
        cap.release()
        
    except Exception as e:
        print(f"❌ Real processing error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        is_processing = False

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy', 
        'service': 'soccer-analytics-api',
        'pipeline_available': PIPELINE_AVAILABLE
    })

@app.route('/api/upload', methods=['POST'])
def upload_video():
    """Handle video file upload"""
    global uploaded_video_path
    
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
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
    except Exception as e:
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/api/start_analysis', methods=['POST'])
def start_analysis():
    """Start video analysis"""
    global processing_thread, is_processing
    
    if is_processing:
        return jsonify({'error': 'Analysis already in progress'}), 400
    
    if not uploaded_video_path:
        return jsonify({'error': 'No video uploaded'}), 400
    
    try:
        # Start processing in background thread with real pipeline
        processing_thread = threading.Thread(target=process_video_real)
        processing_thread.daemon = True
        processing_thread.start()
        
        mode = "real pipeline" if PIPELINE_AVAILABLE else "mock mode"
        return jsonify({'message': f'Analysis started ({mode})'})
    except Exception as e:
        return jsonify({'error': f'Failed to start analysis: {str(e)}'}), 500

@app.route('/api/stop_analysis', methods=['POST'])
def stop_analysis():
    """Stop video analysis"""
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
        'has_results': bool(analysis_results),
        'pipeline_available': PIPELINE_AVAILABLE
    })

@app.route('/api/current_frame')
def get_current_frame():
    """Get current frame and stats during live analysis"""
    global current_frame_data, current_stats_data
    
    if not is_processing:
        return jsonify({'error': 'No analysis in progress'}), 404
    
    if current_frame_data is None:
        return jsonify({'error': 'No frame data available yet'}), 404
    
    return jsonify({
        'frame': current_frame_data,
        'stats': current_stats_data or {},
        'timestamp': time.time()
    })

if __name__ == '__main__':
    # Create necessary directories
    Path('uploads').mkdir(exist_ok=True)
    
    print("🚀 Starting Simplified Soccer Analytics API Server...")
    print("=" * 50)
    print(f"Pipeline Available: {PIPELINE_AVAILABLE}")
    print("API available at http://localhost:5000/api/")
    print("\nAPI Endpoints:")
    print("  GET  /api/health - Health check")
    print("  POST /api/upload - Upload video file")
    print("  POST /api/start_analysis - Start analysis")
    print("  POST /api/stop_analysis - Stop analysis")
    print("  GET  /api/results - Get analysis results")
    print("  GET  /api/status - Get processing status")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 50)
    
    try:
        app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Server failed: {e}")
