# Soccer Analytics Web Integration

This document explains how to integrate the Soccer Analytics Pipeline with your React frontend.

## 🏗️ Architecture Overview

The integration consists of:

1. **Python Backend API** (`web_app.py`) - Flask + SocketIO server
2. **React Frontend Service** (`src/services/soccerAnalyticsApi.ts`) - API client
3. **Live Analysis Component** (`src/components/analysis/LiveAnalysis.tsx`) - Real-time video processing
4. **Updated Analyze Page** (`src/pages/AnalyzePage.tsx`) - Integration with existing UI

## 🚀 Quick Start

### 1. Backend Setup

```bash
cd soccer-analytics-pipeline

# Install web dependencies
pip install -r requirements-web.txt

# Start the API server
python start_api.py
```

The API will be available at `http://localhost:5000`

### 2. Frontend Setup

```bash
# Install socket.io client (already added to package.json)
npm install socket.io-client

# Start your React development server
npm run dev
```

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check |
| POST | `/api/upload` | Upload video file |
| POST | `/api/start_analysis` | Start video analysis |
| POST | `/api/stop_analysis` | Stop analysis (ESC) |
| GET | `/api/results` | Get analysis results |
| GET | `/api/status` | Get processing status |

## 🔄 WebSocket Events

### Client → Server
- `stop_analysis` - Stop the current analysis

### Server → Client
- `connected` - Connection established
- `analysis_started` - Analysis has begun
- `frame_update` - Live frame with statistics
- `analysis_complete` - Analysis finished with results
- `analysis_stopped` - Analysis stopped by user
- `error` - Error occurred

## 🎮 User Flow

1. **Upload Video**: User uploads football match footage
2. **Choose Analysis Type**:
   - **Live Soccer Analytics**: Real-time processing with live feed
   - **Traditional Analysis**: Batch processing with mock data
3. **Live Analysis**: 
   - Real-time video feed with overlays
   - Live statistics (possession, events, performance)
   - Press ESC to stop and view results
4. **Results**: Comprehensive analysis dashboard

## 🔧 Integration Features

### Real-time Analysis
- Live video feed with AI overlays
- Real-time statistics updates
- Player tracking and team classification
- Ball possession analysis
- Performance metrics
- Event detection (goals, cards, fouls)

### Data Format
The backend provides data in the same format as your existing `AnalysisData` interface:

```typescript
interface AnalysisData {
  videoMetrics: {
    duration: string;
    events: Array<{ type: string; time: string; description: string }>;
    highlights: string[];
  };
  mlScores: {
    teamA: { name: string; score: number; metrics: any };
    teamB: { name: string; score: number; metrics: any };
    playerRatings: Array<{ name: string; position: string; rating: number }>;
  };
  // ... plus additional soccer analytics data
}
```

## 🎯 Key Components

### `soccerAnalyticsApi.ts`
- Handles all API communication
- WebSocket management
- Event handling for real-time updates

### `LiveAnalysis.tsx`
- Real-time video display
- Live statistics dashboard
- ESC key handling
- WebSocket event management

### Updated `AnalyzePage.tsx`
- Analysis type selection
- Integration with existing UI
- Seamless switching between live and traditional analysis

## 🔑 Key Features

### ESC Key Functionality
- Press ESC during live analysis to stop and view results
- Automatic transition to results page
- Preserves all analysis data

### Error Handling
- Network error recovery
- Upload failure handling
- Analysis error reporting
- Toast notifications for user feedback

### Performance
- Efficient WebSocket communication
- Base64 frame streaming
- Real-time statistics updates
- Responsive UI during processing

## 🛠️ Configuration

### Backend Configuration
Edit `soccer-analytics-pipeline/src/soccer_analytics/config.py`:

```python
@dataclass
class PipelineConfig:
    frame_resize_width: int | None = 960  # Resize for web streaming
    show_visualization: bool = True       # Enable overlays
    debug: bool = False                   # Debug mode
```

### Frontend Configuration
Edit `src/services/soccerAnalyticsApi.ts`:

```typescript
const API_BASE_URL = 'http://localhost:5000/api';
const WEBSOCKET_URL = 'http://localhost:5000';
```

## 🐛 Troubleshooting

### Backend Issues
1. **Import Error**: Ensure `src` directory is in Python path
2. **YOLO Model**: Check if ultralytics is installed
3. **Port Conflict**: Change port in `web_app.py` if needed

### Frontend Issues
1. **CORS Error**: Ensure Flask-CORS is installed and configured
2. **WebSocket Connection**: Check if backend is running on correct port
3. **Upload Fails**: Check file size limits and format support

### Performance Issues
1. **Slow Processing**: Reduce `frame_resize_width` in config
2. **Memory Usage**: Monitor video file sizes
3. **Network Lag**: Consider local deployment

## 📝 Development Notes

### Adding New Analysis Features
1. Add to `SoccerPipeline` class
2. Update `WebPipelineAdapter.get_final_results()`
3. Extend frontend `AnalysisData` interface
4. Update results display components

### Customizing UI
- Modify `LiveAnalysis.tsx` for different layouts
- Update `AnalyzePage.tsx` for new analysis types
- Extend `AnalysisResults.tsx` for additional data

## 🚀 Deployment

### Production Considerations
1. Use production WSGI server (gunicorn)
2. Configure proper CORS origins
3. Set up file upload limits
4. Monitor memory usage for video processing
5. Consider using Redis for session management

### Docker Deployment
```dockerfile
# Example Dockerfile for backend
FROM python:3.11
WORKDIR /app
COPY requirements-web.txt .
RUN pip install -r requirements-web.txt
COPY . .
CMD ["python", "web_app.py"]
```

## 📊 Monitoring

The integration provides:
- Real-time processing statistics
- Error logging and reporting
- Performance metrics
- User interaction tracking

## 🤝 Contributing

When adding new features:
1. Update both backend and frontend
2. Maintain data format compatibility
3. Add proper error handling
4. Update documentation
5. Test WebSocket communication

---

**Happy Analyzing! ⚽🚀**
