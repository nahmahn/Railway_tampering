"""
Railway Tampering Detection System - FastAPI Server

This module provides REST API endpoints for all expert agents and the combined
inference engine. Designed for easy integration with the frontend.

Endpoints:
- /api/health - Health check
- /api/analyze/visual - Visual integrity analysis (images/videos)
- /api/analyze/lidar - LiDAR/Thermal analysis
- /api/analyze/vibration - Vibration/Structural analysis
- /api/analyze/combined - Combined multi-expert analysis
- /api/query - Natural language query endpoint
- /api/alerts - Real-time alerts management
- /api/orchestrate - Full orchestration pipeline
"""

import os
import json
import uuid
import shutil
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
from contextlib import asynccontextmanager
import certifi

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Import expert modules
import sys
experts_path = Path(__file__).parent / "experts"
if str(experts_path) not in sys.path:
    sys.path.insert(0, str(experts_path))

from experts.visual_integrity_expert import VisualIntegrityExpert
from experts.thermal_anomaly_expert import ThermalAnomalyExpert
from experts.track_structural_expert import TrackStructuralExpert
from experts.combined_inference import CombinedInferenceEngine, RealTimeAlertManager
from experts.contextual_reasoning_expert import process_with_gemini, query_gemini
from experts.combined_inference import CombinedInferenceEngine, RealTimeAlertManager
from experts.contextual_reasoning_expert import process_with_gemini, query_gemini
from experts.orchestration import RailwayTamperingOrchestrator

# MongoDB Imports
try:
    from pymongo import MongoClient
    from dotenv import load_dotenv
    load_dotenv()  # Load environment variables from .env
    MONGO_AVAILABLE = True
except ImportError:
    MONGO_AVAILABLE = False
    print("Warning: pymongo or python-dotenv not installed. Database features disabled.")

# Configuration
UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


# ==================== Pydantic Models ====================

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str = "1.0.0"
    experts_available: Dict[str, bool]


class AnalysisRequest(BaseModel):
    file_paths: List[str] = Field(default=[], description="List of file paths to analyze")
    metadata: Dict[str, Any] = Field(default={}, description="Additional metadata")
    context: Dict[str, Any] = Field(default={}, description="Context information")


class QueryRequest(BaseModel):
    query: str = Field(..., description="Natural language query")
    context: Dict[str, Any] = Field(default={}, description="Expert results context")


class CombinedAnalysisRequest(BaseModel):
    image_paths: List[str] = Field(default=[], description="Image file paths")
    video_paths: List[str] = Field(default=[], description="Video file paths")
    lidar_paths: List[str] = Field(default=[], description="LiDAR file paths")
    csv_paths: List[str] = Field(default=[], description="CSV file paths")
    context: Dict[str, Any] = Field(default={}, description="Additional context")


class OrchestrationRequest(BaseModel):
    inputs: List[Dict[str, Any]] = Field(..., description="List of input items")


class AlertAcknowledgeRequest(BaseModel):
    alert_id: str = Field(..., description="Alert ID to acknowledge")


class AnalysisResponse(BaseModel):
    success: bool
    session_id: str
    timestamp: str
    result: Dict[str, Any]
    alerts: List[Dict[str, Any]] = []


# ==================== Application State ====================

class AppState:
    """Application state management."""
    
    def __init__(self):
        self.visual_expert: Optional[VisualIntegrityExpert] = None
        self.thermal_expert: Optional[ThermalAnomalyExpert] = None
        self.structural_expert: Optional[TrackStructuralExpert] = None
        self.combined_engine: Optional[CombinedInferenceEngine] = None
        self.orchestrator: Optional[RailwayTamperingOrchestrator] = None
        self.alert_manager: Optional[RealTimeAlertManager] = None
        self.websocket_connections: List[WebSocket] = []
        
        # Database
        self.mongo_client: Optional[MongoClient] = None
        self.db = None
        self.history_collection = None
    
    def initialize(self):
        """Initialize all experts and database."""
        print("Initializing experts...")
        
        # Initialize MongoDB
        if MONGO_AVAILABLE:
            try:
                mongo_uri = os.getenv("MONGO_URI")
                db_name = os.getenv("DB_NAME", "railway_tampering_db")
                if mongo_uri:
                    self.mongo_client = MongoClient(mongo_uri, tlsCAFile=certifi.where())
                    self.db = self.mongo_client[db_name]
                    self.history_collection = self.db["history"]
                    # Test connection
                    self.mongo_client.admin.command('ping')
                    print("✅ Connected to MongoDB Atlas")
                else:
                    print("⚠️ MONGO_URI not found in environment variables. Database disabled.")
            except Exception as e:
                print(f"⚠️ MongoDB connection failed: {e}")
        
        try:
            self.visual_expert = VisualIntegrityExpert()
            print("✅ Visual Integrity Expert initialized")
        except Exception as e:
            print(f"⚠️ Visual Expert initialization failed: {e}")
        
        try:
            self.thermal_expert = ThermalAnomalyExpert()
            print("✅ Thermal Anomaly Expert initialized")
        except Exception as e:
            print(f"⚠️ Thermal Expert initialization failed: {e}")
        
        try:
            self.structural_expert = TrackStructuralExpert(
                xgb_model_path="experts/xgb_metro_model.pkl",
                scaler_path="experts/scaler_metro.pkl"
            )
            print("✅ Track Structural Expert initialized")
        except Exception as e:
            print(f"⚠️ Structural Expert initialization failed: {e}")
        
        try:
            self.combined_engine = CombinedInferenceEngine()
            self.alert_manager = self.combined_engine.alert_manager
            # Register websocket broadcast handler (use sync wrapper)
            self.alert_manager.register_handler(self._broadcast_alert_sync)
            print("✅ Combined Inference Engine initialized")
        except Exception as e:
            print(f"⚠️ Combined Engine initialization failed: {e}")
        
        try:
            self.orchestrator = RailwayTamperingOrchestrator()
            print("✅ Orchestrator initialized")
        except Exception as e:
            print(f"⚠️ Orchestrator initialization failed: {e}")
    
    async def _broadcast_alert(self, alert):
        """Broadcast alert to all connected WebSocket clients."""
        try:
            # Convert RealTimeAlert object to dictionary if it's not already
            if hasattr(alert, "to_dict"):
                alert_data = alert.to_dict()
            else:
                alert_data = {
                    "alert_id": alert.alert_id,
                    "timestamp": alert.timestamp,
                    "severity": alert.severity.value if hasattr(alert.severity, "value") else alert.severity,
                    "title": alert.title,
                    "message": alert.message,
                    "action_required": alert.action_required
                }

            message = {
                "type": "alert",
                "data": alert_data
            }
            
            disconnected = []
            for ws in self.websocket_connections:
                try:
                    await ws.send_json(message)
                except:
                    disconnected.append(ws)
            
            for ws in disconnected:
                if ws in self.websocket_connections:
                    self.websocket_connections.remove(ws)
        except Exception as e:
            print(f"Error broadcasting alert: {e}")

    def _broadcast_alert_sync(self, alert):
        """Synchronous wrapper for broadcasting alerts."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self._broadcast_alert(alert))
            else:
                loop.run_until_complete(self._broadcast_alert(alert))
        except Exception as e:
            print(f"Error in sync broadcast: {e}")

app_state = AppState()


# ==================== Lifespan Management ====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    # Startup
    app_state.initialize()
    yield
    # Shutdown
    print("Shutting down...")


# ==================== FastAPI Application ====================

app = FastAPI(
    title="Railway Tampering Detection API",
    description="REST API for railway track tampering detection using multi-expert analysis",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount uploads directory for static access
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")


# ==================== Helper Functions ====================

def save_upload_file(upload_file: UploadFile) -> str:
    """Save uploaded file and return paths."""
    file_id = str(uuid.uuid4())[:8]
    filename = f"{file_id}_{upload_file.filename}"
    file_path = UPLOAD_DIR / filename
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)
    
    # Return: (absolute path for file ops, relative path for URL)
    return str(file_path), f"uploads/{filename}"


def generate_session_id() -> str:
    """Generate unique session ID."""
    return f"API-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6]}"


# ==================== API Endpoints ====================

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat() + "Z",
        experts_available={
            "visual": app_state.visual_expert is not None,
            "thermal": app_state.thermal_expert is not None,
            "structural": app_state.structural_expert is not None,
            "combined": app_state.combined_engine is not None,
            "orchestrator": app_state.orchestrator is not None
        }
    )


@app.post("/api/analyze/visual")
async def analyze_visual(
    files: List[UploadFile] = File(...),
    metadata: str = Form(default="{}")
):
    """
    Analyze images/videos for visual tampering detection.
    
    Accepts image files (jpg, png, etc.) and video files (mp4, avi, etc.)
    """
    if not app_state.visual_expert:
        raise HTTPException(status_code=503, detail="Visual expert not available")
    
    session_id = generate_session_id()
    results = []
    all_alerts = []
    
    try:
        meta = json.loads(metadata)
    except:
        meta = {}
    
    for file in files:
        file_path = save_upload_file(file)
        
        try:
            ext = Path(file.filename).suffix.lower()
            
            if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']:
                result = app_state.visual_expert.analyze_image(file_path, metadata=meta)
            elif ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
                result = app_state.visual_expert.analyze_video(file_path, metadata=meta)
            else:
                results.append({"file": file.filename, "error": f"Unsupported format: {ext}"})
                continue
            
            result_dict = app_state.visual_expert.to_dict(result)
            result_dict["file_url"] = file_url
            results.append(result_dict)
            all_alerts.extend(result.alerts)
            
        except Exception as e:
            results.append({"file": file.filename, "error": str(e)})
        

    
    return AnalysisResponse(
        success=True,
        session_id=session_id,
        timestamp=datetime.utcnow().isoformat() + "Z",
        result={"analyses": results},
        alerts=[{"message": a} for a in all_alerts]
    )


@app.post("/api/analyze/lidar")
async def analyze_lidar(
    files: List[UploadFile] = File(...),
    metadata: str = Form(default="{}")
):
    """
    Analyze LiDAR point cloud files for structural tampering.
    
    Accepts .npy, .pcd, .ply, .las, .laz files
    """
    if not app_state.thermal_expert:
        raise HTTPException(status_code=503, detail="Thermal/LiDAR expert not available")
    
    session_id = generate_session_id()
    results = []
    all_alerts = []
    
    try:
        meta = json.loads(metadata)
    except:
        meta = {}
    
    for file in files:
        file_path = save_upload_file(file)
        
        try:
            result = app_state.thermal_expert.analyze_lidar(file_path, meta)
            result_dict = app_state.thermal_expert.to_dict(result)
            result_dict["file_url"] = file_url
            results.append(result_dict)
            all_alerts.extend(result.alerts)
            
        except Exception as e:
            results.append({"file": file.filename, "error": str(e)})
        

    
    return AnalysisResponse(
        success=True,
        session_id=session_id,
        timestamp=datetime.utcnow().isoformat() + "Z",
        result={"analyses": results},
        alerts=[{"message": a} for a in all_alerts]
    )


@app.post("/api/analyze/vibration")
async def analyze_vibration(
    files: List[UploadFile] = File(...),
    metadata: str = Form(default="{}")
):
    """
    Analyze vibration/sensor CSV files for structural anomalies.
    
    Accepts CSV files with x, y, z accelerometer data or geometric measurements
    """
    if not app_state.structural_expert:
        raise HTTPException(status_code=503, detail="Structural expert not available")
    
    session_id = generate_session_id()
    results = []
    all_alerts = []
    
    try:
        meta = json.loads(metadata)
    except:
        meta = {}
    
    for file in files:
        file_path = save_upload_file(file)
        
        try:
            result = app_state.structural_expert.analyze_csv(file_path, meta)
            result_dict = app_state.structural_expert.to_dict(result)
            result_dict["file_url"] = file_url
            results.append(result_dict)
            all_alerts.extend(result.alerts)
            
        except Exception as e:
            results.append({"file": file.filename, "error": str(e)})
        

    
    return AnalysisResponse(
        success=True,
        session_id=session_id,
        timestamp=datetime.utcnow().isoformat() + "Z",
        result={"analyses": results},
        alerts=[{"message": a} for a in all_alerts]
    )


@app.post("/api/analyze/combined")
async def analyze_combined(
    images: List[UploadFile] = File(default=[]),
    videos: List[UploadFile] = File(default=[]),
    lidar_files: List[UploadFile] = File(default=[]),
    csv_files: List[UploadFile] = File(default=[]),
    context: str = Form(default="{}")
):
    """
    Combined multi-expert analysis.
    
    Runs all applicable experts and produces combined tampering analysis
    with action recommendations.
    """
    if not app_state.combined_engine:
        raise HTTPException(status_code=503, detail="Combined inference engine not available")
    
    session_id = generate_session_id()
    
    try:
        ctx = json.loads(context)
    except:
        ctx = {}
    
    # Save uploaded files - store both absolute paths for analysis and relative URLs for frontend
    image_paths = []
    image_urls = {}  # Map absolute path -> relative URL
    for f in images:
        abs_path, url_path = save_upload_file(f)
        image_paths.append(abs_path)
        image_urls[abs_path] = url_path
        
    video_paths = []
    video_urls = {}
    for f in videos:
        abs_path, url_path = save_upload_file(f)
        video_paths.append(abs_path)
        video_urls[abs_path] = url_path
        
    lidar_paths = []
    for f in lidar_files:
        abs_path, url_path = save_upload_file(f)
        lidar_paths.append(abs_path)
        
    csv_paths = []
    for f in csv_files:
        abs_path, url_path = save_upload_file(f)
        csv_paths.append(abs_path)
    
    try:
        result = app_state.combined_engine.analyze(
            image_paths=image_paths if image_paths else None,
            video_paths=video_paths if video_paths else None,
            lidar_paths=lidar_paths if lidar_paths else None,
            csv_paths=csv_paths if csv_paths else None,
            context=ctx
        )
        
        result_dict = app_state.combined_engine.to_dict(result)
        
        # Post-process: Replace absolute file paths in visual_result with relative URLs for frontend
        if "expert_results" in result_dict and result_dict["expert_results"].get("visual"):
            visual = result_dict["expert_results"]["visual"]
            if isinstance(visual, dict) and visual.get("file_url"):
                abs_path = visual["file_url"]
                if abs_path in image_urls:
                    visual["file_url"] = image_urls[abs_path]
            elif isinstance(visual, list):
                for v in visual:
                    if isinstance(v, dict) and v.get("file_url"):
                        abs_path = v["file_url"]
                        if abs_path in image_urls:
                            v["file_url"] = image_urls[abs_path]

        # Save to Database
        if app_state.history_collection is not None:
            try:
                # Create a copy for DB to avoid mutating original
                db_record = result_dict.copy()
                db_record["_id"] = session_id  # Use session ID as document ID
                db_record["created_at"] = datetime.utcnow()
                app_state.history_collection.insert_one(db_record)
                print(f"Saved analysis {session_id} to MongoDB")
            except Exception as e:
                print(f"Error saving to MongoDB: {e}")

        return AnalysisResponse(
            success=True,
            session_id=session_id,
            timestamp=datetime.utcnow().isoformat() + "Z",
            result=result_dict,
            alerts=result_dict.get("alerts", [])
        )
        
    except Exception as e:
        print(f"Error in combined analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/analyze/combined-paths")
async def analyze_combined_paths(request: CombinedAnalysisRequest):
    """
    Combined analysis using file paths (for files already on server).
    """
    if not app_state.combined_engine:
        raise HTTPException(status_code=503, detail="Combined inference engine not available")
    
    session_id = generate_session_id()
    
    result = app_state.combined_engine.analyze(
        image_paths=request.image_paths if request.image_paths else None,
        video_paths=request.video_paths if request.video_paths else None,
        lidar_paths=request.lidar_paths if request.lidar_paths else None,
        csv_paths=request.csv_paths if request.csv_paths else None,
        context=request.context
    )
    
    result_dict = app_state.combined_engine.to_dict(result)
    
    return AnalysisResponse(
        success=True,
        session_id=session_id,
        timestamp=datetime.utcnow().isoformat() + "Z",
        result=result_dict,
        alerts=result_dict.get("alerts", [])
    )


@app.post("/api/query")
async def query_endpoint(request: QueryRequest):
    """
    Natural language query endpoint.
    
    Processes queries using Gemini with optional expert context.
    """
    session_id = generate_session_id()
    
    try:
        result = query_gemini(
            question=request.query,
            track_structural_result=request.context.get("structural"),
            visual_integrity_result=request.context.get("visual"),
            thermal_anomaly_result=request.context.get("thermal")
        )
        
        return AnalysisResponse(
            success=True,
            session_id=session_id,
            timestamp=datetime.utcnow().isoformat() + "Z",
            result={
                "query": result.query,
                "query_type": result.query_type.value,
                "response": result.response,
                "confidence": result.confidence,
                "insights": result.insights,
                "recommendations": result.recommendations,
                "status": result.status
            },
            alerts=[]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query processing error: {str(e)}")


@app.post("/api/orchestrate")
async def orchestrate_endpoint(request: OrchestrationRequest):
    """
    Full orchestration pipeline endpoint.
    
    Routes inputs to appropriate experts based on input type.
    """
    if not app_state.orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not available")
    
    session_id = generate_session_id()
    
    try:
        result = app_state.orchestrator.process(request.inputs)
        
        return AnalysisResponse(
            success=True,
            session_id=session_id,
            timestamp=datetime.utcnow().isoformat() + "Z",
            result=result,
            alerts=[{"message": a} for a in result.get("alerts", [])]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Orchestration error: {str(e)}")


@app.get("/api/alerts")
async def get_alerts():
    """Get all pending alerts."""
    if not app_state.alert_manager:
        return {"alerts": []}
    
    alerts = app_state.alert_manager.get_pending_alerts()
    
    return {
        "alerts": [
            {
                "alert_id": a.alert_id,
                "timestamp": a.timestamp,
                "severity": a.severity.value,
                "source": a.source_expert,
                "title": a.title,
                "message": a.message,
                "action_required": a.action_required,
                "acknowledged": a.acknowledged
            }
            for a in alerts
        ]
    }


@app.get("/api/history")
async def get_history(limit: int = 10, skip: int = 0):
    """Get analysis history from MongoDB."""
    if app_state.history_collection is None:
        return {"history": []}
    
    try:
        cursor = app_state.history_collection.find().sort("created_at", -1).skip(skip).limit(limit)
        history = []
        for doc in cursor:
            doc["id"] = doc.pop("_id")  # Rename _id to id for frontend
            doc.pop("created_at", None) # Remove if redundant with timestamp
            history.append(doc)
        
        return {"history": history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/alerts/acknowledge")
async def acknowledge_alert(request: AlertAcknowledgeRequest):
    """Acknowledge an alert."""
    if not app_state.alert_manager:
        raise HTTPException(status_code=503, detail="Alert manager not available")
    
    success = app_state.alert_manager.acknowledge_alert(request.alert_id)
    
    if not success:
        raise HTTPException(status_code=404, detail=f"Alert not found: {request.alert_id}")
    
    return {"success": True, "alert_id": request.alert_id}


# ==================== WebSocket for Real-time Alerts ====================

@app.websocket("/ws/alerts")
async def websocket_alerts(websocket: WebSocket):
    """WebSocket endpoint for real-time alerts."""
    await websocket.accept()
    app_state.websocket_connections.append(websocket)
    
    try:
        # Send connection confirmation
        await websocket.send_json({
            "type": "connected",
            "message": "Connected to alert stream"
        })
        
        # Keep connection alive
        while True:
            data = await websocket.receive_text()
            
            # Handle ping/pong
            if data == "ping":
                await websocket.send_text("pong")
            
    except WebSocketDisconnect:
        app_state.websocket_connections.remove(websocket)
    except Exception:
        if websocket in app_state.websocket_connections:
            app_state.websocket_connections.remove(websocket)


# ==================== Static File Serving (Optional) ====================

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Railway Tampering Detection API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/health"
    }


# ==================== Run Server ====================

def run_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Run the FastAPI server."""
    uvicorn.run(
        "api_server:app",
        host=host,
        port=port,
        reload=reload
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Railway Tampering Detection API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Railway Tampering Detection API Server")
    print("=" * 60)
    print(f"Starting server on http://{args.host}:{args.port}")
    print(f"API Documentation: http://{args.host}:{args.port}/docs")
    print("=" * 60)
    
    uvicorn.run(
        "api_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )
