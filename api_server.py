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

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect, Depends, Header
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Google OAuth verification
try:
    from google.oauth2 import id_token as google_id_token
    from google.auth.transport import requests as google_requests
    GOOGLE_AUTH_AVAILABLE = True
except ImportError:
    GOOGLE_AUTH_AVAILABLE = False
    print("Warning: google-auth not installed. Token verification disabled.")

GOOGLE_CLIENT_ID = "741472179168-ujrglefc9vjcusv1pg0muqhqihavds12.apps.googleusercontent.com"

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


class CreateMissionRequest(BaseModel):
    alert_session_id: str = Field(..., description="Session ID from history to create mission for")
    priority: str = Field(default="P2", description="Priority P1-P4")
    crew_id: Optional[str] = Field(default=None, description="Assigned crew ID")
    notes: str = Field(default="", description="Mission notes")


class AdvanceMissionRequest(BaseModel):
    crew_id: Optional[str] = Field(default=None, description="Crew to assign (for ASSIGNED stage)")
    priority: Optional[str] = Field(default=None, description="Priority (for TRIAGED stage)")


class ResolveMissionRequest(BaseModel):
    resolution_notes: str = Field(..., description="Resolution notes")


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
        self.crews_collection = None
        self.missions_collection = None
        self.resolved_issues_collection = None
    
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
                    self.crews_collection = self.db["crews"]
                    self.missions_collection = self.db["missions"]
                    self.resolved_issues_collection = self.db["resolved_issues"]
                    # Test connection
                    self.mongo_client.admin.command('ping')
                    print("✅ Connected to MongoDB Atlas")
                    # Seed crews if empty
                    self._seed_crews()
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

    def _seed_crews(self):
        """Seed crews collection if empty."""
        if self.crews_collection is None:
            return
        try:
            if self.crews_collection.count_documents({}) == 0:
                crews = [
                    {"_id": "CR-001", "team_lead": "R. Kumar", "specialization": "Track P.Way", "zone": "Northern", "status": "available", "members": 4, "contact": "+91-98100-XXXXX"},
                    {"_id": "CR-002", "team_lead": "S. Singh", "specialization": "Signal & Telecom", "zone": "Northern", "status": "available", "members": 3, "contact": "+91-98200-XXXXX"},
                    {"_id": "CR-003", "team_lead": "A. Patel", "specialization": "Structural Engineering", "zone": "Western", "status": "available", "members": 5, "contact": "+91-98300-XXXXX"},
                    {"_id": "CR-004", "team_lead": "M. Das", "specialization": "Emergency Response", "zone": "Northern", "status": "standby", "members": 6, "contact": "+91-98400-XXXXX"},
                    {"_id": "CR-005", "team_lead": "V. Reddy", "specialization": "Track Welding", "zone": "Southern", "status": "available", "members": 3, "contact": "+91-98500-XXXXX"},
                    {"_id": "CR-006", "team_lead": "K. Sharma", "specialization": "Inspection & QC", "zone": "Northern", "status": "available", "members": 4, "contact": "+91-98600-XXXXX"},
                ]
                self.crews_collection.insert_many(crews)
                print(f"✅ Seeded {len(crews)} crews into MongoDB")
            else:
                print(f"✅ Crews collection already has {self.crews_collection.count_documents({})} records")
        except Exception as e:
            print(f"⚠️ Error seeding crews: {e}")

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


async def verify_google_token(authorization: Optional[str] = Header(default=None)):
    """Verify Google OAuth2 ID token from Authorization header."""
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header missing")
    
    # Extract token from "Bearer <token>"
    parts = authorization.split(" ")
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(status_code=401, detail="Invalid authorization format. Use: Bearer <token>")
    
    token = parts[1]
    
    # Allow mock token for development/demo
    if token == "mock-token":
        return {"email": "demo@railways.gov.in", "name": "Demo User", "sub": "mock"}
    
    if not GOOGLE_AUTH_AVAILABLE:
        # If google-auth is not installed, allow all tokens (dev mode)
        return {"email": "unknown", "name": "Unknown", "sub": "dev"}
    
    try:
        id_info = google_id_token.verify_oauth2_token(
            token, google_requests.Request(), GOOGLE_CLIENT_ID
        )
        return id_info
    except ValueError as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {str(e)}")


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
    
    # Fetch recent history context if available
    history_context = []
    if app_state.history_collection is not None:
        try:
            # Fetch last 20 records, projection to keep payload small
            cursor = app_state.history_collection.find(
                {}, 
                {
                    "session_id": 1, 
                    "timestamp": 1, 
                    "overall_assessment.risk_level": 1,
                    "overall_assessment.confidence": 1,
                    "alerts": 1,
                    "summary": 1 
                }
            ).sort("created_at", -1).limit(20)
            
            for doc in cursor:
                doc["id"] = doc.pop("_id")
                # Simplify alerts for context
                if "alerts" in doc and doc["alerts"]:
                    doc["alert_summary"] = [a.get("title", "Alert") for a in doc["alerts"]]
                    del doc["alerts"]
                history_context.append(doc)
        except Exception as e:
            print(f"Failed to fetch history context: {e}")
    
    try:
        result = query_gemini(
            question=request.query,
            track_structural_result=request.context.get("structural"),
            visual_integrity_result=request.context.get("visual"),
            thermal_anomaly_result=request.context.get("thermal"),
            history_context=history_context
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


# ==================== Crews & Missions API ====================

@app.get("/api/crews")
async def get_crews():
    """Get all maintenance crews."""
    if app_state.crews_collection is None:
        return {"crews": []}
    try:
        crews = list(app_state.crews_collection.find())
        for c in crews:
            c["id"] = c.pop("_id")
        return {"crews": crews}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.patch("/api/crews/{crew_id}/status")
async def update_crew_status(crew_id: str, status: str = Form(...), user_info: dict = Depends(verify_google_token)):
    """Update crew availability status."""
    if app_state.crews_collection is None:
        raise HTTPException(status_code=503, detail="Database not available")
    try:
        result = app_state.crews_collection.update_one(
            {"_id": crew_id},
            {"$set": {"status": status}}
        )
        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail=f"Crew {crew_id} not found")
        return {"success": True, "crew_id": crew_id, "status": status}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/missions")
async def get_missions(stage: Optional[str] = None):
    """Get all missions, optionally filtered by stage."""
    if app_state.missions_collection is None:
        return {"missions": []}
    try:
        query = {}
        if stage:
            query["stage"] = stage
        missions = list(app_state.missions_collection.find(query).sort("created_at", -1))
        for m in missions:
            m["id"] = str(m.pop("_id"))
            if "created_at" in m:
                m["created_at"] = m["created_at"].isoformat()
            if "updated_at" in m:
                m["updated_at"] = m["updated_at"].isoformat()
        return {"missions": missions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/missions")
async def create_mission(request: CreateMissionRequest, user_info: dict = Depends(verify_google_token)):
    """Create a new mission from an alert/session."""
    if app_state.missions_collection is None:
        raise HTTPException(status_code=503, detail="Database not available")
    try:
        # Fetch the source alert data from history
        alert_data = {}
        if app_state.history_collection is not None:
            doc = app_state.history_collection.find_one({"_id": request.alert_session_id})
            if doc:
                alert_data = {
                    "session_id": doc.get("_id", ""),
                    "risk_level": doc.get("overall_assessment", {}).get("risk_level", "unknown"),
                    "tampering_type": doc.get("tampering_analysis", {}).get("tampering_assessment", {}).get("tampering_type", "Unknown"),
                    "confidence": doc.get("overall_assessment", {}).get("confidence", 0),
                    "timestamp": doc.get("timestamp", ""),
                }

        mission_id = f"MSN-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:4]}"
        mission = {
            "_id": mission_id,
            "alert_session_id": request.alert_session_id,
            "alert_data": alert_data,
            "stage": "assigned" if request.crew_id else "new",
            "priority": request.priority,
            "crew_id": request.crew_id,
            "notes": request.notes,
            "resolution_notes": "",
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
        }
        
        # If crew is assigned immediately, update their status
        if request.crew_id and app_state.crews_collection is not None:
            app_state.crews_collection.update_one(
                {"_id": request.crew_id},
                {"$set": {"status": "on_mission"}}
            )
            
        app_state.missions_collection.insert_one(mission)
        mission["id"] = mission.pop("_id")
        mission["created_at"] = mission["created_at"].isoformat()
        mission["updated_at"] = mission["updated_at"].isoformat()
        return {"success": True, "mission": mission}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


MISSION_STAGES = ["new", "triaged", "assigned", "in_progress", "resolved"]

@app.patch("/api/missions/{mission_id}/advance")
async def advance_mission(mission_id: str, request: AdvanceMissionRequest, user_info: dict = Depends(verify_google_token)):
    """Advance a mission to the next stage."""
    if app_state.missions_collection is None:
        raise HTTPException(status_code=503, detail="Database not available")
    try:
        mission = app_state.missions_collection.find_one({"_id": mission_id})
        if not mission:
            raise HTTPException(status_code=404, detail=f"Mission {mission_id} not found")

        current_stage = mission.get("stage", "new")
        if current_stage not in MISSION_STAGES:
            raise HTTPException(status_code=400, detail=f"Invalid current stage: {current_stage}")

        current_idx = MISSION_STAGES.index(current_stage)
        if current_idx >= len(MISSION_STAGES) - 1:
            raise HTTPException(status_code=400, detail="Mission already resolved")

        next_stage = MISSION_STAGES[current_idx + 1]
        update = {"$set": {"stage": next_stage, "updated_at": datetime.utcnow()}}

        # Apply optional fields based on stage transition
        if request.priority:
            update["$set"]["priority"] = request.priority
        if request.crew_id:
            update["$set"]["crew_id"] = request.crew_id
            # Update crew status to on_mission
            if app_state.crews_collection:
                app_state.crews_collection.update_one(
                    {"_id": request.crew_id},
                    {"$set": {"status": "on_mission"}}
                )

        app_state.missions_collection.update_one({"_id": mission_id}, update)
        return {"success": True, "mission_id": mission_id, "new_stage": next_stage}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.patch("/api/missions/{mission_id}/resolve")
async def resolve_mission(mission_id: str, request: ResolveMissionRequest, user_info: dict = Depends(verify_google_token)):
    """Resolve a mission with notes."""
    if app_state.missions_collection is None:
        raise HTTPException(status_code=503, detail="Database not available")
    try:
        mission = app_state.missions_collection.find_one({"_id": mission_id})
        if not mission:
            raise HTTPException(status_code=404, detail=f"Mission {mission_id} not found")

        # Free up the crew
        crew_id = mission.get("crew_id")
        if crew_id and app_state.crews_collection:
            app_state.crews_collection.update_one(
                {"_id": crew_id},
                {"$set": {"status": "available"}}
            )

        app_state.missions_collection.update_one(
            {"_id": mission_id},
            {"$set": {
                "stage": "resolved",
                "resolution_notes": request.resolution_notes,
                "resolved_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
            }}
        )
        
        # Archive to resolved_issues collection
        if hasattr(app_state, "resolved_issues_collection") and app_state.resolved_issues_collection is not None:
             # Fetch the updated mission to archive
             updated_mission = app_state.missions_collection.find_one({"_id": mission_id})
             if updated_mission:
                 # Create a copy for archival
                 archive_doc = updated_mission.copy()
                 
                 # Check if already archived to avoid duplicates
                 existing = app_state.resolved_issues_collection.find_one({"_id": mission_id})
                 if not existing:
                     app_state.resolved_issues_collection.insert_one(archive_doc)
        
        return {"success": True, "mission_id": mission_id, "stage": "resolved"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
