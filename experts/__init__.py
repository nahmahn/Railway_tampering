"""
Railway Tampering Detection - Expert Agents Module

SETUP:
1. Install requirements: pip install -r requirements.txt
2. Configure .env file with GEMINI_API_KEY
3. Replace placeholder experts with actual implementations

"""

# Orchestration
from .orchestration import (
    RailwayTamperingOrchestrator,
    create_orchestration_graph,
    compile_orchestrator,
    InputType,
    InputData,
    ExpertResult,
    OrchestratorState,
)

# Track Structural Expert (PLACEHOLDER)
from .track_structural_expert import (
    TrackStructuralAnalysisResult,
    analyze_csv as analyze_track_structural,
)

# Visual Integrity Expert (PLACEHOLDER)
from .visual_integrity_expert import (
    VisualIntegrityResult,
    Detection,
    DetectionType,
    analyze_image,
    analyze_video,
)

# Thermal Anomaly Expert (PLACEHOLDER)
from .thermal_anomaly_expert import (
    ThermalAnomalyResult,
    ThermalAnomaly,
    AnomalyType,
    analyze_lidar,
    analyze_thermal,
)

# Contextual Reasoning Expert (IMPLEMENTED - Gemini)
from .contextual_reasoning_expert import (
    ReasoningResult,
    QueryType,
    process_with_gemini,
    query_gemini,
)


__all__ = [
    # Orchestration
    "RailwayTamperingOrchestrator",
    "create_orchestration_graph",
    "compile_orchestrator",
    "InputType",
    "InputData",
    "ExpertResult",
    "OrchestratorState",
    
    # Track Structural Expert
    "TrackStructuralAnalysisResult",
    "analyze_track_structural",
    
    # Visual Integrity Expert
    "VisualIntegrityResult",
    "Detection",
    "DetectionType",
    "analyze_image",
    "analyze_video",
    
    # Thermal Anomaly Expert
    "ThermalAnomalyResult",
    "ThermalAnomaly",
    "AnomalyType",
    "analyze_lidar",
    "analyze_thermal",
    
    # Contextual Reasoning Expert
    "ReasoningResult",
    "QueryType",
    "process_with_gemini",
    "query_gemini",
]


__version__ = "0.1.0"
