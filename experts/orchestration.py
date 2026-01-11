"""
Railway Tampering Detection - Orchestration Framework using LangGraph

This module implements an algorithmic orchestration agent that routes inputs
to specialized expert agents based on input type:
- CSV → Track Structural Analysis Expert
- Image/Video + JSON → Visual Track Integrity Expert
- LAZ/LiDAR files → Thermal Anomaly Interpretation Expert
- Natural Language Query → Contextual and Reasoning Expert (Gemini API)

Can handle multiple input types simultaneously.

INTEGRATION NOTES FOR TEAM:
===========================
Each expert agent has a PLACEHOLDER implementation. Replace the placeholder
functions with actual implementations:

1. track_structural_analysis_expert() - [PLACEHOLDER] Replace with actual implementation
2. visual_track_integrity_expert() - [PLACEHOLDER] Replace with actual implementation  
3. thermal_anomaly_interpretation_expert() - [PLACEHOLDER] Replace with actual implementation
4. contextual_reasoning_expert() - Uses Gemini API (implemented)

To integrate your expert:
- Implement the function in the respective expert file
- Import it here and replace the placeholder call
- Ensure your function returns an ExpertResult dataclass
"""

from typing import TypedDict, Annotated, Literal, List, Optional, Any, Dict
from dataclasses import dataclass, field
from enum import Enum
import operator
import os
import json
from pathlib import Path
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver

# Load environment variables
load_dotenv(Path(__file__).parent / ".env")


# ==================== Input Types ====================

class InputType(Enum):
    """Supported input types for the orchestration system."""
    CSV = "csv"                    # Track sensor data, vibration data
    IMAGE = "image"                # CCTV input, drone feed images
    VIDEO = "video"                # CCTV video feed
    JSON = "json"                  # Metadata, annotations
    LAZ = "laz"                    # LiDAR point cloud data
    LAS = "las"                    # LiDAR point cloud data (alternative format)
    THERMAL = "thermal"            # IR/Thermal data
    TEXT = "text"                  # Natural language query


# ==================== Data Classes ====================

@dataclass
class InputData:
    """Represents a single input item."""
    input_type: InputType
    file_path: Optional[str] = None
    content: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExpertResult:
    """Result from an expert agent."""
    expert_name: str
    status: Literal["success", "error", "pending"]
    output: Any = None
    confidence: float = 0.0
    alerts: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ==================== State Definition ====================

class OrchestratorState(TypedDict):
    """State for the orchestration graph."""
    # Input handling
    raw_inputs: List[Dict[str, Any]]
    parsed_inputs: Annotated[List[InputData], operator.add]
    
    # Routing information
    routes: Dict[str, List[InputData]]
    active_experts: List[str]
    
    # Expert results
    track_structural_result: Optional[ExpertResult]
    visual_integrity_result: Optional[ExpertResult]
    thermal_anomaly_result: Optional[ExpertResult]
    contextual_reasoning_result: Optional[ExpertResult]
    
    # Aggregated output
    all_results: Annotated[List[ExpertResult], operator.add]
    final_report: Optional[Dict[str, Any]]
    
    # Status tracking
    current_step: str
    errors: Annotated[List[str], operator.add]


# ==================== Expert Input Routing Map ====================

EXPERT_INPUT_MAPPING = {
    "track_structural_analysis": [InputType.CSV],
    "visual_track_integrity": [InputType.IMAGE, InputType.VIDEO, InputType.JSON],
    "thermal_anomaly_interpretation": [InputType.LAZ, InputType.LAS, InputType.THERMAL],
    "contextual_reasoning": [InputType.TEXT],
}


# ==================== Helper Functions ====================

def detect_input_type(file_path: Optional[str] = None, content: Any = None) -> InputType:
    """Detect the type of input based on file extension or content."""
    if file_path:
        ext = Path(file_path).suffix.lower()
        extension_map = {
            ".csv": InputType.CSV,
            ".jpg": InputType.IMAGE,
            ".jpeg": InputType.IMAGE,
            ".png": InputType.IMAGE,
            ".bmp": InputType.IMAGE,
            ".tiff": InputType.IMAGE,
            ".mp4": InputType.VIDEO,
            ".avi": InputType.VIDEO,
            ".mov": InputType.VIDEO,
            ".mkv": InputType.VIDEO,
            ".json": InputType.JSON,
            ".laz": InputType.LAZ,
            ".las": InputType.LAS,
            ".tif": InputType.THERMAL,
            ".ir": InputType.THERMAL,
        }
        if ext in extension_map:
            return extension_map[ext]
    
    if content:
        if isinstance(content, str):
            try:
                json.loads(content)
                return InputType.JSON
            except (json.JSONDecodeError, TypeError):
                return InputType.TEXT
        elif isinstance(content, dict):
            return InputType.JSON
    
    raise ValueError(f"Could not determine input type for: {file_path or content}")


def parse_input(raw_input: Dict[str, Any]) -> InputData:
    """Parse a raw input dictionary into an InputData object."""
    file_path = raw_input.get("file_path")
    content = raw_input.get("content")
    metadata = raw_input.get("metadata", {})
    
    if "input_type" in raw_input:
        input_type = InputType(raw_input["input_type"])
    else:
        input_type = detect_input_type(file_path, content)
    
    return InputData(
        input_type=input_type,
        file_path=file_path,
        content=content,
        metadata=metadata
    )


# ==================== Node Functions ====================

def parse_inputs_node(state: OrchestratorState) -> Dict[str, Any]:
    """Parse raw inputs into structured InputData objects."""
    parsed = []
    errors = []
    
    for raw_input in state.get("raw_inputs", []):
        try:
            parsed_input = parse_input(raw_input)
            parsed.append(parsed_input)
        except Exception as e:
            errors.append(f"Failed to parse input: {raw_input}. Error: {str(e)}")
    
    return {
        "parsed_inputs": parsed,
        "errors": errors,
        "current_step": "inputs_parsed"
    }


def route_inputs_node(state: OrchestratorState) -> Dict[str, Any]:
    """Route parsed inputs to appropriate expert agents."""
    routes: Dict[str, List[InputData]] = {
        "track_structural_analysis": [],
        "visual_track_integrity": [],
        "thermal_anomaly_interpretation": [],
        "contextual_reasoning": [],
    }
    
    for input_data in state.get("parsed_inputs", []):
        for expert, supported_types in EXPERT_INPUT_MAPPING.items():
            if input_data.input_type in supported_types:
                routes[expert].append(input_data)
    
    active_experts = [expert for expert, inputs in routes.items() if inputs]
    
    return {
        "routes": routes,
        "active_experts": active_experts,
        "current_step": "inputs_routed"
    }


# ==============================================================================
# EXPERT AGENT NODES - PLACEHOLDERS FOR TEAM INTEGRATION
# ==============================================================================

def track_structural_analysis_node(state: OrchestratorState) -> Dict[str, Any]:
    """
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║                    TRACK STRUCTURAL ANALYSIS EXPERT                          ║
    ║                              [IMPLEMENTED]                                   ║
    ╠══════════════════════════════════════════════════════════════════════════════╣
    ║ INPUT: CSV files containing:                                                 ║
    ║   - Vibration measurements (x, y, z accelerometer data)                      ║
    ║   - Geometric Sensor Data (gauge, alignment, cant, curvature)                ║
    ║   - Distributed Acoustic Sensing (DAS) Data                                  ║
    ║                                                                              ║
    ║ OUTPUT: ExpertResult with vibration analysis, risk assessment, alerts        ║
    ╚══════════════════════════════════════════════════════════════════════════════╝
    """
    inputs = state.get("routes", {}).get("track_structural_analysis", [])
    
    if not inputs:
        return {"current_step": "track_structural_skipped"}
    
    # Import and use actual track structural expert
    from experts.track_structural_expert import TrackStructuralExpert
    
    expert = TrackStructuralExpert()
    analysis_results = []
    alerts = []
    overall_confidence = 0.0
    
    for input_data in inputs:
        try:
            result = expert.analyze_csv(input_data.file_path, input_data.metadata)
            analysis_results.append(expert.to_dict(result))
            alerts.extend(result.alerts)
            overall_confidence = max(overall_confidence, result.confidence)
        except Exception as e:
            analysis_results.append({
                "file": input_data.file_path,
                "status": "error",
                "error": str(e)
            })
            alerts.append(f"Error analyzing {input_data.file_path}: {str(e)}")
    
    expert_result = ExpertResult(
        expert_name="Track Structural Analysis Expert",
        status="success",
        output=analysis_results,
        confidence=overall_confidence,
        alerts=alerts,
        metadata={"input_count": len(inputs), "implemented": True}
    )
    
    return {
        "track_structural_result": expert_result,
        "all_results": [expert_result],
        "current_step": "track_structural_complete"
    }


def visual_track_integrity_node(state: OrchestratorState) -> Dict[str, Any]:
    """
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║                    VISUAL TRACK INTEGRITY EXPERT                             ║
    ║                              [IMPLEMENTED]                                   ║
    ╠══════════════════════════════════════════════════════════════════════════════╣
    ║ INPUT: Image/Video files + JSON metadata:                                    ║
    ║   - CCTV Images/Video feeds                                                  ║
    ║   - Drone captured images/videos                                             ║
    ║   - JSON annotations (camera info, ROI, etc.)                                ║
    ║                                                                              ║
    ║ OUTPUT: ExpertResult with detections, tampering analysis, risk assessment    ║
    ╚══════════════════════════════════════════════════════════════════════════════╝
    """
    inputs = state.get("routes", {}).get("visual_track_integrity", [])
    
    if not inputs:
        return {"current_step": "visual_integrity_skipped"}
    
    # Separate inputs by type
    images = [i for i in inputs if i.input_type == InputType.IMAGE]
    videos = [i for i in inputs if i.input_type == InputType.VIDEO]
    json_data = [i for i in inputs if i.input_type == InputType.JSON]
    
    # Import and use actual visual integrity expert
    from experts.visual_integrity_expert import VisualIntegrityExpert
    
    expert = VisualIntegrityExpert()
    analysis_results = []
    alerts = []
    overall_confidence = 0.0
    
    for input_data in images:
        try:
            result = expert.analyze_image(input_data.file_path, metadata=input_data.metadata)
            analysis_results.append(expert.to_dict(result))
            alerts.extend(result.alerts)
            overall_confidence = max(overall_confidence, result.confidence)
        except Exception as e:
            analysis_results.append({
                "file": input_data.file_path,
                "type": "image",
                "status": "error",
                "error": str(e)
            })
            alerts.append(f"Error analyzing image {input_data.file_path}: {str(e)}")
    
    for input_data in videos:
        try:
            result = expert.analyze_video(input_data.file_path, metadata=input_data.metadata)
            analysis_results.append(expert.to_dict(result))
            alerts.extend(result.alerts)
            overall_confidence = max(overall_confidence, result.confidence)
        except Exception as e:
            analysis_results.append({
                "file": input_data.file_path,
                "type": "video",
                "status": "error",
                "error": str(e)
            })
            alerts.append(f"Error analyzing video {input_data.file_path}: {str(e)}")
    
    expert_result = ExpertResult(
        expert_name="Visual Track Integrity Expert",
        status="success",
        output=analysis_results,
        confidence=overall_confidence,
        alerts=alerts,
        metadata={
            "image_count": len(images),
            "video_count": len(videos),
            "json_count": len(json_data),
            "implemented": True
        }
    )
    
    return {
        "visual_integrity_result": expert_result,
        "all_results": [expert_result],
        "current_step": "visual_integrity_complete"
    }


def thermal_anomaly_node(state: OrchestratorState) -> Dict[str, Any]:
    """
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║                THERMAL ANOMALY INTERPRETATION EXPERT                         ║
    ║                              [IMPLEMENTED]                                   ║
    ╠══════════════════════════════════════════════════════════════════════════════╣
    ║ INPUT: LiDAR and Thermal data:                                               ║
    ║   - LAZ/LAS/NPY/PCD/PLY files (LiDAR point clouds)                           ║
    ║   - IR/Thermal imagery                                                       ║
    ║                                                                              ║
    ║ OUTPUT: ExpertResult with point cloud analysis, tampering detection, alerts  ║
    ╚══════════════════════════════════════════════════════════════════════════════╝
    """
    inputs = state.get("routes", {}).get("thermal_anomaly_interpretation", [])
    
    if not inputs:
        return {"current_step": "thermal_anomaly_skipped"}
    
    # Separate inputs by type
    lidar_data = [i for i in inputs if i.input_type in [InputType.LAZ, InputType.LAS]]
    thermal_data = [i for i in inputs if i.input_type == InputType.THERMAL]
    
    # Import and use actual thermal anomaly expert
    from experts.thermal_anomaly_expert import ThermalAnomalyExpert
    
    expert = ThermalAnomalyExpert()
    analysis_results = []
    alerts = []
    overall_confidence = 0.0
    
    for input_data in lidar_data:
        try:
            result = expert.analyze_lidar(input_data.file_path, input_data.metadata)
            analysis_results.append(expert.to_dict(result))
            alerts.extend(result.alerts)
            overall_confidence = max(overall_confidence, result.confidence)
        except Exception as e:
            analysis_results.append({
                "file": input_data.file_path,
                "type": "lidar",
                "status": "error",
                "error": str(e)
            })
            alerts.append(f"Error analyzing LiDAR {input_data.file_path}: {str(e)}")
    
    for input_data in thermal_data:
        try:
            result = expert.analyze_thermal_image(input_data.file_path, input_data.metadata)
            analysis_results.append(expert.to_dict(result))
            alerts.extend(result.alerts)
            overall_confidence = max(overall_confidence, result.confidence)
        except Exception as e:
            analysis_results.append({
                "file": input_data.file_path,
                "type": "thermal",
                "status": "error",
                "error": str(e)
            })
            alerts.append(f"Error analyzing thermal {input_data.file_path}: {str(e)}")
    expert_result = ExpertResult(
        expert_name="Thermal Anomaly Interpretation Expert",
        status="success",
        output=analysis_results,
        confidence=overall_confidence,
        alerts=alerts,
        metadata={
            "lidar_count": len(lidar_data),
            "thermal_count": len(thermal_data),
            "implemented": True
        }
    )

    return {
        "thermal_anomaly_result": expert_result,
        "all_results": [expert_result],
        "current_step": "thermal_anomaly_complete"
    }


def contextual_reasoning_node(state: OrchestratorState) -> Dict[str, Any]:
    """
        INPUT: Natural language queries (text)
        FUNCTIONALITY:                                                               ║
       - Processes natural language queries from operators             
       - Aggregates context from all other expert outputs              
       - Uses Gemini API for intelligent reasoning                     
       - Provides insights, recommendations, and risk assessment       
    
    """
    inputs = state.get("routes", {}).get("contextual_reasoning", [])
    
    if not inputs:
        return {"current_step": "contextual_reasoning_skipped"}
    
    # Import the Gemini-based reasoning expert
    from experts.contextual_reasoning_expert import process_with_gemini
    
    # Gather context from other expert results
    context = {
        "track_structural": _serialize_result(state.get("track_structural_result")),
        "visual_integrity": _serialize_result(state.get("visual_integrity_result")),
        "thermal_anomaly": _serialize_result(state.get("thermal_anomaly_result")),
    }
    
    # Extract queries
    queries = [i.content for i in inputs if i.content]
    
    # Process with Gemini
    analysis_results = process_with_gemini(queries, context)
    
    expert_result = ExpertResult(
        expert_name="Contextual and Reasoning Expert",
        status="success",
        output=analysis_results,
        confidence=analysis_results.get("confidence", 0.0),
        alerts=analysis_results.get("alerts", []),
        metadata={"query_count": len(queries), "llm_used": "gemini", "implemented": True}
    )
    
    return {
        "contextual_reasoning_result": expert_result,
        "all_results": [expert_result],
        "current_step": "contextual_reasoning_complete"
    }


def aggregate_results_node(state: OrchestratorState) -> Dict[str, Any]:
    """Aggregate all expert results into a final report."""
    all_results = state.get("all_results", [])
    errors = state.get("errors", [])
    
    # Collect all alerts
    all_alerts = []
    for result in all_results:
        if result and result.alerts:
            all_alerts.extend(result.alerts)
    
    # Build final report
    final_report = {
        "timestamp": None,  # Set by caller
        "summary": {
            "experts_invoked": len(all_results),
            "total_alerts": len(all_alerts),
            "errors": len(errors),
        },
        "expert_results": {
            "track_structural": _serialize_result(state.get("track_structural_result")),
            "visual_integrity": _serialize_result(state.get("visual_integrity_result")),
            "thermal_anomaly": _serialize_result(state.get("thermal_anomaly_result")),
            "contextual_reasoning": _serialize_result(state.get("contextual_reasoning_result")),
        },
        "alerts": all_alerts,
        "errors": errors,
        "overall_status": "success" if not errors else "partial_success",
    }
    
    return {
        "final_report": final_report,
        "current_step": "aggregation_complete"
    }


def _serialize_result(result: Optional[ExpertResult]) -> Optional[Dict[str, Any]]:
    """Serialize an ExpertResult to a dictionary."""
    if result is None:
        return None
    return {
        "expert_name": result.expert_name,
        "status": result.status,
        "output": result.output,
        "confidence": result.confidence,
        "alerts": result.alerts,
        "metadata": result.metadata,
    }


# ==================== Conditional Edge Functions ====================

def route_to_experts(state: OrchestratorState) -> List[str]:
    """Determine which expert nodes to invoke based on parsed inputs."""
    active = state.get("active_experts", [])
    nodes_to_invoke = []
    
    if "track_structural_analysis" in active:
        nodes_to_invoke.append("track_structural_analysis")
    if "visual_track_integrity" in active:
        nodes_to_invoke.append("visual_track_integrity")
    if "thermal_anomaly_interpretation" in active:
        nodes_to_invoke.append("thermal_anomaly")
    if "contextual_reasoning" in active:
        nodes_to_invoke.append("contextual_reasoning")
    
    if not nodes_to_invoke:
        return ["aggregate_results"]
    
    return nodes_to_invoke


# ==================== Graph Construction ====================

def create_orchestration_graph() -> StateGraph:
    """
    Create the LangGraph-based orchestration workflow.
    
    Graph Structure:
    START → parse_inputs → route_inputs → [expert nodes in parallel] → aggregate_results → END
    """
    graph = StateGraph(OrchestratorState)
    
    # Add nodes
    graph.add_node("parse_inputs", parse_inputs_node)
    graph.add_node("route_inputs", route_inputs_node)
    graph.add_node("track_structural_analysis", track_structural_analysis_node)
    graph.add_node("visual_track_integrity", visual_track_integrity_node)
    graph.add_node("thermal_anomaly", thermal_anomaly_node)
    graph.add_node("contextual_reasoning", contextual_reasoning_node)
    graph.add_node("aggregate_results", aggregate_results_node)
    
    # Add edges
    graph.add_edge(START, "parse_inputs")
    graph.add_edge("parse_inputs", "route_inputs")
    
    # Conditional routing to experts
    graph.add_conditional_edges(
        "route_inputs",
        route_to_experts,
        {
            "track_structural_analysis": "track_structural_analysis",
            "visual_track_integrity": "visual_track_integrity",
            "thermal_anomaly": "thermal_anomaly",
            "contextual_reasoning": "contextual_reasoning",
            "aggregate_results": "aggregate_results",
        }
    )
    
    # All expert nodes lead to aggregation
    graph.add_edge("track_structural_analysis", "aggregate_results")
    graph.add_edge("visual_track_integrity", "aggregate_results")
    graph.add_edge("thermal_anomaly", "aggregate_results")
    graph.add_edge("contextual_reasoning", "aggregate_results")
    
    graph.add_edge("aggregate_results", END)
    
    return graph


def compile_orchestrator(checkpointer: bool = False):
    """Compile the orchestration graph into a runnable workflow."""
    graph = create_orchestration_graph()
    
    if checkpointer:
        memory = MemorySaver()
        return graph.compile(checkpointer=memory)
    
    return graph.compile()


# ==================== Main Orchestrator Class ====================

class RailwayTamperingOrchestrator:
    """
    Main orchestrator for the Railway Tampering Detection System.
    
    Usage:
        orchestrator = RailwayTamperingOrchestrator()
        result = orchestrator.process([
            {"file_path": "sensor_data.csv"},
            {"file_path": "cctv_frame.jpg"},
            {"content": "What anomalies were detected today?"}
        ])
    """
    
    def __init__(self, use_checkpointer: bool = False):
        """Initialize the orchestrator with compiled graph."""
        self.workflow = compile_orchestrator(checkpointer=use_checkpointer)
        self.use_checkpointer = use_checkpointer
    
    def process(
        self,
        inputs: List[Dict[str, Any]],
        thread_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a list of inputs through the orchestration pipeline.
        
        Args:
            inputs: List of input dictionaries with file_path, content, input_type, metadata
            thread_id: Thread ID for checkpointing
        
        Returns:
            Final aggregated report from all experts
        """
        initial_state: OrchestratorState = {
            "raw_inputs": inputs,
            "parsed_inputs": [],
            "routes": {},
            "active_experts": [],
            "track_structural_result": None,
            "visual_integrity_result": None,
            "thermal_anomaly_result": None,
            "contextual_reasoning_result": None,
            "all_results": [],
            "final_report": None,
            "current_step": "initialized",
            "errors": [],
        }
        
        config = {}
        if self.use_checkpointer and thread_id:
            config["configurable"] = {"thread_id": thread_id}
        
        final_state = self.workflow.invoke(initial_state, config)
        
        return final_state.get("final_report", {})
    
    def process_csv(self, file_path: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process a CSV file (routes to Track Structural Expert)."""
        return self.process([{"file_path": file_path, "input_type": "csv", "metadata": metadata or {}}])
    
    def process_image(self, file_path: str, json_annotations: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process an image (routes to Visual Integrity Expert)."""
        inputs = [{"file_path": file_path}]
        if json_annotations:
            inputs.append({"content": json_annotations, "input_type": "json"})
        return self.process(inputs)
    
    def process_video(self, file_path: str, json_annotations: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process a video (routes to Visual Integrity Expert)."""
        inputs = [{"file_path": file_path}]
        if json_annotations:
            inputs.append({"content": json_annotations, "input_type": "json"})
        return self.process(inputs)
    
    def process_lidar(self, file_path: str) -> Dict[str, Any]:
        """Process a LiDAR file (routes to Thermal Anomaly Expert)."""
        return self.process([{"file_path": file_path}])
    
    def process_thermal(self, file_path: str) -> Dict[str, Any]:
        """Process thermal data (routes to Thermal Anomaly Expert)."""
        return self.process([{"file_path": file_path, "input_type": "thermal"}])
    
    def query(self, question: str) -> Dict[str, Any]:
        """Process a natural language query (routes to Contextual Reasoning Expert)."""
        return self.process([{"content": question, "input_type": "text"}])
    
    def process_multi_modal(
        self,
        csv_files: List[str] = None,
        image_files: List[str] = None,
        video_files: List[str] = None,
        lidar_files: List[str] = None,
        thermal_files: List[str] = None,
        queries: List[str] = None,
        json_annotations: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Process multiple input types simultaneously."""
        inputs = []
        
        if csv_files:
            inputs.extend([{"file_path": f, "input_type": "csv"} for f in csv_files])
        if image_files:
            inputs.extend([{"file_path": f} for f in image_files])
        if video_files:
            inputs.extend([{"file_path": f} for f in video_files])
        if lidar_files:
            inputs.extend([{"file_path": f} for f in lidar_files])
        if thermal_files:
            inputs.extend([{"file_path": f, "input_type": "thermal"} for f in thermal_files])
        if queries:
            inputs.extend([{"content": q, "input_type": "text"} for q in queries])
        if json_annotations:
            inputs.append({"content": json_annotations, "input_type": "json"})
        
        return self.process(inputs)


# ==================== Example Usage ====================

if __name__ == "__main__":
    print("=" * 60)
    print("Railway Tampering Detection - Orchestration Framework")
    print("=" * 60)
    
    orchestrator = RailwayTamperingOrchestrator()
    
    # Example: Multi-modal processing
    print("\nExample: Multi-Modal Input Processing")
    print("-" * 40)
    
    result = orchestrator.process_multi_modal(
        csv_files=["sensor_data.csv"],
        image_files=["drone_image.jpg"],
        lidar_files=["track_scan.laz"],
        queries=["What are the main risks detected?"]
    )
    
    print(f"Result: {json.dumps(result, indent=2, default=str)}")
