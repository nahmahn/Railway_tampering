"""
Combined Inference Module with Gemini Integration

This module combines outputs from all expert agents and uses Gemini 2.5 Flash
to structure outputs in JSON format, then passes through the contextual reasoning
expert for tampering analysis and action recommendation report generation.

Features:
- Multi-expert output aggregation
- Gemini-based JSON structuring
- Tampering analysis with contextual reasoning
- Action recommendation report generation
- Real-time alerts system
"""

import os
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path

# Import expert modules
from experts.visual_integrity_expert import VisualIntegrityExpert, VisualIntegrityResult
from experts.thermal_anomaly_expert import ThermalAnomalyExpert, ThermalAnomalyResult
from experts.track_structural_expert import TrackStructuralExpert, TrackStructuralAnalysisResult

# Import system prompt for action recommendations
from outputs.analysis import SYSTEM_PROMPT as ACTION_RECOMMENDATION_PROMPT

# Try to import Gemini
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# Load environment variables
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / "experts" / ".env")


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RealTimeAlert:
    """Real-time alert structure."""
    alert_id: str
    timestamp: str
    severity: AlertSeverity
    source_expert: str
    title: str
    message: str
    location: Optional[str] = None
    action_required: bool = False
    acknowledged: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CombinedAnalysisResult:
    """Combined result from all experts."""
    timestamp: str
    session_id: str
    
    # Individual expert results
    visual_result: Optional[Dict[str, Any]] = None
    thermal_result: Optional[Dict[str, Any]] = None
    structural_result: Optional[Dict[str, Any]] = None
    
    # Structured JSON output (from Gemini)
    structured_output: Dict[str, Any] = field(default_factory=dict)
    
    # Contextual reasoning output
    tampering_analysis: Dict[str, Any] = field(default_factory=dict)
    
    # Action recommendation report
    action_report: Dict[str, Any] = field(default_factory=dict)
    
    # Real-time alerts
    alerts: List[RealTimeAlert] = field(default_factory=list)
    
    # Overall assessment
    overall_risk_level: str = "low"
    tampering_detected: bool = False
    confidence: float = 0.0


class GeminiOutputStructurer:
    """Uses Gemini 2.5 Flash to structure expert outputs to JSON format."""
    
    STRUCTURING_PROMPT = """You are an AI assistant that structures railway tampering detection expert outputs into clean, consistent JSON format.
Given the raw outputs from one or more expert analysis systems, structure them into a unified JSON format.
OUTPUT SCHEMA:
{
    "analysis_summary": {
        "timestamp": "ISO timestamp",
        "experts_used": ["list of expert names"],
        "overall_status": "NORMAL | ALERT | CRITICAL",
        "confidence_score": 0.0-1.0
    },
    "detections": [
        {
            "type": "detection type",
            "source": "expert name",
            "confidence": 0.0-1.0,
            "severity": "low | medium | high | critical",
            "location": "if available",
            "details": {}
        }
    ],
    "risk_assessment": {
        "tampering_probability": 0.0-1.0,
        "structural_risk": "low | medium | high | critical",
        "immediate_threat": true/false,
        "risk_factors": ["list of factors"]
    },
    "anomalies": [
        {
            "type": "anomaly type",
            "severity": "severity level",
            "description": "description",
            "recommended_action": "action"
        }
    ],
    "metadata": {
        "processing_time": "if available",
        "data_quality": "assessment",
        "additional_info": {}
    }
}

Ensure all outputs are valid JSON. Extract key information and structure it consistently.
If certain fields are not available from the input, use null or omit them.
"""
    
    def __init__(self):
        """Initialize Gemini client."""
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self.model = None
        
        if GEMINI_AVAILABLE and self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel(self.model_name)
            except Exception as e:
                print(f"Warning: Could not initialize Gemini: {e}")
    
    def structure_output(self, expert_outputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Structure expert outputs using Gemini.
        
        Args:
            expert_outputs: Dictionary with raw outputs from experts
        
        Returns:
            Structured JSON output
        """
        if not self.model:
            # Fallback to manual structuring
            return self._manual_structure(expert_outputs)
        
        try:
            prompt = f"""{self.STRUCTURING_PROMPT}

RAW EXPERT OUTPUTS:
{json.dumps(expert_outputs, indent=2, default=str)}

Please structure this into the JSON format specified above. Return ONLY valid JSON.
"""
            
            response = self.model.generate_content(prompt)
            
            # Extract JSON from response
            response_text = response.text
            
            # Try to parse JSON from response
            try:
                # Find JSON block
                if "```json" in response_text:
                    json_str = response_text.split("```json")[1].split("```")[0]
                elif "```" in response_text:
                    json_str = response_text.split("```")[1].split("```")[0]
                else:
                    json_str = response_text
                
                return json.loads(json_str.strip())
            except json.JSONDecodeError:
                return self._manual_structure(expert_outputs)
                
        except Exception as e:
            print(f"Gemini structuring error: {e}")
            return self._manual_structure(expert_outputs)
    
    def _manual_structure(self, expert_outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Manual fallback for structuring outputs."""
        structured = {
            "analysis_summary": {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "experts_used": list(expert_outputs.keys()),
                "overall_status": "NORMAL",
                "confidence_score": 0.0
            },
            "detections": [],
            "risk_assessment": {
                "tampering_probability": 0.0,
                "structural_risk": "low",
                "immediate_threat": False,
                "risk_factors": []
            },
            "anomalies": [],
            "metadata": {}
        }
        
        max_confidence = 0.0
        any_tampering = False
        
        # Process visual results
        if "visual" in expert_outputs and expert_outputs["visual"]:
            visual = expert_outputs["visual"]
            if visual.get("tampering", {}).get("tampering_probability", 0) > 0:
                structured["risk_assessment"]["tampering_probability"] = max(
                    structured["risk_assessment"]["tampering_probability"],
                    visual["tampering"]["tampering_probability"]
                )
                if visual["tampering"].get("person_on_track"):
                    structured["risk_assessment"]["risk_factors"].append("Person detected on track")
                    any_tampering = True
                if visual["tampering"].get("foreign_object_on_track"):
                    structured["risk_assessment"]["risk_factors"].append("Foreign object on track")
                    any_tampering = True
            
            for detection in visual.get("detections", []):
                structured["detections"].append({
                    "type": detection.get("type"),
                    "source": "Visual Integrity Expert",
                    "confidence": detection.get("confidence", 0),
                    "severity": "high" if detection.get("on_track") else "low",
                    "details": detection
                })
            
            max_confidence = max(max_confidence, visual.get("risk_assessment", {}).get("confidence", 0))
        
        # Process thermal/LiDAR results
        if "thermal" in expert_outputs and expert_outputs["thermal"]:
            thermal = expert_outputs["thermal"]
            if thermal.get("result", {}).get("tampering_detected"):
                any_tampering = True
                structured["risk_assessment"]["risk_factors"].append("LiDAR tampering detected")
            
            for anomaly in thermal.get("analysis", {}).get("anomalies", []):
                structured["anomalies"].append({
                    "type": anomaly.get("type"),
                    "severity": anomaly.get("severity"),
                    "description": anomaly.get("description"),
                    "recommended_action": anomaly.get("recommended_action")
                })
            
            max_confidence = max(max_confidence, thermal.get("result", {}).get("confidence_score", 0))
        
        # Process structural results
        if "structural" in expert_outputs and expert_outputs["structural"]:
            structural = expert_outputs["structural"]
            if structural.get("result", {}).get("tampering_detected"):
                any_tampering = True
                structured["risk_assessment"]["risk_factors"].append("Vibration anomaly detected")
            
            max_confidence = max(max_confidence, structural.get("result", {}).get("confidence", 0))
        
        # Set overall status
        structured["analysis_summary"]["confidence_score"] = max_confidence
        if any_tampering:
            structured["analysis_summary"]["overall_status"] = "CRITICAL" if max_confidence > 0.7 else "ALERT"
            structured["risk_assessment"]["immediate_threat"] = max_confidence > 0.7
        
        return structured


class ContextualReasoningEngine:
    """Uses Gemini for tampering analysis and contextual reasoning."""
    
    TAMPERING_ANALYSIS_PROMPT = """You are an expert AI system for railway track tampering detection analysis.

Analyze the following structured detection data and provide:
1. Detailed tampering assessment
2. Root cause analysis
3. Severity classification
4. Confidence assessment

Be specific, technical, and focus on actionable insights.

DETECTION DATA:
{data}

Provide your analysis in the following JSON format:
{{
    "tampering_assessment": {{
        "is_tampering_detected": true/false,
        "tampering_type": "type if detected",
        "confidence": 0.0-1.0,
        "evidence": ["list of evidence"],
        "affected_areas": ["list of affected areas"]
    }},
    "root_cause_analysis": {{
        "probable_cause": "description",
        "contributing_factors": ["list"],
        "natural_vs_deliberate": "assessment"
    }},
    "severity_classification": {{
        "tier": 1-4,
        "description": "tier description",
        "immediate_risk": true/false
    }},
    "key_findings": ["list of key findings"],
    "uncertainties": ["list of uncertainties"]
}}
"""
    
    def __init__(self):
        """Initialize reasoning engine."""
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self.model = None
        
        if GEMINI_AVAILABLE and self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel(self.model_name)
            except Exception as e:
                print(f"Warning: Could not initialize Gemini: {e}")
    
    def analyze_tampering(self, structured_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform tampering analysis using Gemini.
        
        Args:
            structured_data: Structured detection data
        
        Returns:
            Tampering analysis results
        """
        if not self.model:
            return self._fallback_analysis(structured_data)
        
        try:
            prompt = self.TAMPERING_ANALYSIS_PROMPT.format(
                data=json.dumps(structured_data, indent=2, default=str)
            )
            
            response = self.model.generate_content(prompt)
            response_text = response.text
            
            # Extract JSON
            try:
                if "```json" in response_text:
                    json_str = response_text.split("```json")[1].split("```")[0]
                elif "```" in response_text:
                    json_str = response_text.split("```")[1].split("```")[0]
                else:
                    json_str = response_text
                
                return json.loads(json_str.strip())
            except json.JSONDecodeError:
                return self._fallback_analysis(structured_data)
                
        except Exception as e:
            print(f"Tampering analysis error: {e}")
            return self._fallback_analysis(structured_data)
    
    def _fallback_analysis(self, structured_data: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback tampering analysis."""
        risk = structured_data.get("risk_assessment", {})
        
        is_tampering = risk.get("tampering_probability", 0) > 0.3 or risk.get("immediate_threat", False)
        
        tier = 1
        if risk.get("immediate_threat"):
            tier = 4
        elif risk.get("tampering_probability", 0) > 0.7:
            tier = 3
        elif risk.get("tampering_probability", 0) > 0.4:
            tier = 2
        
        return {
            "tampering_assessment": {
                "is_tampering_detected": is_tampering,
                "tampering_type": "Unknown" if is_tampering else None,
                "confidence": risk.get("tampering_probability", 0),
                "evidence": risk.get("risk_factors", []),
                "affected_areas": []
            },
            "root_cause_analysis": {
                "probable_cause": "Requires further investigation",
                "contributing_factors": risk.get("risk_factors", []),
                "natural_vs_deliberate": "Unknown"
            },
            "severity_classification": {
                "tier": tier,
                "description": f"Tier {tier} - {'Critical' if tier == 4 else 'High' if tier == 3 else 'Medium' if tier == 2 else 'Low'} severity",
                "immediate_risk": tier >= 3
            },
            "key_findings": risk.get("risk_factors", []),
            "uncertainties": ["Limited data for full analysis"]
        }


class ActionRecommendationGenerator:
    """Generates action recommendation reports using Gemini."""
    
    def __init__(self):
        """Initialize recommendation generator."""
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self.model = None
        self.system_prompt = ACTION_RECOMMENDATION_PROMPT
        
        if GEMINI_AVAILABLE and self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel(self.model_name)
            except Exception as e:
                print(f"Warning: Could not initialize Gemini: {e}")
    
    def generate_report(
        self,
        structured_data: Dict[str, Any],
        tampering_analysis: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Generate action recommendation report.
        
        Args:
            structured_data: Structured detection data
            tampering_analysis: Tampering analysis results
            context: Additional context (location, time to train, etc.)
        
        Returns:
            Action recommendation report
        """
        if not self.model:
            return self._fallback_report(structured_data, tampering_analysis, context)
        
        try:
            prompt = f"""{self.system_prompt}

ANALYSIS DATA:
{json.dumps(structured_data, indent=2, default=str)}

TAMPERING ANALYSIS:
{json.dumps(tampering_analysis, indent=2, default=str)}

CONTEXT:
{json.dumps(context or {}, indent=2, default=str)}

Generate the Action Recommendation Report following the exact structure specified above.
Return the report in JSON format.
"""
            
            response = self.model.generate_content(prompt)
            response_text = response.text
            
            # Try to parse as JSON or return as structured text
            try:
                if "```json" in response_text:
                    json_str = response_text.split("```json")[1].split("```")[0]
                    return json.loads(json_str.strip())
                else:
                    # Return as text report
                    return {
                        "report_type": "text",
                        "content": response_text,
                        "timestamp": datetime.utcnow().isoformat() + "Z"
                    }
            except json.JSONDecodeError:
                return {
                    "report_type": "text",
                    "content": response_text,
                    "timestamp": datetime.utcnow().isoformat() + "Z"
                }
                
        except Exception as e:
            print(f"Report generation error: {e}")
            return self._fallback_report(structured_data, tampering_analysis, context)
    
    def _fallback_report(
        self,
        structured_data: Dict[str, Any],
        tampering_analysis: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Fallback report generation."""
        severity = tampering_analysis.get("severity_classification", {})
        assessment = tampering_analysis.get("tampering_assessment", {})
        
        tier = severity.get("tier", 1)
        
        # Generate recommendations based on tier
        actions = []
        if tier == 4:
            actions = [
                {"action": "EMERGENCY STOP", "owner": "Train Control", "urgency": "T+0", "impact": "Full line stoppage"},
                {"action": "Evacuate section", "owner": "Station Staff", "urgency": "T+5min", "impact": "Passenger safety"},
                {"action": "Notify Law Enforcement", "owner": "Control Center", "urgency": "T+5min", "impact": "Investigation"}
            ]
        elif tier == 3:
            actions = [
                {"action": "Speed Restriction 20km/h", "owner": "Train Control", "urgency": "T+5min", "impact": "Service delay"},
                {"action": "Deploy Inspection Team", "owner": "Maintenance", "urgency": "T+15min", "impact": "Verification"}
            ]
        elif tier == 2:
            actions = [
                {"action": "Speed Restriction 60km/h", "owner": "Train Control", "urgency": "T+30min", "impact": "Minor delay"},
                {"action": "Schedule Inspection", "owner": "Maintenance", "urgency": "T+2hr", "impact": "Planned check"}
            ]
        else:
            actions = [
                {"action": "Continue Monitoring", "owner": "Control Center", "urgency": "Ongoing", "impact": "None"}
            ]
        
        return {
            "incident_snapshot": {
                "alert_id": f"ALERT-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "location": context.get("location", "Unknown") if context else "Unknown",
                "severity_tier": tier,
                "time_to_next_train": context.get("time_to_train", "Unknown") if context else "Unknown"
            },
            "what_system_sees": {
                "anomaly_scores": structured_data.get("risk_assessment", {}),
                "modalities_triggered": structured_data.get("analysis_summary", {}).get("experts_used", []),
                "fusion_confidence": structured_data.get("analysis_summary", {}).get("confidence_score", 0),
                "tampering_likelihood": assessment.get("confidence", 0)
            },
            "why_this_matters": {
                "risk_explanation": assessment.get("evidence", []),
                "immediate_risk": severity.get("immediate_risk", False)
            },
            "recommended_actions": actions,
            "decision_authority": {
                "decision_owner": "Shift Supervisor" if tier >= 3 else "Control Operator",
                "escalation_required": tier >= 3
            },
            "next_review_trigger": {
                "conditions": [
                    f"Re-evaluate if confidence exceeds 0.8",
                    f"Escalate if additional anomalies detected",
                    f"Close if inspection confirms no issue"
                ]
            }
        }


class RealTimeAlertManager:
    """Manages real-time alerts generation and distribution."""
    
    def __init__(self):
        """Initialize alert manager."""
        self.alert_queue: List[RealTimeAlert] = []
        self.alert_handlers: List[callable] = []
    
    def register_handler(self, handler: callable):
        """Register an alert handler callback."""
        self.alert_handlers.append(handler)
    
    def generate_alerts(
        self,
        structured_data: Dict[str, Any],
        tampering_analysis: Dict[str, Any]
    ) -> List[RealTimeAlert]:
        """
        Generate real-time alerts from analysis results.
        
        Args:
            structured_data: Structured detection data
            tampering_analysis: Tampering analysis results
        
        Returns:
            List of generated alerts
        """
        alerts = []
        timestamp = datetime.utcnow().isoformat() + "Z"
        
        severity_map = {
            4: AlertSeverity.CRITICAL,
            3: AlertSeverity.HIGH,
            2: AlertSeverity.WARNING,
            1: AlertSeverity.INFO
        }
        
        tier = tampering_analysis.get("severity_classification", {}).get("tier", 1)
        severity = severity_map.get(tier, AlertSeverity.INFO)
        
        # Main tampering alert
        if tampering_analysis.get("tampering_assessment", {}).get("is_tampering_detected"):
            alert = RealTimeAlert(
                alert_id=f"TAMPER-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
                timestamp=timestamp,
                severity=severity,
                source_expert="Combined Analysis",
                title=f"Tampering Detected - Tier {tier}",
                message=f"Tampering evidence: {', '.join(tampering_analysis.get('tampering_assessment', {}).get('evidence', []))}",
                action_required=tier >= 2
            )
            alerts.append(alert)
        
        # Individual detection alerts
        for detection in structured_data.get("detections", []):
            if detection.get("severity") in ["high", "critical"]:
                alert = RealTimeAlert(
                    alert_id=f"DET-{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')[:17]}",
                    timestamp=timestamp,
                    severity=AlertSeverity.HIGH if detection["severity"] == "high" else AlertSeverity.CRITICAL,
                    source_expert=detection.get("source", "Unknown"),
                    title=f"Detection: {detection.get('type', 'Unknown')}",
                    message=f"Confidence: {detection.get('confidence', 0):.1%}",
                    action_required=True
                )
                alerts.append(alert)
        
        # Anomaly alerts
        for anomaly in structured_data.get("anomalies", []):
            if anomaly.get("severity") in ["high", "critical"]:
                alert = RealTimeAlert(
                    alert_id=f"ANOM-{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')[:17]}",
                    timestamp=timestamp,
                    severity=AlertSeverity.HIGH if anomaly["severity"] == "high" else AlertSeverity.CRITICAL,
                    source_expert="Anomaly Detection",
                    title=f"Anomaly: {anomaly.get('type', 'Unknown')}",
                    message=anomaly.get("description", ""),
                    action_required=True
                )
                alerts.append(alert)
        
        # Dispatch to handlers
        for alert in alerts:
            self.alert_queue.append(alert)
            for handler in self.alert_handlers:
                try:
                    handler(alert)
                except Exception as e:
                    print(f"Alert handler error: {e}")
        
        return alerts
    
    def get_pending_alerts(self) -> List[RealTimeAlert]:
        """Get all unacknowledged alerts."""
        return [a for a in self.alert_queue if not a.acknowledged]
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        for alert in self.alert_queue:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                return True
        return False


class CombinedInferenceEngine:
    """Main engine that combines all experts and produces final analysis."""
    
    def __init__(self):
        """Initialize combined inference engine."""
        self.visual_expert = VisualIntegrityExpert()
        self.thermal_expert = ThermalAnomalyExpert()
        self.structural_expert = TrackStructuralExpert(
            xgb_model_path="experts/xgb_metro_model.pkl",
            scaler_path="experts/scaler_metro.pkl"
        )
        
        self.output_structurer = GeminiOutputStructurer()
        self.reasoning_engine = ContextualReasoningEngine()
        self.report_generator = ActionRecommendationGenerator()
        self.alert_manager = RealTimeAlertManager()
        
        self._session_counter = 0
    
    def _generate_session_id(self) -> str:
        """Generate unique session ID."""
        self._session_counter += 1
        return f"SESSION-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}-{self._session_counter:04d}"
    
    def analyze(
        self,
        image_paths: List[str] = None,
        video_paths: List[str] = None,
        lidar_paths: List[str] = None,
        csv_paths: List[str] = None,
        context: Dict[str, Any] = None
    ) -> CombinedAnalysisResult:
        """
        Run combined analysis across all experts.
        
        Args:
            image_paths: List of image file paths for visual analysis
            video_paths: List of video file paths for visual analysis
            lidar_paths: List of LiDAR file paths for thermal/structural analysis
            csv_paths: List of CSV file paths for vibration analysis
            context: Additional context (location, train schedule, etc.)
        
        Returns:
            CombinedAnalysisResult with all analysis results
        """
        result = CombinedAnalysisResult(
            timestamp=datetime.utcnow().isoformat() + "Z",
            session_id=self._generate_session_id()
        )
        
        expert_outputs = {}
        
        # Run visual analysis
        if image_paths or video_paths:
            visual_results = []
            
            for img_path in (image_paths or []):
                try:
                    img_result = self.visual_expert.analyze_image(img_path)
                    result_dict = self.visual_expert.to_dict(img_result)
                    result_dict["file_url"] = img_path  # Add file URL for frontend
                    visual_results.append(result_dict)
                except Exception as e:
                    visual_results.append({"file": img_path, "error": str(e), "file_url": img_path})
            
            for vid_path in (video_paths or []):
                try:
                    vid_result = self.visual_expert.analyze_video(vid_path)
                    result_dict = self.visual_expert.to_dict(vid_result)
                    result_dict["file_url"] = vid_path  # Add file URL for frontend
                    visual_results.append(result_dict)
                except Exception as e:
                    visual_results.append({"file": vid_path, "error": str(e), "file_url": vid_path})
            
            if visual_results:
                result.visual_result = visual_results[0] if len(visual_results) == 1 else visual_results
                expert_outputs["visual"] = result.visual_result
        
        # Run thermal/LiDAR analysis
        if lidar_paths:
            thermal_results = []
            
            for lidar_path in lidar_paths:
                try:
                    lidar_result = self.thermal_expert.analyze_lidar(lidar_path)
                    thermal_results.append(self.thermal_expert.to_dict(lidar_result))
                except Exception as e:
                    thermal_results.append({"file": lidar_path, "error": str(e)})
            
            if thermal_results:
                result.thermal_result = thermal_results[0] if len(thermal_results) == 1 else thermal_results
                expert_outputs["thermal"] = result.thermal_result
        
        # Run structural/vibration analysis
        if csv_paths:
            structural_results = []
            
            try:
                # Use analyze_files to handle potential merging of x/y/z files
                structural_results_objs = self.structural_expert.analyze_files(csv_paths)
                structural_results = [self.structural_expert.to_dict(r) for r in structural_results_objs]
            except Exception as e:
                structural_results.append({"error": str(e)})
            
            if structural_results:
                result.structural_result = structural_results[0] if len(structural_results) == 1 else structural_results
                expert_outputs["structural"] = result.structural_result
        
        # Structure outputs using Gemini
        result.structured_output = self.output_structurer.structure_output(expert_outputs)
        
        # Perform tampering analysis
        result.tampering_analysis = self.reasoning_engine.analyze_tampering(result.structured_output)
        
        # Generate action recommendation report
        result.action_report = self.report_generator.generate_report(
            result.structured_output,
            result.tampering_analysis,
            context
        )
        
        # Generate real-time alerts
        result.alerts = self.alert_manager.generate_alerts(
            result.structured_output,
            result.tampering_analysis
        )
        
        # Set overall assessment
        assessment = result.tampering_analysis.get("tampering_assessment", {})
        result.tampering_detected = assessment.get("is_tampering_detected", False)
        result.confidence = assessment.get("confidence", 0)
        
        # Aggregate severity from all experts to ensure we don't miss high/critical alerts
        severity_rank = {"critical": 4, "high": 3, "medium": 2, "low": 1}
        max_tier = result.tampering_analysis.get("severity_classification", {}).get("tier", 1)
        
        # Check individual results for higher severity
        for exp_name, exp_res in expert_outputs.items():
            res_list = exp_res if isinstance(exp_res, list) else [exp_res]
            for r in res_list:
                # Check visual/structural risk assessment fields
                r_level = (r.get("risk_assessment", {}).get("risk_level") or 
                           r.get("result", {}).get("risk_level") or 
                           "low")
                max_tier = max(max_tier, severity_rank.get(r_level.lower(), 1))
                
                # Check for "CRITICAL" keywords in alerts
                alerts_list = r.get("alerts", [])
                if any("CRITICAL" in str(a).upper() for a in alerts_list):
                    max_tier = max(max_tier, 4)
                elif any("HIGH" in str(a).upper() for a in alerts_list):
                    max_tier = max(max_tier, 3)

        result.overall_risk_level = {4: "critical", 3: "high", 2: "medium", 1: "low"}.get(max_tier, "low")
        
        return result
    
    def to_dict(self, result: CombinedAnalysisResult) -> Dict[str, Any]:
        """Convert CombinedAnalysisResult to dictionary and ensure serialization safety."""
        
        def recursive_sanitize(obj):
            if isinstance(obj, dict):
                return {k: recursive_sanitize(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [recursive_sanitize(v) for v in obj]
            elif hasattr(obj, 'item'):  # Handle numpy types (bool_, float32, etc.)
                return obj.item()
            elif hasattr(obj, 'tolist'):  # Handle numpy arrays
                return obj.tolist()
            return obj

        raw_dict = {
            "timestamp": result.timestamp,
            "session_id": result.session_id,
            "expert_results": {
                "visual": result.visual_result,
                "thermal": result.thermal_result,
                "structural": result.structural_result
            },
            "structured_output": result.structured_output,
            "tampering_analysis": result.tampering_analysis,
            "action_report": result.action_report,
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
                for a in result.alerts
            ],
            "overall_assessment": {
                "risk_level": result.overall_risk_level,
                "tampering_detected": result.tampering_detected,
                "confidence": result.confidence
            }
        }
        
        return recursive_sanitize(raw_dict)


# ==================== Example Usage ====================

if __name__ == "__main__":
    print("=" * 60)
    print("Combined Inference Engine - Test")
    print("=" * 60)
    
    print(f"Gemini Available: {GEMINI_AVAILABLE}")
    
    engine = CombinedInferenceEngine()
    
    # Test with sample data (if available)
    print("\nCombined Inference Engine initialized successfully")
    print("Use engine.analyze() with file paths to run analysis")
