"""
Thermal Anomaly Interpretation Expert - IMPLEMENTED

Uses LiDAR point cloud processing for structural analysis and tampering detection.
Based on the lidar/ folder implementation.

INPUT TYPES:
- LAZ/LAS files: LiDAR point cloud data
- NPY files: NumPy point cloud arrays
- PCD/PLY files: Standard point cloud formats

EXPECTED FUNCTIONALITY:
- Point cloud processing and analysis
- 3D track geometry reconstruction
- Structural deviation detection
- Debris and obstruction detection
- Gauge anomaly detection

OUTPUT:
- Return ExpertResult with anomalies, structural analysis, and alerts
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import os
import json
import numpy as np
from datetime import datetime
from pathlib import Path

# Import LiDAR module components
import sys
lidar_path = Path(__file__).parent / "lidar"
if str(lidar_path) not in sys.path:
    sys.path.insert(0, str(lidar_path))

try:
    from lidar.point_cloud_processor import PointCloudProcessor
    from lidar.tampering_detector import TamperingDetector
    from lidar.output_formatter import OutputFormatter, AlertGenerator
    LIDAR_AVAILABLE = True
except ImportError:
    try:
        from lidar.point_cloud_processor import PointCloudProcessor
        from lidar.tampering_detector import TamperingDetector
        from lidar.output_formatter import OutputFormatter, AlertGenerator
        LIDAR_AVAILABLE = True
    except ImportError:
        LIDAR_AVAILABLE = False


class AnomalyType(Enum):
    """Types of thermal/LiDAR anomalies."""
    HOTSPOT = "hotspot"
    COLDSPOT = "coldspot"
    TEMPERATURE_GRADIENT = "temperature_gradient"
    STRUCTURAL_DEVIATION = "structural_deviation"
    MISSING_STRUCTURE = "missing_structure"
    OBSTRUCTION_3D = "obstruction_3d"
    SUBSIDENCE = "subsidence"
    ELEVATION_ANOMALY = "elevation_anomaly"
    POINT_DENSITY_ANOMALY = "point_density_anomaly"
    GAUGE_DEVIATION = "gauge_deviation"
    RAIL_MISALIGNMENT = "rail_misalignment"
    VERTICAL_DISPLACEMENT = "vertical_displacement"
    DEBRIS = "debris"
    SLEEPER_ANOMALY = "sleeper_anomaly"


@dataclass
class ThermalAnomaly:
    """A single thermal/structural anomaly detection."""
    anomaly_type: AnomalyType
    severity: str  # low, medium, high, critical
    location: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    temperature_delta: Optional[float] = None
    confidence: float = 0.0
    description: str = ""
    recommended_action: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LiDARAnomalyResult:
    """Result from LiDAR point cloud analysis."""
    file_path: str
    point_count: int = 0
    coverage_area_sqm: float = 0.0
    
    # Track geometry from point cloud
    rail_profile_deviation: float = 0.0
    sleeper_spacing_variance: float = 0.0
    ballast_level_variance: float = 0.0
    
    # Track gauge analysis
    track_gauge: Dict[str, float] = field(default_factory=dict)
    rail_alignment: Dict[str, float] = field(default_factory=dict)
    rail_continuity: Dict[str, float] = field(default_factory=dict)
    vertical_displacement: Dict[str, float] = field(default_factory=dict)
    
    # Debris analysis
    debris_count: int = 0
    debris_density: float = 0.0
    
    # Sleeper analysis
    sleeper_count: int = 0
    
    # Structural anomalies
    structural_anomalies: List[ThermalAnomaly] = field(default_factory=list)
    obstruction_detected: bool = False
    subsidence_detected: bool = False
    
    # Quality metrics
    scan_quality: float = 0.0
    point_density: float = 0.0


@dataclass
class ThermalAnomalyResult:
    """Combined result structure for thermal/LiDAR analysis."""
    file_path: str
    file_type: str  # "lidar" or "thermal"
    analysis_status: str
    timestamp: str = ""
    
    # Detection results
    tampering_detected: bool = False
    confidence_score: float = 0.0
    severity: str = "low"
    
    # Anomalies
    anomalies: List[ThermalAnomaly] = field(default_factory=list)
    anomaly_count: int = 0
    
    # LiDAR specific results
    lidar_result: Optional[LiDARAnomalyResult] = None
    
    # Features analyzed
    features_analyzed: Dict[str, Any] = field(default_factory=dict)
    
    # Overall assessment
    risk_level: str = "low"
    alerts: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    confidence: float = 0.0


class ThermalAnomalyExpert:
    """
    Thermal Anomaly Interpretation Expert using LiDAR analysis.
    """
    
    def __init__(self, config_path: str = None):
        """Initialize the expert with LiDAR processing pipeline."""
        self.config = {}
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            # Default config from lidar/config.json
            default_config_path = Path(__file__).parent / "lidar" / "config.json"
            if default_config_path.exists():
                with open(default_config_path, 'r') as f:
                    self.config = json.load(f)
        
        # Initialize LiDAR processing components
        if LIDAR_AVAILABLE:
            self.processor = PointCloudProcessor(self.config.get('processing', {}))
            self.detector = TamperingDetector(self.config.get('detection', {}))
            self.formatter = OutputFormatter()
            self.alert_gen = AlertGenerator()
        else:
            self.processor = None
            self.detector = None
            self.formatter = None
            self.alert_gen = None
    
    def analyze_lidar(
        self,
        file_path: str,
        metadata: Dict[str, Any] = None
    ) -> ThermalAnomalyResult:
        """
        Analyze a LiDAR point cloud file for tampering detection.
        
        Args:
            file_path: Path to the point cloud file (.npy, .pcd, .ply, .las, .laz)
            metadata: Additional metadata about the scan
        
        Returns:
            ThermalAnomalyResult with analysis findings
        """
        result = ThermalAnomalyResult(
            file_path=file_path,
            file_type="lidar",
            analysis_status="processing",
            timestamp=datetime.utcnow().isoformat() + "Z"
        )
        
        if not LIDAR_AVAILABLE:
            result.analysis_status = "error"
            result.alerts.append("LiDAR processing modules not available")
            return result
        
        try:
            import time
            start_time = time.time()
            
            # Step 1: Load point cloud
            points = self.processor.load_point_cloud(file_path)
            
            # Step 2: Preprocess
            points = self.processor.preprocess(points)
            
            # Step 3: Segment rails and extract features
            segments = self.processor.segment_rails(points)
            features = self.processor.extract_features(points, segments)
            
            # Step 4: Detect tampering
            detection_result = self.detector.detect_tampering(features, metadata)
            
            processing_time = time.time() - start_time
            
            # Convert detection result to our format
            result.tampering_detected = detection_result['tampering_detected']
            result.confidence_score = detection_result['confidence_score']
            result.severity = detection_result['severity']
            result.anomaly_count = detection_result['anomaly_count']
            result.features_analyzed = detection_result.get('features_analyzed', {})
            
            # Convert anomalies
            for anomaly in detection_result.get('anomalies', []):
                result.anomalies.append(ThermalAnomaly(
                    anomaly_type=self._map_anomaly_type(anomaly['type']),
                    severity=anomaly.get('severity', 'low'),
                    confidence=anomaly.get('confidence', 0.0),
                    description=anomaly.get('description', ''),
                    recommended_action=anomaly.get('recommended_action', ''),
                    details=anomaly.get('details', {})
                ))
            
            # Build LiDAR-specific result
            lidar_result = LiDARAnomalyResult(
                file_path=file_path,
                point_count=len(points),
                track_gauge=features.get('track_gauge', {}),
                rail_alignment=features.get('rail_alignment', {}),
                rail_continuity=features.get('rail_continuity', {}),
                vertical_displacement=features.get('vertical_displacement', {}),
                debris_count=features.get('debris_count', 0),
                debris_density=features.get('debris_density', 0.0),
                sleeper_count=features.get('sleeper_count', 0),
                obstruction_detected=detection_result['tampering_detected'],
                structural_anomalies=result.anomalies
            )
            result.lidar_result = lidar_result
            
            # Calculate risk level
            result.risk_level = self._calculate_risk_level(result)
            result.confidence = result.confidence_score
            
            # Generate alerts
            result.alerts = self._generate_alerts(result)
            result.recommendations = self._generate_recommendations(result)
            
            result.analysis_status = "success"
            
        except FileNotFoundError:
            result.analysis_status = "error"
            result.alerts.append(f"File not found: {file_path}")
        except Exception as e:
            result.analysis_status = "error"
            result.alerts.append(f"Analysis error: {str(e)}")
        
        return result
    
    def analyze_thermal_image(
        self,
        file_path: str,
        metadata: Dict[str, Any] = None
    ) -> ThermalAnomalyResult:
        """
        Analyze a thermal/IR image for temperature anomalies.
        
        This is a placeholder for thermal image processing.
        Currently returns basic analysis structure.
        """
        result = ThermalAnomalyResult(
            file_path=file_path,
            file_type="thermal",
            analysis_status="processing",
            timestamp=datetime.utcnow().isoformat() + "Z"
        )
        
        try:
            # Placeholder for thermal image processing
            # TODO: Implement actual thermal image analysis
            result.analysis_status = "success"
            result.alerts.append("Thermal image analysis: Basic processing only")
            result.recommendations.append("Consider LiDAR analysis for more accurate results")
            
        except Exception as e:
            result.analysis_status = "error"
            result.alerts.append(f"Thermal analysis error: {str(e)}")
        
        return result
    
    def _map_anomaly_type(self, type_str: str) -> AnomalyType:
        """Map string anomaly type to enum."""
        mapping = {
            "gauge_deviation": AnomalyType.GAUGE_DEVIATION,
            "rail_misalignment": AnomalyType.RAIL_MISALIGNMENT,
            "vertical_displacement": AnomalyType.VERTICAL_DISPLACEMENT,
            "continuity_break": AnomalyType.STRUCTURAL_DEVIATION,
            "debris_obstruction": AnomalyType.DEBRIS,
            "sleeper_missing": AnomalyType.SLEEPER_ANOMALY,
            "structural_deviation": AnomalyType.STRUCTURAL_DEVIATION,
            "obstruction": AnomalyType.OBSTRUCTION_3D,
            "subsidence": AnomalyType.SUBSIDENCE,
        }
        return mapping.get(type_str.lower(), AnomalyType.STRUCTURAL_DEVIATION)
    
    def _calculate_risk_level(self, result: ThermalAnomalyResult) -> str:
        """Determine risk level based on analysis."""
        if result.severity == "critical":
            return "critical"
        if result.severity == "high" or result.confidence_score > 0.7:
            return "high"
        if result.severity == "medium" or result.confidence_score > 0.4:
            return "medium"
        return "low"
    
    def _generate_alerts(self, result: ThermalAnomalyResult) -> List[str]:
        """Generate alerts based on analysis."""
        alerts = []
        
        if result.tampering_detected:
            alerts.append(f"🚨 TAMPERING DETECTED - Severity: {result.severity.upper()}")
        
        for anomaly in result.anomalies:
            if anomaly.severity in ["high", "critical"]:
                alerts.append(f"⚠️ {anomaly.anomaly_type.value}: {anomaly.description}")
        
        return alerts
    
    def _generate_recommendations(self, result: ThermalAnomalyResult) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []
        
        if result.risk_level == "critical":
            recommendations.extend([
                "Immediately halt train operations",
                "Dispatch emergency inspection team",
                "Notify railway control center"
            ])
        elif result.risk_level == "high":
            recommendations.extend([
                "Reduce train speed in affected section",
                "Schedule urgent inspection",
                "Increase monitoring frequency"
            ])
        elif result.risk_level == "medium":
            recommendations.extend([
                "Monitor section closely",
                "Schedule routine inspection"
            ])
        else:
            recommendations.append("Continue normal operations")
        
        # Add specific recommendations from anomalies
        for anomaly in result.anomalies:
            if anomaly.recommended_action:
                recommendations.append(anomaly.recommended_action)
        
        return list(dict.fromkeys(recommendations))  # Remove duplicates
    
    def to_dict(self, result: ThermalAnomalyResult) -> Dict[str, Any]:
        """Convert result to dictionary for JSON serialization."""
        output = {
            "file_path": result.file_path,
            "file_type": result.file_type,
            "analysis_status": result.analysis_status,
            "timestamp": result.timestamp,
            "result": {
                "tampering_detected": result.tampering_detected,
                "confidence_score": result.confidence_score,
                "severity_level": result.severity,
                "status": "TAMPERED" if result.tampering_detected else "NORMAL"
            },
            "analysis": {
                "anomaly_count": result.anomaly_count,
                "anomalies": [
                    {
                        "type": a.anomaly_type.value,
                        "severity": a.severity,
                        "confidence": a.confidence,
                        "description": a.description,
                        "recommended_action": a.recommended_action,
                        "details": a.details
                    }
                    for a in result.anomalies
                ],
                "features": result.features_analyzed
            },
            "risk_assessment": {
                "risk_level": result.risk_level,
                "confidence": result.confidence
            },
            "alerts": result.alerts,
            "recommendations": result.recommendations
        }
        
        if result.lidar_result:
            output["lidar_details"] = {
                "point_count": result.lidar_result.point_count,
                "track_gauge": result.lidar_result.track_gauge,
                "rail_alignment": result.lidar_result.rail_alignment,
                "rail_continuity": result.lidar_result.rail_continuity,
                "debris": {
                    "count": result.lidar_result.debris_count,
                    "density": result.lidar_result.debris_density
                },
                "sleeper_count": result.lidar_result.sleeper_count
            }
        
        return output
    
    def create_sample_data(self, output_dir: str = "./sample_data") -> Path:
        """Create sample LiDAR data for testing."""
        # Note: This method only uses numpy, so it works without LiDAR modules
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate normal track
        normal_points = self._generate_normal_track()
        normal_file = output_path / "normal_track.npy"
        np.save(normal_file, normal_points)
        
        # Generate tampered track
        tampered_points = self._generate_tampered_track()
        tampered_file = output_path / "tampered_track.npy"
        np.save(tampered_file, tampered_points)
        
        return normal_file  # Return path to first file for testing
    
    def _generate_normal_track(self) -> np.ndarray:
        """Generate normal railway track point cloud."""
        np.random.seed(42)
        
        length = 30
        gauge = 1.435
        rail_height = 0.15
        
        x = np.linspace(0, length, 300)
        
        left_rail = np.column_stack([
            x,
            np.full_like(x, -gauge/2) + np.random.normal(0, 0.005, len(x)),
            np.full_like(x, rail_height) + np.random.normal(0, 0.01, len(x))
        ])
        
        right_rail = np.column_stack([
            x,
            np.full_like(x, gauge/2) + np.random.normal(0, 0.005, len(x)),
            np.full_like(x, rail_height) + np.random.normal(0, 0.01, len(x))
        ])
        
        num_sleepers = 60
        sleeper_x = np.linspace(0, length, num_sleepers)
        sleepers = []
        for sx in sleeper_x:
            sy = np.linspace(-gauge/2 - 0.2, gauge/2 + 0.2, 50)
            sleeper_points = np.column_stack([
                np.full_like(sy, sx) + np.random.normal(0, 0.01, len(sy)),
                sy + np.random.normal(0, 0.01, len(sy)),
                np.full_like(sy, 0.05) + np.random.normal(0, 0.01, len(sy))
            ])
            sleepers.append(sleeper_points)
        sleepers = np.vstack(sleepers)
        
        x_ground = np.random.uniform(0, length, 6000)
        y_ground = np.random.uniform(-2, 2, 6000)
        z_ground = np.random.uniform(0, 0.02, 6000)
        ground = np.column_stack([x_ground, y_ground, z_ground])
        
        return np.vstack([left_rail, right_rail, sleepers, ground])
    
    def _generate_tampered_track(self) -> np.ndarray:
        """Generate tampered railway track point cloud."""
        np.random.seed(123)
        
        length = 30
        gauge = 1.435
        rail_height = 0.15
        tamper_start = 15
        tamper_end = 25
        
        x = np.linspace(0, length, 300)
        gauge_variation = np.zeros_like(x)
        tamper_mask = (x >= tamper_start) & (x <= tamper_end)
        gauge_variation[tamper_mask] = 0.08 * np.sin(
            (x[tamper_mask] - tamper_start) * np.pi / (tamper_end - tamper_start)
        )
        
        left_rail = np.column_stack([
            x,
            np.full_like(x, -gauge/2) - gauge_variation/2 + np.random.normal(0, 0.01, len(x)),
            np.full_like(x, rail_height) + np.random.normal(0, 0.015, len(x))
        ])
        
        right_rail = np.column_stack([
            x,
            np.full_like(x, gauge/2) + gauge_variation/2 + np.random.normal(0, 0.01, len(x)),
            np.full_like(x, rail_height) + np.random.normal(0, 0.015, len(x))
        ])
        
        num_sleepers = 60
        sleeper_x = np.linspace(0, length, num_sleepers)
        sleepers = []
        for sx in sleeper_x:
            sy = np.linspace(-gauge/2 - 0.2, gauge/2 + 0.2, 50)
            sleeper_points = np.column_stack([
                np.full_like(sy, sx) + np.random.normal(0, 0.01, len(sy)),
                sy + np.random.normal(0, 0.01, len(sy)),
                np.full_like(sy, 0.05) + np.random.normal(0, 0.01, len(sy))
            ])
            sleepers.append(sleeper_points)
        sleepers = np.vstack(sleepers)
        
        x_ground = np.random.uniform(0, length, 6000)
        y_ground = np.random.uniform(-2, 2, 6000)
        z_ground = np.random.uniform(0, 0.02, 6000)
        ground = np.column_stack([x_ground, y_ground, z_ground])
        
        # Add debris in tampered section
        debris_x = np.random.uniform(tamper_start, tamper_end, 200)
        debris_y = np.random.uniform(-0.5, 0.5, 200)
        debris_z = np.random.uniform(rail_height + 0.05, rail_height + 0.15, 200)
        debris = np.column_stack([debris_x, debris_y, debris_z])
        
        return np.vstack([left_rail, right_rail, sleepers, ground, debris])


# Module-level functions for backward compatibility
_expert_instance = None


def get_expert(config_path: str = None) -> ThermalAnomalyExpert:
    """Get or create the singleton expert instance."""
    global _expert_instance
    if _expert_instance is None:
        _expert_instance = ThermalAnomalyExpert(config_path)
    return _expert_instance


def analyze_lidar(
    file_path: str,
    metadata: Dict[str, Any] = None
) -> ThermalAnomalyResult:
    """Analyze a LiDAR point cloud file."""
    return get_expert().analyze_lidar(file_path, metadata)


def analyze_thermal(
    file_path: str,
    metadata: Dict[str, Any] = None
) -> ThermalAnomalyResult:
    """Analyze a thermal/IR image."""
    return get_expert().analyze_thermal_image(file_path, metadata)


# ==================== Example Usage ====================

if __name__ == "__main__":
    print("=" * 60)
    print("Thermal Anomaly Expert (LiDAR) - Test")
    print("=" * 60)
    
    print(f"LiDAR Modules Available: {LIDAR_AVAILABLE}")
    
    if LIDAR_AVAILABLE:
        expert = ThermalAnomalyExpert()
        
        # Create sample data
        print("\nCreating sample data...")
        sample_dir = expert.create_sample_data("./test_samples")
        
        # Test normal track
        print("\nAnalyzing normal track...")
        result = expert.analyze_lidar(str(sample_dir / "normal_track.npy"))
        print(f"Status: {result.analysis_status}")
        print(f"Tampering Detected: {result.tampering_detected}")
        print(f"Risk Level: {result.risk_level}")
        
        # Test tampered track
        print("\nAnalyzing tampered track...")
        result = expert.analyze_lidar(str(sample_dir / "tampered_track.npy"))
        print(f"Status: {result.analysis_status}")
        print(f"Tampering Detected: {result.tampering_detected}")
        print(f"Risk Level: {result.risk_level}")
        print(f"Alerts: {result.alerts}")
