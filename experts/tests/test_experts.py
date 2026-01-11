"""
Comprehensive Test Suite for Railway Tampering Detection Experts

Tests cover:
1. Visual Integrity Expert
2. Thermal Anomaly Expert  
3. Track Structural Expert
4. Contextual Reasoning Expert
5. Orchestration
6. Combined Inference

Run with: pytest tests/test_experts.py -v
"""

import pytest
import os
import sys
import json
import tempfile
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports - bypass __init__.py
parent_dir = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, parent_dir)

# Mock tensorflow to avoid import issues
sys.modules['tensorflow'] = type(sys)('tensorflow')


# ==================== Fixtures ====================

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_csv_vibration(temp_dir):
    """Create a sample vibration CSV file."""
    np.random.seed(42)
    n_samples = 500
    
    data = {
        'x': np.random.randn(n_samples) * 0.1,
        'y': np.random.randn(n_samples) * 0.1,
        'z': np.random.randn(n_samples) * 0.1 + 9.8,  # gravity
    }
    
    df = pd.DataFrame(data)
    filepath = os.path.join(temp_dir, "vibration_test.csv")
    df.to_csv(filepath, index=False)
    return filepath


@pytest.fixture
def sample_csv_geometric(temp_dir):
    """Create a sample geometric sensor CSV file."""
    np.random.seed(42)
    n_samples = 100
    
    data = {
        'gauge': np.random.randn(n_samples) * 0.5 + 1435,  # Standard gauge ~1435mm
        'alignment': np.random.randn(n_samples) * 0.3,
        'cant': np.random.randn(n_samples) * 2,
        'curvature': np.random.randn(n_samples) * 0.001,
        'twist': np.random.randn(n_samples) * 0.5,
    }
    
    df = pd.DataFrame(data)
    filepath = os.path.join(temp_dir, "geometric_test.csv")
    df.to_csv(filepath, index=False)
    return filepath


@pytest.fixture
def sample_image(temp_dir):
    """Create a sample test image."""
    try:
        from PIL import Image
        img = Image.new('RGB', (640, 480), color=(128, 128, 128))
        # Add some variation
        pixels = img.load()
        for i in range(100, 200):
            for j in range(200, 300):
                pixels[i, j] = (50, 50, 50)  # darker region
        
        filepath = os.path.join(temp_dir, "test_image.jpg")
        img.save(filepath)
        return filepath
    except ImportError:
        pytest.skip("PIL not available")


@pytest.fixture
def sample_lidar_npy(temp_dir):
    """Create a sample LiDAR point cloud as NPY file."""
    np.random.seed(42)
    n_points = 1000
    
    # Create simple track-like point cloud
    x = np.random.uniform(-10, 10, n_points)
    y = np.random.uniform(0, 100, n_points)  # along track
    z = np.random.uniform(0, 2, n_points)
    
    points = np.column_stack([x, y, z])
    
    filepath = os.path.join(temp_dir, "test_pointcloud.npy")
    np.save(filepath, points)
    return filepath


# ==================== Track Structural Expert Tests ====================

class TestTrackStructuralExpert:
    """Tests for Track Structural Analysis Expert."""
    
    def test_import(self):
        """Test that module can be imported."""
        from experts.track_structural_expert import (
            TrackStructuralExpert,
            TrackStructuralAnalysisResult,
            VibrationAnomalyType
        )
        assert TrackStructuralExpert is not None
        assert TrackStructuralAnalysisResult is not None
    
    def test_expert_initialization(self):
        """Test expert initialization without model files."""
        from experts.track_structural_expert import TrackStructuralExpert
        
        expert = TrackStructuralExpert()
        assert expert is not None
        assert expert.WINDOW_SIZE == 100
        assert expert.STEP_SIZE == 50
    
    def test_analyze_vibration_csv(self, sample_csv_vibration):
        """Test vibration CSV analysis."""
        from experts.track_structural_expert import TrackStructuralExpert
        
        expert = TrackStructuralExpert()
        result = expert.analyze_csv(sample_csv_vibration)
        
        assert result is not None
        assert result.file_path == sample_csv_vibration
        assert result.analysis_status in ["success", "partial", "error"]
        assert result.sample_count > 0
        assert result.data_type == "vibration"
        assert result.risk_level in ["low", "medium", "high", "critical"]
    
    def test_analyze_geometric_csv(self, sample_csv_geometric):
        """Test geometric sensor CSV analysis."""
        from experts.track_structural_expert import TrackStructuralExpert
        
        expert = TrackStructuralExpert()
        result = expert.analyze_csv(sample_csv_geometric)
        
        assert result is not None
        assert result.analysis_status in ["success", "partial", "error"]
    
    def test_feature_extraction(self, sample_csv_vibration):
        """Test feature extraction from vibration data."""
        from experts.track_structural_expert import TrackStructuralExpert
        
        expert = TrackStructuralExpert()
        
        # Load data manually to test features
        df = pd.read_csv(sample_csv_vibration)
        features = expert.extract_features(df['x'].values[:100])
        
        assert features is not None
        assert len(features) == 12  # 12 features per axis
    
    def test_to_dict_conversion(self, sample_csv_vibration):
        """Test result to dictionary conversion."""
        from experts.track_structural_expert import TrackStructuralExpert
        
        expert = TrackStructuralExpert()
        result = expert.analyze_csv(sample_csv_vibration)
        result_dict = expert.to_dict(result)
        
        assert isinstance(result_dict, dict)
        assert "file_path" in result_dict
        assert "risk_level" in result_dict
        assert "alerts" in result_dict
    
    def test_invalid_csv_path(self):
        """Test handling of invalid file path."""
        from experts.track_structural_expert import TrackStructuralExpert
        
        expert = TrackStructuralExpert()
        result = expert.analyze_csv("/nonexistent/path/file.csv")
        
        assert result.analysis_status == "error"
    
    def test_empty_csv(self, temp_dir):
        """Test handling of empty CSV file."""
        from experts.track_structural_expert import TrackStructuralExpert
        
        filepath = os.path.join(temp_dir, "empty.csv")
        with open(filepath, 'w') as f:
            f.write("")
        
        expert = TrackStructuralExpert()
        result = expert.analyze_csv(filepath)
        
        assert result.analysis_status == "error"


# ==================== Visual Integrity Expert Tests ====================

class TestVisualIntegrityExpert:
    """Tests for Visual Integrity Expert."""
    
    def test_import(self):
        """Test that module can be imported."""
        from experts.visual_integrity_expert import (
            VisualIntegrityExpert,
            VisualIntegrityResult,
            Detection,
            DetectionType
        )
        assert VisualIntegrityExpert is not None
        assert DetectionType.OBSTACLE is not None
    
    def test_expert_initialization(self):
        """Test expert initialization."""
        from experts.visual_integrity_expert import VisualIntegrityExpert
        
        expert = VisualIntegrityExpert(load_models=False)
        assert expert is not None
    
    @pytest.mark.skipif(
        not pytest.importorskip("PIL", reason="PIL required"),
        reason="PIL not available"
    )
    def test_analyze_image(self, sample_image):
        """Test image analysis (without models)."""
        from experts.visual_integrity_expert import VisualIntegrityExpert
        
        expert = VisualIntegrityExpert(load_models=False)
        result = expert.analyze_image(sample_image)
        
        assert result is not None
        assert result.file_path == sample_image
        assert result.analysis_status in ["success", "partial", "error"]
    
    def test_invalid_image_path(self):
        """Test handling of invalid image path."""
        from experts.visual_integrity_expert import VisualIntegrityExpert
        
        expert = VisualIntegrityExpert(load_models=False)
        result = expert.analyze_image("/nonexistent/image.jpg")
        
        assert result.analysis_status == "error"
    
    def test_to_dict_conversion(self, sample_image):
        """Test result to dictionary conversion."""
        from experts.visual_integrity_expert import VisualIntegrityExpert
        
        expert = VisualIntegrityExpert(load_models=False)
        result = expert.analyze_image(sample_image)
        result_dict = expert.to_dict(result)
        
        assert isinstance(result_dict, dict)
        assert "file_path" in result_dict
    
    def test_detection_type_enum(self):
        """Test detection type enumeration."""
        from experts.visual_integrity_expert import DetectionType
        
        assert DetectionType.OBSTACLE.value == "obstacle"
        assert DetectionType.TAMPERING.value == "tampering"
        assert DetectionType.CRACK.value == "crack"


# ==================== Thermal Anomaly Expert Tests ====================

class TestThermalAnomalyExpert:
    """Tests for Thermal Anomaly Expert."""
    
    def test_import(self):
        """Test that module can be imported."""
        from experts.thermal_anomaly_expert import (
            ThermalAnomalyExpert,
            ThermalAnomalyResult,
            AnomalyType
        )
        assert ThermalAnomalyExpert is not None
        assert AnomalyType.DEBRIS is not None
    
    def test_expert_initialization(self):
        """Test expert initialization."""
        from experts.thermal_anomaly_expert import ThermalAnomalyExpert
        
        expert = ThermalAnomalyExpert()
        assert expert is not None
    
    def test_analyze_npy_pointcloud(self, sample_lidar_npy):
        """Test NPY point cloud analysis."""
        from experts.thermal_anomaly_expert import ThermalAnomalyExpert
        
        expert = ThermalAnomalyExpert()
        result = expert.analyze_lidar(sample_lidar_npy)
        
        assert result is not None
        assert result.file_path == sample_lidar_npy
        assert result.analysis_status in ["success", "partial", "error"]
    
    def test_create_sample_data(self, temp_dir):
        """Test sample data creation."""
        from experts.thermal_anomaly_expert import ThermalAnomalyExpert
        
        expert = ThermalAnomalyExpert()
        sample_path = expert.create_sample_data(temp_dir)
        
        assert sample_path is not None
        assert os.path.exists(sample_path)
        
        # Load and verify
        data = np.load(sample_path)
        assert data.shape[1] == 3  # x, y, z
    
    def test_invalid_file_path(self):
        """Test handling of invalid file path."""
        from experts.thermal_anomaly_expert import ThermalAnomalyExpert
        
        expert = ThermalAnomalyExpert()
        result = expert.analyze_lidar("/nonexistent/file.npy")
        
        assert result.analysis_status == "error"
    
    def test_anomaly_type_enum(self):
        """Test anomaly type enumeration."""
        from experts.thermal_anomaly_expert import AnomalyType
        
        assert AnomalyType.DEBRIS.value == "debris"
        assert AnomalyType.GAUGE_DEVIATION.value == "gauge_deviation"
        assert AnomalyType.OBSTRUCTION_3D.value == "obstruction_3d"


# ==================== Contextual Reasoning Expert Tests ====================

class TestContextualReasoningExpert:
    """Tests for Contextual Reasoning Expert."""
    
    def test_import(self):
        """Test that module can be imported."""
        from experts.contextual_reasoning_expert import (
            QueryType,
            ReasoningResult,
            SYSTEM_PROMPT
        )
        assert QueryType.RISK_ASSESSMENT is not None
        assert len(SYSTEM_PROMPT) > 0
    
    def test_query_classification(self):
        """Test query type classification."""
        from experts.contextual_reasoning_expert import _classify_query, QueryType
        
        assert _classify_query("What is the current risk?") == QueryType.RISK_ASSESSMENT
        assert _classify_query("What should we do?") == QueryType.RECOMMENDATION
        assert _classify_query("What is the status?") == QueryType.STATUS_INQUIRY
        assert _classify_query("Why did this happen?") == QueryType.ANOMALY_EXPLANATION
        assert _classify_query("What is the weather?") == QueryType.GENERAL
    
    def test_context_string_builder(self):
        """Test context string building."""
        from experts.contextual_reasoning_expert import _build_context_string
        
        context = {
            "track_structural": {
                "status": "success",
                "confidence": 0.85,
                "alerts": ["High vibration detected"],
                "output": {"risk_level": "medium"}
            }
        }
        
        result = _build_context_string(context)
        
        assert "Track Structural Analysis" in result
        assert "85" in result  # confidence
        assert "High vibration" in result
    
    def test_reasoning_result_dataclass(self):
        """Test ReasoningResult dataclass."""
        from experts.contextual_reasoning_expert import ReasoningResult, QueryType
        
        result = ReasoningResult(
            query="What is the risk?",
            query_type=QueryType.RISK_ASSESSMENT,
            response="Low risk detected",
            confidence=0.9
        )
        
        assert result.query == "What is the risk?"
        assert result.query_type == QueryType.RISK_ASSESSMENT
        assert result.confidence == 0.9


# ==================== Orchestration Tests ====================

class TestOrchestration:
    """Tests for Orchestration module."""
    
    def test_import(self):
        """Test that module can be imported."""
        from experts.orchestration import (
            InputType,
            InputData,
            ExpertResult,
            OrchestratorState
        )
        assert InputType.CSV is not None
        assert InputData is not None
    
    def test_input_type_enum(self):
        """Test input type enumeration."""
        from experts.orchestration import InputType
        
        assert InputType.CSV.value == "csv"
        assert InputType.IMAGE.value == "image"
        assert InputType.LAZ.value == "laz"
        assert InputType.TEXT.value == "text"
    
    def test_expert_result_dataclass(self):
        """Test ExpertResult dataclass."""
        from experts.orchestration import ExpertResult
        
        result = ExpertResult(
            expert_name="test_expert",
            status="success",
            output={"key": "value"},
            confidence=0.95,
            alerts=["Test alert"]
        )
        
        assert result.expert_name == "test_expert"
        assert result.status == "success"
        assert result.confidence == 0.95
    
    def test_input_data_dataclass(self):
        """Test InputData dataclass."""
        from experts.orchestration import InputData, InputType
        
        data = InputData(
            input_type=InputType.CSV,
            file_path="/path/to/file.csv",
            metadata={"source": "test"}
        )
        
        assert data.input_type == InputType.CSV
        assert data.file_path == "/path/to/file.csv"


# ==================== Combined Inference Tests ====================

class TestCombinedInference:
    """Tests for Combined Inference module."""
    
    def test_import(self):
        """Test that module can be imported."""
        from experts.combined_inference import (
            AlertSeverity,
            RealTimeAlert,
            CombinedAnalysisResult
        )
        assert AlertSeverity.CRITICAL is not None
        assert RealTimeAlert is not None
    
    def test_alert_severity_enum(self):
        """Test alert severity enumeration."""
        from experts.combined_inference import AlertSeverity
        
        assert AlertSeverity.INFO.value == "info"
        assert AlertSeverity.WARNING.value == "warning"
        assert AlertSeverity.HIGH.value == "high"
        assert AlertSeverity.CRITICAL.value == "critical"
    
    def test_real_time_alert_dataclass(self):
        """Test RealTimeAlert dataclass."""
        from experts.combined_inference import RealTimeAlert, AlertSeverity
        
        alert = RealTimeAlert(
            alert_id="ALT-001",
            timestamp=datetime.utcnow().isoformat(),
            severity=AlertSeverity.HIGH,
            source_expert="visual",
            title="Obstacle Detected",
            message="Large object on track",
            action_required=True
        )
        
        assert alert.alert_id == "ALT-001"
        assert alert.severity == AlertSeverity.HIGH
        assert alert.action_required == True
    
    def test_combined_analysis_result(self):
        """Test CombinedAnalysisResult dataclass."""
        from experts.combined_inference import CombinedAnalysisResult
        
        result = CombinedAnalysisResult(
            timestamp=datetime.utcnow().isoformat(),
            session_id="TEST-001",
            overall_risk_level="medium",
            tampering_detected=True,
            confidence=0.85
        )
        
        assert result.session_id == "TEST-001"
        assert result.tampering_detected == True
        assert result.overall_risk_level == "medium"


# ==================== Integration Tests ====================

class TestIntegration:
    """Integration tests across multiple experts."""
    
    def test_structural_to_dict_pipeline(self, sample_csv_vibration):
        """Test full pipeline from CSV to dict output."""
        from experts.track_structural_expert import TrackStructuralExpert
        
        expert = TrackStructuralExpert()
        result = expert.analyze_csv(sample_csv_vibration)
        result_dict = expert.to_dict(result)
        
        # Verify JSON serializable
        json_str = json.dumps(result_dict)
        assert len(json_str) > 0
        
        # Verify can be parsed back
        parsed = json.loads(json_str)
        assert parsed["file_path"] == sample_csv_vibration
    
    def test_all_experts_importable(self):
        """Test that all expert modules can be imported together."""
        from experts.visual_integrity_expert import VisualIntegrityExpert
        from experts.thermal_anomaly_expert import ThermalAnomalyExpert
        from experts.track_structural_expert import TrackStructuralExpert
        from experts.contextual_reasoning_expert import ReasoningResult
        from experts.orchestration import InputType, ExpertResult
        
        # All imports successful
        assert VisualIntegrityExpert is not None
        assert ThermalAnomalyExpert is not None
        assert TrackStructuralExpert is not None
    
    def test_expert_result_consistency(self, sample_csv_vibration, sample_image, sample_lidar_npy):
        """Test that all experts return consistent result structures."""
        from experts.track_structural_expert import TrackStructuralExpert
        from experts.visual_integrity_expert import VisualIntegrityExpert
        from experts.thermal_anomaly_expert import ThermalAnomalyExpert
        
        # Initialize experts
        structural = TrackStructuralExpert()
        visual = VisualIntegrityExpert(load_models=False)
        thermal = ThermalAnomalyExpert()
        
        # Analyze
        struct_result = structural.analyze_csv(sample_csv_vibration)
        visual_result = visual.analyze_image(sample_image)
        thermal_result = thermal.analyze_lidar(sample_lidar_npy)
        
        # Convert to dict
        struct_dict = structural.to_dict(struct_result)
        visual_dict = visual.to_dict(visual_result)
        thermal_dict = thermal.to_dict(thermal_result)
        
        # Check common fields
        for result_dict in [struct_dict, visual_dict, thermal_dict]:
            assert "file_path" in result_dict
            assert "analysis_status" in result_dict


# ==================== Edge Case Tests ====================

class TestEdgeCases:
    """Edge case and error handling tests."""
    
    def test_empty_data_handling(self, temp_dir):
        """Test handling of empty data files."""
        from experts.track_structural_expert import TrackStructuralExpert
        
        # Create empty CSV
        filepath = os.path.join(temp_dir, "empty.csv")
        with open(filepath, 'w') as f:
            f.write("x,y,z\n")  # Headers only
        
        expert = TrackStructuralExpert()
        result = expert.analyze_csv(filepath)
        
        assert result is not None
        # Should handle gracefully without crashing
    
    def test_malformed_csv(self, temp_dir):
        """Test handling of malformed CSV."""
        from experts.track_structural_expert import TrackStructuralExpert
        
        filepath = os.path.join(temp_dir, "malformed.csv")
        with open(filepath, 'w') as f:
            f.write("this,is,not,proper,data\n")
            f.write("1,2,3,4,5\n")
            f.write("malformed row here\n")
        
        expert = TrackStructuralExpert()
        result = expert.analyze_csv(filepath)
        
        # Should not crash
        assert result is not None
    
    def test_large_point_cloud(self, temp_dir):
        """Test handling of larger point cloud."""
        from experts.thermal_anomaly_expert import ThermalAnomalyExpert
        
        # Create larger point cloud
        np.random.seed(42)
        points = np.random.randn(10000, 3)
        
        filepath = os.path.join(temp_dir, "large_cloud.npy")
        np.save(filepath, points)
        
        expert = ThermalAnomalyExpert()
        result = expert.analyze_lidar(filepath)
        
        assert result is not None
        assert result.point_count > 0


# ==================== Main Entry ====================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
