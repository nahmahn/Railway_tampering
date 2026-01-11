"""
Expert Module Integration Tests
Tests the actual expert classes with mocked heavy dependencies.

Run with: python test_expert_modules.py
"""

import os
import sys
import json
import tempfile
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Mock tensorflow BEFORE any imports that might use it
class MockTF:
    """Mock tensorflow module."""
    def __getattr__(self, name):
        return MockTF()
    def __call__(self, *args, **kwargs):
        return MockTF()

sys.modules['tensorflow'] = MockTF()
sys.modules['tensorflow.keras'] = MockTF()
sys.modules['tensorflow.keras.models'] = MockTF()

# Mock torch and transformers for visual expert
class MockTorch:
    """Mock torch module."""
    class _cuda:
        @staticmethod
        def is_available():
            return False
    cuda = _cuda()
    float32 = 'float32'
    
    @staticmethod
    def no_grad():
        class Ctx:
            def __enter__(self):
                return None
            def __exit__(self, *args):
                return None
        return Ctx()
    
    @staticmethod
    def tensor(*args, **kwargs):
        return np.array(args[0]) if args else np.array([])
    
    def __getattr__(self, name):
        return MockTorch()
    def __call__(self, *args, **kwargs):
        return MockTorch()

sys.modules['torch'] = MockTorch()
sys.modules['transformers'] = MockTorch()
sys.modules['segment_anything'] = MockTorch()
sys.modules['ultralytics'] = MockTorch()

# Now import expert modules
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))


class TestResults:
    """Track test results."""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.errors = []
    
    def add_pass(self, name):
        self.passed += 1
        print(f"  ✅ PASS: {name}")
    
    def add_fail(self, name, error):
        self.failed += 1
        self.errors.append((name, str(error)))
        print(f"  ❌ FAIL: {name}")
        print(f"      Error: {str(error)[:200]}")
    
    def add_skip(self, name, reason):
        self.skipped += 1
        print(f"  ⚠️ SKIP: {name} ({reason})")
    
    def summary(self):
        total = self.passed + self.failed + self.skipped
        print(f"\n{'='*60}")
        print(f"RESULTS: {self.passed}/{total} passed, {self.failed} failed, {self.skipped} skipped")
        if self.errors:
            print(f"\nFailed tests:")
            for name, error in self.errors:
                print(f"  - {name}: {error[:100]}...")
        return self.failed == 0


def create_test_files(temp_dir):
    """Create all test files."""
    files = {}
    
    # Vibration CSV
    np.random.seed(42)
    df = pd.DataFrame({
        'x': np.random.randn(500) * 0.1,
        'y': np.random.randn(500) * 0.1,
        'z': np.random.randn(500) * 0.1 + 9.8,
    })
    files['csv'] = os.path.join(temp_dir, "vibration.csv")
    df.to_csv(files['csv'], index=False)
    
    # LiDAR NPY
    points = np.column_stack([
        np.random.uniform(-10, 10, 1000),
        np.random.uniform(0, 100, 1000),
        np.random.uniform(0, 2, 1000)
    ])
    files['npy'] = os.path.join(temp_dir, "pointcloud.npy")
    np.save(files['npy'], points)
    
    # Image (if PIL available)
    try:
        from PIL import Image
        img = Image.new('RGB', (640, 480), color=(128, 128, 128))
        files['image'] = os.path.join(temp_dir, "test.jpg")
        img.save(files['image'])
    except:
        files['image'] = None
    
    return files


def run_tests():
    results = TestResults()
    temp_dir = tempfile.mkdtemp()
    files = create_test_files(temp_dir)
    
    print("="*60)
    print("Railway Tampering Detection - Expert Module Tests")
    print("="*60)
    
    # ==================== Track Structural Expert ====================
    print("\n🔧 Track Structural Expert Tests")
    print("-"*40)
    
    try:
        from experts.track_structural_expert import (
            TrackStructuralExpert,
            TrackStructuralAnalysisResult,
            VibrationAnomalyType,
            VibrationWindow
        )
        results.add_pass("Import track_structural_expert module")
    except Exception as e:
        results.add_fail("Import track_structural_expert", e)
        # Skip remaining tests for this module
        print("  ⚠️ Skipping remaining structural expert tests due to import failure")
    else:
        try:
            expert = TrackStructuralExpert()
            assert expert.WINDOW_SIZE == 100
            assert expert.STEP_SIZE == 50
            results.add_pass("TrackStructuralExpert initialization")
        except Exception as e:
            results.add_fail("TrackStructuralExpert initialization", e)
        
        try:
            result = expert.analyze_csv(files['csv'])
            assert isinstance(result, TrackStructuralAnalysisResult)
            assert result.file_path == files['csv']
            assert result.analysis_status in ["success", "partial", "error"]
            results.add_pass(f"analyze_csv (status: {result.analysis_status})")
        except Exception as e:
            results.add_fail("analyze_csv", e)
        
        try:
            result_dict = expert.to_dict(result)
            assert isinstance(result_dict, dict)
            assert "file_path" in result_dict
            assert "risk_level" in result_dict or "result" in result_dict
            json.dumps(result_dict)  # Test JSON serializable
            results.add_pass("to_dict conversion")
        except Exception as e:
            results.add_fail("to_dict conversion", e)
        
        try:
            df = pd.read_csv(files['csv'])
            # extract_features expects 2D array with shape (n, 3) for x, y, z
            window_data = df[['x', 'y', 'z']].values[:100]
            features = expert.extract_features(window_data)
            assert len(features) == 36  # 12 features * 3 axes
            results.add_pass("extract_features")
        except Exception as e:
            results.add_fail("extract_features", e)
        
        try:
            # Test enum
            assert VibrationAnomalyType.NORMAL.value == "normal"
            assert VibrationAnomalyType.FAILURE.value == "failure"
            results.add_pass("VibrationAnomalyType enum")
        except Exception as e:
            results.add_fail("VibrationAnomalyType enum", e)
    
    # ==================== Thermal Anomaly Expert ====================
    print("\n🌡️ Thermal Anomaly Expert Tests")
    print("-"*40)
    
    try:
        from experts.thermal_anomaly_expert import (
            ThermalAnomalyExpert,
            ThermalAnomalyResult,
            AnomalyType,
            ThermalAnomaly
        )
        results.add_pass("Import thermal_anomaly_expert module")
    except Exception as e:
        results.add_fail("Import thermal_anomaly_expert", e)
        print("  ⚠️ Skipping remaining thermal expert tests due to import failure")
    else:
        try:
            expert = ThermalAnomalyExpert()
            results.add_pass("ThermalAnomalyExpert initialization")
        except Exception as e:
            results.add_fail("ThermalAnomalyExpert initialization", e)
        
        try:
            result = expert.analyze_lidar(files['npy'])
            assert isinstance(result, ThermalAnomalyResult)
            assert result.file_path == files['npy']
            results.add_pass(f"analyze_lidar (status: {result.analysis_status})")
        except Exception as e:
            results.add_fail("analyze_lidar", e)
        
        try:
            result_dict = expert.to_dict(result)
            assert isinstance(result_dict, dict)
            json.dumps(result_dict)
            results.add_pass("to_dict conversion")
        except Exception as e:
            results.add_fail("to_dict conversion", e)
        
        try:
            sample_path = expert.create_sample_data(temp_dir)
            assert os.path.exists(sample_path)
            data = np.load(sample_path)
            assert data.shape[1] == 3
            results.add_pass("create_sample_data")
        except Exception as e:
            results.add_fail("create_sample_data", e)
        
        try:
            assert AnomalyType.DEBRIS.value == "debris"
            assert AnomalyType.GAUGE_DEVIATION.value == "gauge_deviation"
            results.add_pass("AnomalyType enum")
        except Exception as e:
            results.add_fail("AnomalyType enum", e)
    
    # ==================== Visual Integrity Expert ====================
    print("\n👁️ Visual Integrity Expert Tests")
    print("-"*40)
    
    try:
        from experts.visual_integrity_expert import (
            VisualIntegrityExpert,
            VisualIntegrityResult,
            Detection,
            DetectionType
        )
        results.add_pass("Import visual_integrity_expert module")
    except Exception as e:
        results.add_fail("Import visual_integrity_expert", e)
        print("  ⚠️ Skipping remaining visual expert tests due to import failure")
    else:
        try:
            # Initialize without load_models parameter (it's not supported)
            expert = VisualIntegrityExpert()
            results.add_pass("VisualIntegrityExpert initialization")
        except Exception as e:
            results.add_fail("VisualIntegrityExpert initialization", e)
            expert = None
        
        if files['image'] and expert:
            try:
                result = expert.analyze_image(files['image'])
                assert isinstance(result, VisualIntegrityResult)
                results.add_pass(f"analyze_image (status: {result.analysis_status})")
            except Exception as e:
                results.add_fail("analyze_image", e)
            
            try:
                result_dict = expert.to_dict(result)
                assert isinstance(result_dict, dict)
                json.dumps(result_dict)
                results.add_pass("to_dict conversion (visual)")
            except Exception as e:
                results.add_fail("to_dict conversion (visual)", e)
        else:
            results.add_skip("analyze_image", "PIL not available or expert init failed")
            results.add_skip("to_dict conversion (visual)", "PIL not available or expert init failed")
        
        try:
            assert DetectionType.OBSTACLE.value == "obstacle"
            assert DetectionType.TAMPERING.value == "tampering"
            assert DetectionType.CRACK.value == "crack"
            results.add_pass("DetectionType enum")
        except Exception as e:
            results.add_fail("DetectionType enum", e)
    
    # ==================== Contextual Reasoning Expert ====================
    print("\n🧠 Contextual Reasoning Expert Tests")
    print("-"*40)
    
    try:
        from experts.contextual_reasoning_expert import (
            QueryType,
            ReasoningResult,
            SYSTEM_PROMPT,
            _classify_query,
            _build_context_string
        )
        results.add_pass("Import contextual_reasoning_expert module")
    except Exception as e:
        results.add_fail("Import contextual_reasoning_expert", e)
        print("  ⚠️ Skipping remaining reasoning expert tests due to import failure")
    else:
        try:
            assert len(SYSTEM_PROMPT) > 100
            results.add_pass("SYSTEM_PROMPT defined")
        except Exception as e:
            results.add_fail("SYSTEM_PROMPT defined", e)
        
        try:
            assert _classify_query("What is the risk?") == QueryType.RISK_ASSESSMENT
            assert _classify_query("What should we do?") == QueryType.RECOMMENDATION
            assert _classify_query("Current status?") == QueryType.STATUS_INQUIRY
            results.add_pass("_classify_query function")
        except Exception as e:
            results.add_fail("_classify_query function", e)
        
        try:
            context = {"track_structural": {"status": "success", "confidence": 0.85, "alerts": [], "output": {}}}
            context_str = _build_context_string(context)
            assert "Track Structural" in context_str
            results.add_pass("_build_context_string function")
        except Exception as e:
            results.add_fail("_build_context_string function", e)
        
        try:
            result = ReasoningResult(
                query="Test",
                query_type=QueryType.GENERAL,
                response="Test response",
                confidence=0.9
            )
            assert result.query == "Test"
            assert result.confidence == 0.9
            results.add_pass("ReasoningResult dataclass")
        except Exception as e:
            results.add_fail("ReasoningResult dataclass", e)
    
    # ==================== Orchestration ====================
    print("\n🎭 Orchestration Tests")
    print("-"*40)
    
    try:
        from experts.orchestration import (
            InputType,
            InputData,
            ExpertResult,
            OrchestratorState
        )
        results.add_pass("Import orchestration module")
    except Exception as e:
        results.add_fail("Import orchestration", e)
        print("  ⚠️ Skipping remaining orchestration tests due to import failure")
    else:
        try:
            assert InputType.CSV.value == "csv"
            assert InputType.IMAGE.value == "image"
            assert InputType.LAZ.value == "laz"
            results.add_pass("InputType enum")
        except Exception as e:
            results.add_fail("InputType enum", e)
        
        try:
            data = InputData(
                input_type=InputType.CSV,
                file_path="/test/path.csv",
                metadata={"source": "test"}
            )
            assert data.file_path == "/test/path.csv"
            results.add_pass("InputData dataclass")
        except Exception as e:
            results.add_fail("InputData dataclass", e)
        
        try:
            result = ExpertResult(
                expert_name="test",
                status="success",
                output={"key": "value"},
                confidence=0.95
            )
            assert result.expert_name == "test"
            assert result.confidence == 0.95
            results.add_pass("ExpertResult dataclass")
        except Exception as e:
            results.add_fail("ExpertResult dataclass", e)
    
    # ==================== Combined Inference ====================
    print("\n🔗 Combined Inference Tests")
    print("-"*40)
    
    try:
        from experts.combined_inference import (
            AlertSeverity,
            RealTimeAlert,
            CombinedAnalysisResult
        )
        results.add_pass("Import combined_inference module")
    except Exception as e:
        results.add_fail("Import combined_inference", e)
        print("  ⚠️ Skipping remaining combined inference tests due to import failure")
    else:
        try:
            assert AlertSeverity.INFO.value == "info"
            assert AlertSeverity.CRITICAL.value == "critical"
            results.add_pass("AlertSeverity enum")
        except Exception as e:
            results.add_fail("AlertSeverity enum", e)
        
        try:
            alert = RealTimeAlert(
                alert_id="TEST-001",
                timestamp=datetime.utcnow().isoformat(),
                severity=AlertSeverity.HIGH,
                source_expert="visual",
                title="Test Alert",
                message="Test message"
            )
            assert alert.severity == AlertSeverity.HIGH
            results.add_pass("RealTimeAlert dataclass")
        except Exception as e:
            results.add_fail("RealTimeAlert dataclass", e)
        
        try:
            result = CombinedAnalysisResult(
                timestamp=datetime.utcnow().isoformat(),
                session_id="TEST-001",
                overall_risk_level="medium",
                tampering_detected=True
            )
            assert result.tampering_detected == True
            results.add_pass("CombinedAnalysisResult dataclass")
        except Exception as e:
            results.add_fail("CombinedAnalysisResult dataclass", e)
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)
    
    return results.summary()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
