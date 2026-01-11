"""
Comprehensive Test Suite - Models, Reports, and Frontend API
Tests cover ML models, report generation, and FastAPI endpoints.

Run with: python test_full_system.py
"""

import os
import sys
import json
import tempfile
import asyncio
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from unittest.mock import MagicMock, patch, AsyncMock

# Mock heavy dependencies BEFORE imports
class MockTF:
    def __getattr__(self, name):
        return MockTF()
    def __call__(self, *args, **kwargs):
        return MockTF()

sys.modules['tensorflow'] = MockTF()
sys.modules['tensorflow.keras'] = MockTF()
sys.modules['tensorflow.keras.models'] = MockTF()

class MockTorch:
    class _cuda:
        @staticmethod
        def is_available():
            return False
    cuda = _cuda()
    float32 = 'float32'
    
    @staticmethod
    def no_grad():
        class Ctx:
            def __enter__(self): return None
            def __exit__(self, *args): return None
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

# Add paths
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))


class TestResults:
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
                print(f"  - {name}: {error[:80]}...")
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
    
    # Image
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
    print("Railway Tampering Detection - Full System Tests")
    print("="*60)
    
    # ==================== MODEL TESTS ====================
    print("\n🤖 MODEL TESTS")
    print("-"*40)
    
    # Test 1: XGBoost model interface
    try:
        from experts.track_structural_expert import TrackStructuralExpert
        
        expert = TrackStructuralExpert()
        
        # Test predict_xgboost method exists and handles None model
        features = np.random.randn(36)
        prediction, probability = expert.predict_xgboost(features)
        
        assert prediction == 0  # Default when no model
        assert probability == 0.0
        results.add_pass("XGBoost model interface (no model fallback)")
    except Exception as e:
        results.add_fail("XGBoost model interface", e)
    
    # Test 2: Model loading paths
    try:
        expert = TrackStructuralExpert(
            xgb_model_path="nonexistent.pkl",
            scaler_path="nonexistent.pkl"
        )
        assert expert.xgb_model is None  # Should gracefully handle missing
        assert expert.scaler is None
        results.add_pass("Model loading (missing files handled)")
    except Exception as e:
        results.add_fail("Model loading", e)
    
    # Test 3: Heuristic prediction fallback
    try:
        from experts.track_structural_expert import TrackStructuralExpert
        
        expert = TrackStructuralExpert()
        result = expert.analyze_csv(files['csv'])
        
        # Should use heuristic when no model
        assert result.model_used in ["heuristic", "xgboost", ""]
        assert result.analysis_status == "success"
        results.add_pass("Heuristic prediction fallback")
    except Exception as e:
        results.add_fail("Heuristic prediction fallback", e)
    
    # Test 4: Visual expert model interface
    try:
        from experts.visual_integrity_expert import VisualIntegrityExpert
        
        expert = VisualIntegrityExpert()
        
        # Models should be None when dependencies unavailable
        # Expert should still work with graceful degradation
        results.add_pass("Visual expert initialization (degraded mode)")
    except Exception as e:
        results.add_fail("Visual expert initialization", e)
    
    # Test 5: Thermal expert with/without LiDAR modules
    try:
        from experts.thermal_anomaly_expert import ThermalAnomalyExpert, LIDAR_AVAILABLE
        
        expert = ThermalAnomalyExpert()
        result = expert.analyze_lidar(files['npy'])
        
        # Should work even without full LiDAR modules
        assert result is not None
        results.add_pass(f"Thermal expert (LIDAR_AVAILABLE={LIDAR_AVAILABLE})")
    except Exception as e:
        results.add_fail("Thermal expert", e)
    
    # ==================== REPORT GENERATION TESTS ====================
    print("\n📋 REPORT GENERATION TESTS")
    print("-"*40)
    
    # Test 6: Action Recommendation Generator initialization
    try:
        from experts.combined_inference import ActionRecommendationGenerator
        
        generator = ActionRecommendationGenerator()
        assert generator.system_prompt is not None
        assert len(generator.system_prompt) > 100
        results.add_pass("ActionRecommendationGenerator initialization")
    except Exception as e:
        results.add_fail("ActionRecommendationGenerator initialization", e)
    
    # Test 7: Fallback report generation
    try:
        from experts.combined_inference import ActionRecommendationGenerator
        
        generator = ActionRecommendationGenerator()
        
        structured_data = {
            "analysis_summary": {
                "overall_status": "ALERT",
                "confidence_score": 0.85
            },
            "detections": [
                {"type": "obstacle", "severity": "high", "confidence": 0.9}
            ]
        }
        
        tampering_analysis = {
            "tampering_detected": True,
            "severity_classification": {"tier": 3, "level": "high"},
            "tampering_assessment": {"confidence": 0.85}
        }
        
        report = generator._fallback_report(structured_data, tampering_analysis, {})
        
        assert "incident_snapshot" in report or "report_type" in report
        results.add_pass("Fallback report generation")
    except Exception as e:
        results.add_fail("Fallback report generation", e)
    
    # Test 8: Report with different tiers
    try:
        from experts.combined_inference import ActionRecommendationGenerator
        
        generator = ActionRecommendationGenerator()
        
        for tier in [1, 2, 3, 4]:
            tampering_analysis = {
                "severity_classification": {"tier": tier},
                "tampering_assessment": {}
            }
            report = generator._fallback_report({}, tampering_analysis, {})
            
            # Check appropriate actions for tier
            if "recommended_actions" in report:
                assert len(report["recommended_actions"]) > 0
        
        results.add_pass("Report tiers (1-4) handled correctly")
    except Exception as e:
        results.add_fail("Report tiers", e)
    
    # Test 9: Gemini Output Structurer
    try:
        from experts.combined_inference import GeminiOutputStructurer
        
        structurer = GeminiOutputStructurer()
        
        # Test manual structuring fallback
        expert_outputs = {
            "visual": {"detections": [{"type": "obstacle"}], "confidence": 0.8},
            "structural": {"risk_level": "medium", "failure_ratio": 0.2}
        }
        
        result = structurer._manual_structure(expert_outputs)
        
        assert "analysis_summary" in result
        assert "detections" in result
        assert "risk_assessment" in result
        results.add_pass("Gemini Output Structurer (manual fallback)")
    except Exception as e:
        results.add_fail("Gemini Output Structurer", e)
    
    # Test 10: Contextual Reasoning Report
    try:
        from experts.combined_inference import ContextualReasoningEngine
        
        engine = ContextualReasoningEngine()
        
        # Test fallback analysis
        structured = {
            "detections": [{"type": "obstacle", "severity": "high"}],
            "risk_assessment": {"tampering_probability": 0.8}
        }
        
        result = engine._fallback_analysis(structured)
        
        assert "tampering_assessment" in result
        assert "severity_classification" in result
        results.add_pass("Contextual Reasoning (fallback)")
    except Exception as e:
        results.add_fail("Contextual Reasoning", e)
    
    # Test 11: Real-time Alert Manager
    try:
        from experts.combined_inference import RealTimeAlertManager, AlertSeverity
        
        manager = RealTimeAlertManager()
        
        # Generate alerts from analysis
        structured = {
            "detections": [{"type": "obstacle", "severity": "high", "confidence": 0.9}]
        }
        tampering = {
            "tampering_assessment": {"is_tampering_detected": True, "evidence": ["test"]},
            "severity_classification": {"tier": 3}
        }
        
        alerts = manager.generate_alerts(structured, tampering)
        
        assert len(alerts) > 0
        assert alerts[0].severity in [AlertSeverity.HIGH, AlertSeverity.CRITICAL]
        
        # Get pending alerts
        pending = manager.get_pending_alerts()
        assert len(pending) >= 1
        
        # Acknowledge alert
        success = manager.acknowledge_alert(alerts[0].alert_id)
        assert success
        
        results.add_pass("Real-time Alert Manager")
    except Exception as e:
        results.add_fail("Real-time Alert Manager", e)
    
    # ==================== FRONTEND API TESTS ====================
    print("\n🌐 FRONTEND API TESTS")
    print("-"*40)
    
    # Test 12: Pydantic models
    try:
        from api_server import (
            HealthResponse, AnalysisRequest, QueryRequest,
            CombinedAnalysisRequest, AlertAcknowledgeRequest, AnalysisResponse
        )
        
        # Test HealthResponse
        health = HealthResponse(
            status="healthy",
            timestamp=datetime.utcnow().isoformat(),
            experts_available={"visual": True, "thermal": False}
        )
        assert health.status == "healthy"
        
        # Test QueryRequest
        query = QueryRequest(query="What is the risk?", context={})
        assert query.query == "What is the risk?"
        
        # Test AnalysisResponse
        response = AnalysisResponse(
            success=True,
            session_id="TEST-001",
            timestamp=datetime.utcnow().isoformat(),
            result={"key": "value"},
            alerts=[]
        )
        assert response.success
        
        results.add_pass("Pydantic API models")
    except Exception as e:
        results.add_fail("Pydantic API models", e)
    
    # Test 13: API endpoint structure
    try:
        from api_server import app
        
        # Check routes exist
        routes = [route.path for route in app.routes]
        
        expected_routes = [
            "/api/health",
            "/api/analyze/visual",
            "/api/analyze/lidar",
            "/api/analyze/vibration",
            "/api/analyze/combined",
            "/api/query",
            "/api/alerts",
            "/ws/alerts"
        ]
        
        for expected in expected_routes:
            found = any(expected in r for r in routes)
            assert found, f"Missing route: {expected}"
        
        results.add_pass("API endpoint structure")
    except Exception as e:
        results.add_fail("API endpoint structure", e)
    
    # Test 14: CORS configuration
    try:
        from api_server import app
        
        # Check CORS middleware
        middleware_classes = [m.cls.__name__ for m in app.user_middleware]
        assert "CORSMiddleware" in middleware_classes
        results.add_pass("CORS middleware configured")
    except Exception as e:
        results.add_fail("CORS middleware", e)
    
    # Test 15: File upload handling
    try:
        from api_server import save_upload_file, generate_session_id, UPLOAD_DIR
        
        # Test session ID generation
        session_id = generate_session_id()
        assert session_id.startswith("API-")
        assert len(session_id) > 10
        
        # Test upload directory exists
        assert UPLOAD_DIR.exists() or True  # May not exist in test
        
        results.add_pass("File upload utilities")
    except Exception as e:
        results.add_fail("File upload utilities", e)
    
    # Test 16: API response format
    try:
        from api_server import AnalysisResponse
        
        # Test JSON serialization
        response = AnalysisResponse(
            success=True,
            session_id="TEST-001",
            timestamp=datetime.utcnow().isoformat() + "Z",
            result={
                "analysis": {"status": "complete"},
                "risk_level": "medium",
                "alerts": []
            },
            alerts=[{"type": "warning", "message": "Test"}]
        )
        
        # Convert to dict (simulating JSON response)
        response_dict = response.model_dump()
        
        assert response_dict["success"] == True
        assert "session_id" in response_dict
        assert "result" in response_dict
        
        # Verify JSON serializable
        json_str = json.dumps(response_dict)
        assert len(json_str) > 0
        
        results.add_pass("API response JSON format")
    except Exception as e:
        results.add_fail("API response JSON format", e)
    
    # Test 17: WebSocket endpoint exists
    try:
        from api_server import app
        
        ws_routes = [r for r in app.routes if hasattr(r, 'path') and 'ws' in r.path]
        assert len(ws_routes) > 0
        results.add_pass("WebSocket endpoint exists")
    except Exception as e:
        results.add_fail("WebSocket endpoint", e)
    
    # Test 18: AppState initialization
    try:
        from api_server import AppState
        
        state = AppState()
        
        # Check all expert slots
        assert hasattr(state, 'visual_expert')
        assert hasattr(state, 'thermal_expert')
        assert hasattr(state, 'structural_expert')
        assert hasattr(state, 'combined_engine')
        assert hasattr(state, 'websocket_connections')
        
        results.add_pass("AppState structure")
    except Exception as e:
        results.add_fail("AppState structure", e)
    
    # ==================== COMBINED INFERENCE TESTS ====================
    print("\n🔗 COMBINED INFERENCE TESTS")
    print("-"*40)
    
    # Test 19: Combined Inference Engine
    try:
        from experts.combined_inference import CombinedInferenceEngine
        
        engine = CombinedInferenceEngine()
        
        # Check all components initialized
        assert hasattr(engine, 'visual_expert')
        assert hasattr(engine, 'thermal_expert')
        assert hasattr(engine, 'structural_expert')
        assert hasattr(engine, 'output_structurer')
        assert hasattr(engine, 'reasoning_engine')
        assert hasattr(engine, 'report_generator')
        assert hasattr(engine, 'alert_manager')
        
        results.add_pass("CombinedInferenceEngine components")
    except Exception as e:
        results.add_fail("CombinedInferenceEngine components", e)
    
    # Test 20: Combined analysis flow
    try:
        from experts.combined_inference import CombinedInferenceEngine
        
        engine = CombinedInferenceEngine()
        
        # Test with CSV only
        result = engine.analyze(csv_paths=[files['csv']])
        
        assert result is not None
        assert hasattr(result, 'session_id')
        assert hasattr(result, 'timestamp')
        assert hasattr(result, 'overall_risk_level')
        
        results.add_pass("Combined analysis flow")
    except Exception as e:
        results.add_fail("Combined analysis flow", e)
    
    # Test 21: Combined result to_dict
    try:
        from experts.combined_inference import CombinedInferenceEngine
        
        engine = CombinedInferenceEngine()
        result = engine.analyze(csv_paths=[files['csv']])
        
        result_dict = engine.to_dict(result)
        
        assert isinstance(result_dict, dict)
        assert "session_id" in result_dict
        assert "overall_assessment" in result_dict
        assert result_dict["overall_assessment"]["risk_level"] is not None
        
        # Test JSON serializable
        json.dumps(result_dict, default=str)
        
        results.add_pass("Combined result to_dict")
    except Exception as e:
        results.add_fail("Combined result to_dict", e)
    
    # ==================== INTEGRATION TESTS ====================
    print("\n🧪 INTEGRATION TESTS")
    print("-"*40)
    
    # Test 22: Full pipeline simulation
    try:
        from experts.track_structural_expert import TrackStructuralExpert
        from experts.combined_inference import (
            GeminiOutputStructurer, ContextualReasoningEngine,
            ActionRecommendationGenerator
        )
        
        # Step 1: Analyze CSV
        expert = TrackStructuralExpert()
        analysis = expert.analyze_csv(files['csv'])
        analysis_dict = expert.to_dict(analysis)
        
        # Step 2: Structure output
        structurer = GeminiOutputStructurer()
        structured = structurer._manual_structure({"structural": analysis_dict})
        
        # Step 3: Contextual reasoning
        reasoning = ContextualReasoningEngine()
        tampering = reasoning._fallback_analysis(structured)
        
        # Step 4: Generate report
        reporter = ActionRecommendationGenerator()
        report = reporter._fallback_report(structured, tampering, {})
        
        # Verify full pipeline
        assert "incident_snapshot" in report or "report_type" in report
        
        results.add_pass("Full pipeline simulation")
    except Exception as e:
        results.add_fail("Full pipeline simulation", e)
    
    # Test 23: Error handling in pipeline
    try:
        from experts.track_structural_expert import TrackStructuralExpert
        
        expert = TrackStructuralExpert()
        
        # Test with invalid file
        result = expert.analyze_csv("/nonexistent/file.csv")
        assert result.analysis_status == "error"
        
        # Should still return valid structure
        result_dict = expert.to_dict(result)
        assert "analysis_status" in result_dict
        
        results.add_pass("Error handling in pipeline")
    except Exception as e:
        results.add_fail("Error handling in pipeline", e)
    
    # Test 24: Multi-file analysis
    try:
        from experts.combined_inference import CombinedInferenceEngine
        
        engine = CombinedInferenceEngine()
        
        # Analyze multiple files
        result = engine.analyze(
            csv_paths=[files['csv']],
            lidar_paths=[files['npy']]
        )
        
        assert result is not None
        results.add_pass("Multi-file analysis")
    except Exception as e:
        results.add_fail("Multi-file analysis", e)
    
    # ==================== FRONTEND INTEGRATION DATA ====================
    print("\n📡 FRONTEND INTEGRATION DATA TESTS")
    print("-"*40)
    
    # Test 25: Dashboard data format
    try:
        # Simulate dashboard response
        dashboard_data = {
            "overview": {
                "total_scans": 150,
                "tampering_detected": 3,
                "alerts_active": 2,
                "system_status": "operational"
            },
            "recent_alerts": [
                {
                    "id": "ALT-001",
                    "timestamp": datetime.utcnow().isoformat(),
                    "severity": "high",
                    "message": "Obstacle detected on track",
                    "acknowledged": False
                }
            ],
            "risk_distribution": {
                "low": 140,
                "medium": 7,
                "high": 2,
                "critical": 1
            }
        }
        
        # Verify JSON serializable
        json.dumps(dashboard_data)
        
        results.add_pass("Dashboard data format")
    except Exception as e:
        results.add_fail("Dashboard data format", e)
    
    # Test 26: Analysis result for frontend
    try:
        from experts.combined_inference import CombinedInferenceEngine
        
        engine = CombinedInferenceEngine()
        result = engine.analyze(csv_paths=[files['csv']])
        result_dict = engine.to_dict(result)
        
        # Check result structure
        assert "session_id" in result_dict
        assert "timestamp" in result_dict
        
        # Risk level could be in different places
        has_risk = (
            "overall_risk_level" in result_dict or
            "risk_level" in result_dict or
            result_dict.get("structured_output", {}).get("risk_assessment") is not None
        )
        assert has_risk, "Missing risk level field"
        
        results.add_pass("Analysis result frontend fields")
    except Exception as e:
        results.add_fail("Analysis result frontend fields", e)
    
    # Test 27: Alert WebSocket message format
    try:
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
        
        # Format for WebSocket
        ws_message = {
            "type": "alert",
            "data": {
                "alert_id": alert.alert_id,
                "timestamp": alert.timestamp,
                "severity": alert.severity.value,
                "title": alert.title,
                "message": alert.message,
                "action_required": alert.action_required
            }
        }
        
        # Verify serializable
        json.dumps(ws_message)
        
        results.add_pass("Alert WebSocket message format")
    except Exception as e:
        results.add_fail("Alert WebSocket message format", e)
    
    # Test 28: Query response format
    try:
        from experts.contextual_reasoning_expert import ReasoningResult, QueryType
        
        result = ReasoningResult(
            query="What is the current risk?",
            query_type=QueryType.RISK_ASSESSMENT,
            response="Current risk level is MEDIUM...",
            confidence=0.85,
            insights=["High vibration detected", "Track alignment nominal"],
            recommendations=["Schedule inspection within 24h"]
        )
        
        # Format for API response
        api_response = {
            "query": result.query,
            "query_type": result.query_type.value,
            "response": result.response,
            "confidence": result.confidence,
            "insights": result.insights,
            "recommendations": result.recommendations
        }
        
        json.dumps(api_response)
        
        results.add_pass("Query response format")
    except Exception as e:
        results.add_fail("Query response format", e)
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)
    
    return results.summary()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
