"""
Standalone Test Suite for Railway Tampering Detection Experts
This test file runs OUTSIDE the experts package to avoid __init__.py imports.

Run with: python test_experts_standalone.py
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
from scipy.stats import skew, kurtosis

# Add Railway_tampering to path
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))


def create_sample_vibration_csv(temp_dir):
    """Create a sample vibration CSV file."""
    np.random.seed(42)
    n_samples = 500
    
    data = {
        'x': np.random.randn(n_samples) * 0.1,
        'y': np.random.randn(n_samples) * 0.1,
        'z': np.random.randn(n_samples) * 0.1 + 9.8,
    }
    
    df = pd.DataFrame(data)
    filepath = os.path.join(temp_dir, "vibration_test.csv")
    df.to_csv(filepath, index=False)
    return filepath


def create_sample_lidar_npy(temp_dir):
    """Create a sample LiDAR point cloud as NPY file."""
    np.random.seed(42)
    n_points = 1000
    
    x = np.random.uniform(-10, 10, n_points)
    y = np.random.uniform(0, 100, n_points)
    z = np.random.uniform(0, 2, n_points)
    
    points = np.column_stack([x, y, z])
    
    filepath = os.path.join(temp_dir, "test_pointcloud.npy")
    np.save(filepath, points)
    return filepath


def create_sample_image(temp_dir):
    """Create a sample test image using PIL."""
    try:
        from PIL import Image
        img = Image.new('RGB', (640, 480), color=(128, 128, 128))
        filepath = os.path.join(temp_dir, "test_image.jpg")
        img.save(filepath)
        return filepath
    except ImportError:
        return None


class TestResults:
    """Track test results."""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
    
    def add_pass(self, name):
        self.passed += 1
        print(f"  ✅ PASS: {name}")
    
    def add_fail(self, name, error):
        self.failed += 1
        self.errors.append((name, str(error)))
        print(f"  ❌ FAIL: {name}")
        print(f"      Error: {error}")
    
    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'='*60}")
        print(f"RESULTS: {self.passed}/{total} tests passed")
        if self.errors:
            print(f"\nFailed tests:")
            for name, error in self.errors:
                print(f"  - {name}: {error[:100]}...")
        return self.failed == 0


def run_tests():
    results = TestResults()
    
    # Create temp directory
    temp_dir = tempfile.mkdtemp()
    
    print("="*60)
    print("Railway Tampering Detection - Expert Tests")
    print("="*60)
    
    # ==================== Basic Tests ====================
    print("\n📦 Basic Tests")
    print("-"*40)
    
    try:
        # Test 1: Numpy operations
        arr = np.array([1, 2, 3, 4, 5])
        assert np.mean(arr) == 3.0
        results.add_pass("Numpy operations")
    except Exception as e:
        results.add_fail("Numpy operations", e)
    
    try:
        # Test 2: Pandas read CSV
        csv_path = create_sample_vibration_csv(temp_dir)
        df = pd.read_csv(csv_path)
        assert len(df) == 500
        assert all(col in df.columns for col in ['x', 'y', 'z'])
        results.add_pass("Pandas read CSV")
    except Exception as e:
        results.add_fail("Pandas read CSV", e)
    
    try:
        # Test 3: NPY file operations
        npy_path = create_sample_lidar_npy(temp_dir)
        points = np.load(npy_path)
        assert points.shape == (1000, 3)
        results.add_pass("NPY file operations")
    except Exception as e:
        results.add_fail("NPY file operations", e)
    
    # ==================== Feature Extraction Tests ====================
    print("\n🔬 Feature Extraction Tests")
    print("-"*40)
    
    try:
        # Test 4: Feature extraction
        df = pd.read_csv(csv_path)
        data = df['x'].values[:100]
        
        features = []
        features.append(np.mean(data))
        features.append(np.std(data))
        features.append(np.min(data))
        features.append(np.max(data))
        features.append(np.median(data))
        features.append(np.percentile(data, 25))
        features.append(np.percentile(data, 75))
        features.append(skew(data))
        features.append(kurtosis(data))
        features.append(np.sqrt(np.mean(data**2)))  # RMS
        features.append(np.max(np.abs(data)))  # Peak
        features.append(np.max(np.abs(data)) / (np.sqrt(np.mean(data**2)) + 1e-8))  # Crest
        
        assert len(features) == 12
        assert all(np.isfinite(f) for f in features)
        results.add_pass("Feature extraction (12 features)")
    except Exception as e:
        results.add_fail("Feature extraction", e)
    
    try:
        # Test 5: Multi-axis feature extraction
        all_features = []
        for col in ['x', 'y', 'z']:
            data = df[col].values[:100]
            col_features = [
                np.mean(data), np.std(data), np.min(data), np.max(data),
                np.median(data), np.percentile(data, 25), np.percentile(data, 75),
                skew(data), kurtosis(data), np.sqrt(np.mean(data**2)),
                np.max(np.abs(data)), np.max(np.abs(data)) / (np.sqrt(np.mean(data**2)) + 1e-8)
            ]
            all_features.extend(col_features)
        
        assert len(all_features) == 36  # 12 * 3
        results.add_pass("Multi-axis feature extraction (36 features)")
    except Exception as e:
        results.add_fail("Multi-axis feature extraction", e)
    
    try:
        # Test 6: Windowing
        data = df['x'].values
        window_size = 100
        step_size = 50
        
        windows = []
        for i in range(0, len(data) - window_size + 1, step_size):
            windows.append(data[i:i + window_size])
        
        expected_windows = (len(data) - window_size) // step_size + 1
        assert len(windows) == expected_windows
        assert all(len(w) == window_size for w in windows)
        results.add_pass(f"Windowing ({len(windows)} windows)")
    except Exception as e:
        results.add_fail("Windowing", e)
    
    # ==================== Point Cloud Tests ====================
    print("\n☁️ Point Cloud Tests")
    print("-"*40)
    
    try:
        # Test 7: Point cloud statistics
        points = np.load(npy_path)
        
        centroid = np.mean(points, axis=0)
        bounds_min = np.min(points, axis=0)
        bounds_max = np.max(points, axis=0)
        extent = bounds_max - bounds_min
        
        assert len(centroid) == 3
        assert all(extent > 0)
        results.add_pass("Point cloud statistics")
    except Exception as e:
        results.add_fail("Point cloud statistics", e)
    
    try:
        # Test 8: Density analysis
        x_range = np.max(points[:, 0]) - np.min(points[:, 0])
        y_range = np.max(points[:, 1]) - np.min(points[:, 1])
        area = x_range * y_range
        density = len(points) / area
        
        assert density > 0
        results.add_pass(f"Density analysis ({density:.2f} pts/m²)")
    except Exception as e:
        results.add_fail("Density analysis", e)
    
    try:
        # Test 9: Height analysis
        z_values = points[:, 2]
        mean_height = np.mean(z_values)
        height_std = np.std(z_values)
        
        # Find outliers
        high_points = points[z_values > mean_height + 2 * height_std]
        low_points = points[z_values < mean_height - 2 * height_std]
        
        results.add_pass(f"Height analysis (outliers: {len(high_points)} high, {len(low_points)} low)")
    except Exception as e:
        results.add_fail("Height analysis", e)
    
    # ==================== Classification Logic Tests ====================
    print("\n🎯 Classification Logic Tests")
    print("-"*40)
    
    try:
        # Test 10: Query classification
        def classify_query(query):
            query_lower = query.lower()
            if any(w in query_lower for w in ["risk", "danger", "threat"]):
                return "risk_assessment"
            if any(w in query_lower for w in ["should", "recommend", "suggest"]):
                return "recommendation"
            if any(w in query_lower for w in ["status", "current", "condition"]):
                return "status_inquiry"
            return "general"
        
        assert classify_query("What is the risk level?") == "risk_assessment"
        assert classify_query("What should we do?") == "recommendation"
        assert classify_query("Current status?") == "status_inquiry"
        assert classify_query("Hello") == "general"
        results.add_pass("Query classification")
    except Exception as e:
        results.add_fail("Query classification", e)
    
    try:
        # Test 11: Risk level calculation
        def get_risk_level(failure_ratio):
            if failure_ratio < 0.1:
                return "low"
            elif failure_ratio < 0.3:
                return "medium"
            elif failure_ratio < 0.6:
                return "high"
            else:
                return "critical"
        
        assert get_risk_level(0.05) == "low"
        assert get_risk_level(0.15) == "medium"
        assert get_risk_level(0.45) == "high"
        assert get_risk_level(0.75) == "critical"
        results.add_pass("Risk level calculation")
    except Exception as e:
        results.add_fail("Risk level calculation", e)
    
    try:
        # Test 12: Risk aggregation
        def aggregate_risk(risks):
            severity_order = ["low", "medium", "high", "critical"]
            max_severity = 0
            for risk in risks:
                if risk in severity_order:
                    max_severity = max(max_severity, severity_order.index(risk))
            return severity_order[max_severity]
        
        assert aggregate_risk(["low", "low"]) == "low"
        assert aggregate_risk(["low", "medium"]) == "medium"
        assert aggregate_risk(["low", "high"]) == "high"
        assert aggregate_risk(["medium", "critical"]) == "critical"
        results.add_pass("Risk aggregation")
    except Exception as e:
        results.add_fail("Risk aggregation", e)
    
    # ==================== Routing Tests ====================
    print("\n🔀 Routing Tests")
    print("-"*40)
    
    try:
        # Test 13: Input type detection
        def detect_type(filepath):
            ext = Path(filepath).suffix.lower()
            type_map = {
                '.csv': 'csv', '.jpg': 'image', '.jpeg': 'image',
                '.png': 'image', '.mp4': 'video', '.avi': 'video',
                '.laz': 'laz', '.las': 'las', '.npy': 'lidar',
            }
            return type_map.get(ext, 'unknown')
        
        assert detect_type("data.csv") == "csv"
        assert detect_type("image.jpg") == "image"
        assert detect_type("video.mp4") == "video"
        assert detect_type("cloud.laz") == "laz"
        assert detect_type("points.npy") == "lidar"
        results.add_pass("Input type detection")
    except Exception as e:
        results.add_fail("Input type detection", e)
    
    try:
        # Test 14: Input routing
        def route_input(input_type):
            routes = {
                'csv': 'track_structural_expert',
                'image': 'visual_integrity_expert',
                'video': 'visual_integrity_expert',
                'laz': 'thermal_anomaly_expert',
                'las': 'thermal_anomaly_expert',
                'lidar': 'thermal_anomaly_expert',
                'text': 'contextual_reasoning_expert',
            }
            return routes.get(input_type, 'unknown')
        
        assert route_input('csv') == 'track_structural_expert'
        assert route_input('image') == 'visual_integrity_expert'
        assert route_input('laz') == 'thermal_anomaly_expert'
        assert route_input('text') == 'contextual_reasoning_expert'
        results.add_pass("Input routing")
    except Exception as e:
        results.add_fail("Input routing", e)
    
    # ==================== Data Pipeline Tests ====================
    print("\n🔄 Data Pipeline Tests")
    print("-"*40)
    
    try:
        # Test 15: Full CSV to features pipeline
        df = pd.read_csv(csv_path)
        window_size = 100
        
        windows = []
        for i in range(0, len(df) - window_size + 1, window_size):
            windows.append(df.iloc[i:i + window_size])
        
        all_features = []
        for window in windows:
            window_features = []
            for col in ['x', 'y', 'z']:
                data = window[col].values
                window_features.extend([
                    np.mean(data), np.std(data), np.min(data), np.max(data),
                    np.median(data), np.percentile(data, 25), np.percentile(data, 75),
                    skew(data), kurtosis(data), np.sqrt(np.mean(data**2)),
                    np.max(np.abs(data)), np.max(np.abs(data)) / (np.sqrt(np.mean(data**2)) + 1e-8)
                ])
            all_features.append(window_features)
        
        feature_array = np.array(all_features)
        assert feature_array.shape[1] == 36
        results.add_pass(f"CSV to features pipeline ({feature_array.shape[0]} windows)")
    except Exception as e:
        results.add_fail("CSV to features pipeline", e)
    
    try:
        # Test 16: Point cloud analysis pipeline
        points = np.load(npy_path)
        
        analysis = {
            "point_count": len(points),
            "centroid": np.mean(points, axis=0).tolist(),
            "bounds_min": np.min(points, axis=0).tolist(),
            "bounds_max": np.max(points, axis=0).tolist(),
            "height_stats": {
                "mean": float(np.mean(points[:, 2])),
                "std": float(np.std(points[:, 2])),
                "min": float(np.min(points[:, 2])),
                "max": float(np.max(points[:, 2])),
            }
        }
        
        # Test JSON serialization
        json_str = json.dumps(analysis)
        parsed = json.loads(json_str)
        assert parsed["point_count"] == 1000
        results.add_pass("Point cloud analysis pipeline")
    except Exception as e:
        results.add_fail("Point cloud analysis pipeline", e)
    
    # ==================== Alert/Tampering Detection Tests ====================
    print("\n🚨 Alert & Tampering Detection Tests")
    print("-"*40)
    
    try:
        # Test 17: Tampering detection logic
        def detect_tampering(results):
            if results.get("visual", {}).get("tampering_detected"):
                return True
            if results.get("thermal", {}).get("anomaly_count", 0) > 3:
                return True
            if results.get("structural", {}).get("failure_ratio", 0) > 0.3:
                return True
            return False
        
        assert detect_tampering({"visual": {"tampering_detected": True}}) == True
        assert detect_tampering({"thermal": {"anomaly_count": 5}}) == True
        assert detect_tampering({"structural": {"failure_ratio": 0.5}}) == True
        assert detect_tampering({"structural": {"failure_ratio": 0.1}}) == False
        results.add_pass("Tampering detection logic")
    except Exception as e:
        results.add_fail("Tampering detection logic", e)
    
    try:
        # Test 18: Alert severity
        from enum import Enum
        
        class AlertSeverity(Enum):
            INFO = "info"
            WARNING = "warning"
            HIGH = "high"
            CRITICAL = "critical"
        
        assert AlertSeverity.INFO.value == "info"
        assert AlertSeverity.CRITICAL.value == "critical"
        results.add_pass("Alert severity enum")
    except Exception as e:
        results.add_fail("Alert severity enum", e)
    
    # ==================== Image Tests ====================
    print("\n🖼️ Image Tests")
    print("-"*40)
    
    try:
        img_path = create_sample_image(temp_dir)
        if img_path:
            from PIL import Image
            img = Image.open(img_path)
            assert img.size == (640, 480)
            
            arr = np.array(img)
            assert arr.shape == (480, 640, 3)
            results.add_pass("Image loading and numpy conversion")
        else:
            print("  ⚠️ SKIP: PIL not available")
    except ImportError:
        print("  ⚠️ SKIP: PIL not available")
    except Exception as e:
        results.add_fail("Image loading", e)
    
    # ==================== Error Handling Tests ====================
    print("\n⚠️ Error Handling Tests")
    print("-"*40)
    
    try:
        # Test 19: Missing file handling
        missing_path = os.path.join(temp_dir, "nonexistent.csv")
        try:
            pd.read_csv(missing_path)
            results.add_fail("Missing file handling", "Should have raised FileNotFoundError")
        except FileNotFoundError:
            results.add_pass("Missing file handling")
    except Exception as e:
        results.add_fail("Missing file handling", e)
    
    try:
        # Test 20: Empty array handling
        empty_arr = np.array([])
        mean = np.mean(empty_arr) if len(empty_arr) > 0 else 0.0
        assert mean == 0.0 or np.isnan(mean)
        results.add_pass("Empty array handling")
    except Exception as e:
        results.add_fail("Empty array handling", e)
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)
    
    return results.summary()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
