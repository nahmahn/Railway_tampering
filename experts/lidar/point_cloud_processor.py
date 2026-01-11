import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional


class PointCloudProcessor:
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        rail_params = self.config.get('rail_parameters', {})
        self.standard_gauge = rail_params.get('standard_gauge_m', 1.435)
        self.rail_height_range = tuple(rail_params.get('rail_height_range_m', [0.10, 0.20]))
        self.track_width_tolerance = rail_params.get('track_width_tolerance_m', 0.05)
        
        outlier_config = self.config.get('outlier_removal', {})
        self.outlier_enabled = outlier_config.get('enabled', True)
        self.outlier_neighbors = outlier_config.get('nb_neighbors', 20)
        self.outlier_std_ratio = outlier_config.get('std_ratio', 2.0)
        
    def load_point_cloud(self, file_path: str) -> np.ndarray:
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Point cloud file not found: {file_path}")
        
        if path.suffix == '.npy':
            points = np.load(file_path)
        elif path.suffix in ['.pcd', '.ply']:
            try:
                import open3d as o3d
                pcd = o3d.io.read_point_cloud(str(file_path))
                points = np.asarray(pcd.points)
            except ImportError:
                raise ImportError("open3d required for .pcd/.ply files")
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
        
        if points.shape[1] != 3:
            raise ValueError(f"Expected point cloud with 3 coordinates (x,y,z), got shape {points.shape}")
        
        print(f"Loaded {len(points)} points from {path.name}")
        return points
    
    def preprocess(self, points: np.ndarray) -> np.ndarray:
        if self.outlier_enabled:
            points = self._remove_statistical_outliers(points)
        
        print(f"Preprocessed point cloud: {len(points)} points remaining")
        return points
    
    def _remove_statistical_outliers(self, points: np.ndarray) -> np.ndarray:
        from scipy.spatial import cKDTree
        
        if len(points) < self.outlier_neighbors:
            return points
        
        tree = cKDTree(points)
        distances, _ = tree.query(points, k=self.outlier_neighbors+1)
        
        avg_distances = np.mean(distances[:, 1:], axis=1)
        
        mean_dist = np.mean(avg_distances)
        std_dist = np.std(avg_distances)
        threshold = mean_dist + self.outlier_std_ratio * std_dist
        
        mask = avg_distances < threshold
        return points[mask]
    
    def segment_rails(self, points: np.ndarray) -> Dict[str, np.ndarray]:
        z_min, z_max = self.rail_height_range
        rail_candidates = points[(points[:, 2] >= z_min) & (points[:, 2] <= z_max)]
        
        if len(rail_candidates) == 0:
            return {
                'rails': np.array([]),
                'sleepers': np.array([]),
                'ground': points,
                'debris': np.array([])
            }
        
        y_coords = rail_candidates[:, 1]
        
        left_rail_mask = y_coords < -0.3
        right_rail_mask = y_coords > 0.3
        
        left_rail = rail_candidates[left_rail_mask]
        right_rail = rail_candidates[right_rail_mask]
        rails = np.vstack([left_rail, right_rail]) if len(left_rail) > 0 and len(right_rail) > 0 else rail_candidates
        
        sleeper_mask = (points[:, 2] < z_min) & (points[:, 2] > 0.02)
        sleeper_mask &= (np.abs(points[:, 1]) < 1.0)
        sleepers = points[sleeper_mask]
        
        ground_mask = points[:, 2] <= 0.02
        ground = points[ground_mask]
        
        debris_mask = points[:, 2] > z_max
        debris = points[debris_mask]
        
        segments = {
            'rails': rails,
            'sleepers': sleepers,
            'ground': ground,
            'debris': debris
        }
        
        print(f"Segmentation: {len(rails)} rail points, {len(sleepers)} sleeper points, "
              f"{len(ground)} ground points, {len(debris)} debris points")
        
        return segments
    
    def extract_features(self, points: np.ndarray, segments: Dict[str, np.ndarray]) -> Dict:
        features = {}
        
        rails = segments['rails']
        if len(rails) > 0:
            features['track_gauge'] = self._compute_track_gauge(rails)
            features['rail_alignment'] = self._compute_rail_alignment(rails)
            features['rail_continuity'] = self._compute_rail_continuity(rails)
            features['vertical_displacement'] = self._compute_vertical_displacement(rails)
        else:
            features['track_gauge'] = {'mean': None, 'std': None, 'min': None, 'max': None}
            features['rail_alignment'] = {'deviation': 0.0, 'affected_rails': []}
            features['rail_continuity'] = {'score': 0.0, 'gaps': []}
            features['vertical_displacement'] = {'mean': 0.0, 'max': 0.0, 'std': 0.0}
        
        features['debris_count'] = len(segments['debris'])
        features['debris_density'] = len(segments['debris']) / len(points) if len(points) > 0 else 0.0
        features['sleeper_count'] = len(segments['sleepers'])
        
        return features
    
    def _compute_track_gauge(self, rails: np.ndarray) -> Dict:
        y_coords = rails[:, 1]
        
        left_points = rails[y_coords < 0]
        right_points = rails[y_coords > 0]
        
        if len(left_points) == 0 or len(right_points) == 0:
            return {'mean': None, 'std': None, 'min': None, 'max': None}
        
        gauges = []
        x_positions = np.unique(np.round(rails[:, 0], 1))
        
        for x in x_positions:
            left_at_x = left_points[np.isclose(left_points[:, 0], x, atol=0.1)]
            right_at_x = right_points[np.isclose(right_points[:, 0], x, atol=0.1)]
            
            if len(left_at_x) > 0 and len(right_at_x) > 0:
                left_y = np.mean(left_at_x[:, 1])
                right_y = np.mean(right_at_x[:, 1])
                gauge = abs(right_y - left_y)
                gauges.append(gauge)
        
        if not gauges:
            return {'mean': None, 'std': None, 'min': None, 'max': None}
        
        return {
            'mean': float(np.mean(gauges)),
            'std': float(np.std(gauges)),
            'min': float(np.min(gauges)),
            'max': float(np.max(gauges))
        }
    
    def _compute_rail_alignment(self, rails: np.ndarray) -> Dict:
        y_coords = rails[:, 1]
        left_rail = rails[y_coords < 0]
        right_rail = rails[y_coords > 0]
        
        affected_rails = []
        max_deviation = 0.0
        
        for rail_points, rail_name in [(left_rail, 'left'), (right_rail, 'right')]:
            if len(rail_points) > 10:
                y_std = np.std(rail_points[:, 1])
                z_std = np.std(rail_points[:, 2])
                deviation = float(np.sqrt(y_std**2 + z_std**2))
                
                if deviation > 0.05:
                    affected_rails.append(rail_name)
                    max_deviation = max(max_deviation, deviation)
        
        return {
            'deviation': max_deviation,
            'affected_rails': affected_rails
        }
    
    def _compute_rail_continuity(self, rails: np.ndarray) -> Dict:
        if len(rails) == 0:
            return {'score': 0.0, 'gaps': []}
        
        x_coords = rails[:, 0]
        x_sorted = np.sort(x_coords)
        
        gaps = []
        if len(x_sorted) > 1:
            diffs = np.diff(x_sorted)
            median_spacing = np.median(diffs)
            threshold = median_spacing * 3
            
            gap_indices = np.where(diffs > threshold)[0]
            for idx in gap_indices:
                gaps.append({
                    'start': float(x_sorted[idx]),
                    'end': float(x_sorted[idx + 1]),
                    'length': float(diffs[idx])
                })
        
        total_length = float(x_sorted[-1] - x_sorted[0]) if len(x_sorted) > 1 else 0.0
        gap_length = sum(g['length'] for g in gaps)
        continuity_score = 1.0 - (gap_length / total_length) if total_length > 0 else 1.0
        
        return {
            'score': max(0.0, continuity_score),
            'gaps': gaps
        }
    
    def _compute_vertical_displacement(self, rails: np.ndarray) -> Dict:
        if len(rails) == 0:
            return {'mean': 0.0, 'max': 0.0, 'std': 0.0}
        
        z_coords = rails[:, 2]
        z_mean = np.mean(z_coords)
        
        displacements = np.abs(z_coords - z_mean)
        
        return {
            'mean': float(np.mean(displacements)),
            'max': float(np.max(displacements)),
            'std': float(np.std(z_coords))
        }
