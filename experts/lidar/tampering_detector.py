import numpy as np
from typing import Dict, List
from datetime import datetime


class TamperingDetector:
    def __init__(self, config: Dict = None):
        self.config = {
            'gauge_deviation_threshold': 0.05,
            'gauge_std_threshold': 0.03,
            'rail_misalignment_threshold': 0.08,
            'vertical_displacement_threshold': 0.05,
            'continuity_threshold': 0.85,
            'debris_density_threshold': 0.05,
            'debris_count_threshold': 100,
            'sleeper_missing_threshold': 0.3,
            'confidence_weights': {
                'gauge': 0.25,
                'alignment': 0.20,
                'vertical': 0.20,
                'continuity': 0.15,
                'debris': 0.15,
                'sleepers': 0.05
            }
        }
        
        if config:
            self.config.update(config)
    
    def detect_tampering(self, features: Dict, metadata: Dict = None) -> Dict:
        anomalies = []
        confidence_scores = {}
        
        gauge_anomaly = self._check_gauge_anomaly(features.get('track_gauge'))
        if gauge_anomaly:
            anomalies.append(gauge_anomaly)
            confidence_scores['gauge'] = gauge_anomaly['confidence']
        else:
            confidence_scores['gauge'] = 0.0
        
        alignment_anomaly = self._check_alignment_anomaly(features.get('rail_alignment'))
        if alignment_anomaly:
            anomalies.append(alignment_anomaly)
            confidence_scores['alignment'] = alignment_anomaly['confidence']
        else:
            confidence_scores['alignment'] = 0.0
        
        vertical_anomaly = self._check_vertical_anomaly(features.get('vertical_displacement'))
        if vertical_anomaly:
            anomalies.append(vertical_anomaly)
            confidence_scores['vertical'] = vertical_anomaly['confidence']
        else:
            confidence_scores['vertical'] = 0.0
        
        continuity_anomaly = self._check_continuity_anomaly(features.get('rail_continuity'))
        if continuity_anomaly:
            anomalies.append(continuity_anomaly)
            confidence_scores['continuity'] = continuity_anomaly['confidence']
        else:
            confidence_scores['continuity'] = 0.0
        
        debris_anomaly = self._check_debris_anomaly(
            features.get('debris_count', 0),
            features.get('debris_density', 0.0)
        )
        if debris_anomaly:
            anomalies.append(debris_anomaly)
            confidence_scores['debris'] = debris_anomaly['confidence']
        else:
            confidence_scores['debris'] = 0.0
        
        sleeper_anomaly = self._check_sleeper_anomaly(features.get('sleeper_count', 0), features)
        if sleeper_anomaly:
            anomalies.append(sleeper_anomaly)
            confidence_scores['sleepers'] = sleeper_anomaly['confidence']
        else:
            confidence_scores['sleepers'] = 0.0
        
        overall_confidence = self._calculate_overall_confidence(confidence_scores)
        
        tampering_detected = overall_confidence > 0.4
        severity = self._determine_severity(overall_confidence, anomalies)
        
        result = {
            'tampering_detected': tampering_detected,
            'confidence_score': round(overall_confidence, 3),
            'severity': severity,
            'anomaly_count': len(anomalies),
            'anomalies': anomalies,
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'features_analyzed': features,
            'metadata': metadata or {}
        }
        
        return result
    
    def _check_gauge_anomaly(self, gauge_data: Dict) -> Dict:
        if not gauge_data or gauge_data.get('mean') is None:
            return None
        
        mean_gauge = gauge_data['mean']
        std_gauge = gauge_data['std']
        
        standard_gauge = self.config.get('standard_gauge_m', 1.435)
        deviation = abs(mean_gauge - standard_gauge)
        
        if deviation > self.config['gauge_deviation_threshold'] or \
           std_gauge > self.config['gauge_std_threshold']:
            
            confidence = min(1.0, deviation / 0.15)
            
            return {
                'type': 'gauge_deviation',
                'description': 'Track gauge deviates from standard',
                'confidence': round(confidence, 3),
                'details': {
                    'measured_gauge_m': round(mean_gauge, 4),
                    'standard_gauge_m': standard_gauge,
                    'deviation_m': round(deviation, 4),
                    'std_deviation_m': round(std_gauge, 4)
                },
                'severity': 'high' if deviation > 0.10 else 'medium',
                'recommended_action': 'Inspect and adjust track gauge to standard specifications'
            }
        
        return None
    
    def _check_alignment_anomaly(self, alignment_data: Dict) -> Dict:
        if not alignment_data:
            return None
        
        deviation = alignment_data.get('deviation', 0.0)
        affected_rails = alignment_data.get('affected_rails', [])
        
        if deviation > self.config['rail_misalignment_threshold'] and affected_rails:
            confidence = min(1.0, deviation / 0.15)
            
            return {
                'type': 'rail_misalignment',
                'description': f'Rails are not properly aligned: {", ".join(affected_rails)}',
                'confidence': round(confidence, 3),
                'details': {
                    'max_deviation_m': round(deviation, 4),
                    'affected_rails': affected_rails
                },
                'severity': 'high' if deviation > 0.15 else 'medium',
                'recommended_action': 'Realign affected rails'
            }
        
        return None
    
    def _check_vertical_anomaly(self, vertical_data: Dict) -> Dict:
        if not vertical_data:
            return None
        
        max_displacement = vertical_data.get('max', 0.0)
        std_displacement = vertical_data.get('std', 0.0)
        
        if max_displacement > self.config['vertical_displacement_threshold']:
            confidence = min(1.0, max_displacement / 0.10)
            
            return {
                'type': 'vertical_displacement',
                'description': 'Abnormal vertical rail displacement detected',
                'confidence': round(confidence, 3),
                'details': {
                    'max_displacement_m': round(max_displacement, 4),
                    'std_deviation_m': round(std_displacement, 4)
                },
                'severity': 'high' if max_displacement > 0.08 else 'medium',
                'recommended_action': 'Inspect rail foundation and adjust vertical alignment'
            }
        
        return None
    
    def _check_continuity_anomaly(self, continuity_data: Dict) -> Dict:
        if not continuity_data:
            return None
        
        score = continuity_data.get('score', 1.0)
        gaps = continuity_data.get('gaps', [])
        
        if score < self.config['continuity_threshold'] and gaps:
            confidence = 1.0 - score
            
            return {
                'type': 'rail_discontinuity',
                'description': f'Rail continuity compromised with {len(gaps)} gap(s)',
                'confidence': round(confidence, 3),
                'details': {
                    'continuity_score': round(score, 3),
                    'gap_count': len(gaps),
                    'largest_gap_m': round(max(g['length'] for g in gaps), 3) if gaps else 0
                },
                'severity': 'critical' if score < 0.5 else 'high',
                'recommended_action': 'Immediate inspection for missing or damaged rail sections'
            }
        
        return None
    
    def _check_debris_anomaly(self, debris_count: int, debris_density: float) -> Dict:
        if debris_count > self.config['debris_count_threshold'] or \
           debris_density > self.config['debris_density_threshold']:
            
            confidence = min(1.0, debris_density / 0.10)
            
            return {
                'type': 'debris_presence',
                'description': 'Significant debris detected on track',
                'confidence': round(confidence, 3),
                'details': {
                    'debris_count': debris_count,
                    'debris_density': round(debris_density, 4)
                },
                'severity': 'medium' if debris_count < 500 else 'high',
                'recommended_action': 'Clear debris from track area'
            }
        
        return None
    
    def _check_sleeper_anomaly(self, sleeper_count: int, features: Dict) -> Dict:
        return None
    
    def _calculate_overall_confidence(self, confidence_scores: Dict) -> float:
        weights = self.config['confidence_weights']
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for key, weight in weights.items():
            if key in confidence_scores:
                weighted_sum += confidence_scores[key] * weight
                total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        return weighted_sum / total_weight
    
    def _determine_severity(self, confidence: float, anomalies: List[Dict]) -> str:
        if confidence >= 0.8:
            return 'critical'
        elif confidence >= 0.6:
            return 'high'
        elif confidence >= 0.4:
            return 'medium'
        else:
            return 'low'
