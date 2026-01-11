import json
from typing import Dict, Any
from datetime import datetime
from pathlib import Path


class OutputFormatter:
    def __init__(self, format_version: str = "1.0"):
        self.format_version = format_version
    
    def format_detection_output(self, 
                               detection_result: Dict,
                               input_file: str = None,
                               processing_time: float = None) -> Dict:
        output = {
            'version': self.format_version,
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'input': {
                'file': str(input_file) if input_file else None,
                'filename': Path(input_file).name if input_file else None
            },
            'result': {
                'status': 'TAMPERED' if detection_result['tampering_detected'] else 'NORMAL',
                'tampering_detected': detection_result['tampering_detected'],
                'confidence_score': detection_result['confidence_score'],
                'severity_level': detection_result['severity']
            },
            'analysis': {
                'anomaly_count': detection_result['anomaly_count'],
                'anomalies': self._format_anomalies(detection_result['anomalies']),
                'features': self._format_features(detection_result.get('features_analyzed', {}))
            },
            'location': self._format_location(detection_result.get('metadata', {})),
            'recommendations': self._format_recommendations(detection_result),
            'processing': {
                'timestamp': detection_result['timestamp'],
                'processing_time_seconds': round(processing_time, 3) if processing_time else None
            }
        }
        
        return output
    
    def _format_anomalies(self, anomalies: list) -> list:
        formatted = []
        
        for idx, anomaly in enumerate(anomalies, 1):
            formatted.append({
                'id': idx,
                'type': anomaly['type'],
                'description': anomaly['description'],
                'confidence': anomaly['confidence'],
                'severity': anomaly['severity'],
                'details': anomaly['details'],
                'recommended_action': anomaly.get('recommended_action')
            })
        
        return formatted
    
    def _format_features(self, features: Dict) -> Dict:
        return {
            'track_gauge': features.get('track_gauge'),
            'rail_alignment': features.get('rail_alignment'),
            'rail_continuity': features.get('rail_continuity'),
            'vertical_displacement': features.get('vertical_displacement'),
            'debris': {
                'count': features.get('debris_count', 0),
                'density': features.get('debris_density', 0.0)
            },
            'sleepers': {
                'count': features.get('sleeper_count', 0)
            }
        }
    
    def _format_location(self, metadata: Dict) -> Dict:
        location = {
            'coordinates': None,
            'track_section': None,
            'description': None
        }
        
        if 'gps' in metadata or 'coordinates' in metadata:
            coords = metadata.get('gps') or metadata.get('coordinates')
            if coords and len(coords) >= 2:
                location['coordinates'] = {
                    'latitude': coords[0],
                    'longitude': coords[1],
                    'format': 'WGS84'
                }
        
        if 'track_section' in metadata or 'location' in metadata:
            location['track_section'] = metadata.get('track_section') or metadata.get('location')
        
        if 'description' in metadata:
            location['description'] = metadata['description']
        
        return location
    
    def _format_recommendations(self, detection_result: Dict) -> list:
        recommendations = []
        
        if not detection_result['tampering_detected']:
            return [{
                'priority': 'low',
                'action': 'Continue regular monitoring',
                'urgency': 'routine'
            }]
        
        severity = detection_result['severity']
        urgency_map = {
            'low': 'routine',
            'medium': 'elevated',
            'high': 'urgent',
            'critical': 'immediate'
        }
        
        if severity in ['high', 'critical']:
            recommendations.append({
                'priority': 'critical',
                'action': 'Immediate track inspection required',
                'urgency': 'immediate',
                'note': 'Consider stopping train traffic until inspection'
            })
        
        for anomaly in detection_result['anomalies']:
            if 'recommended_action' in anomaly:
                recommendations.append({
                    'priority': anomaly['severity'],
                    'action': anomaly['recommended_action'],
                    'urgency': urgency_map.get(anomaly['severity'], 'routine'),
                    'related_anomaly': anomaly['type']
                })
        
        return recommendations
    
    def format_summary(self, detection_result: Dict) -> str:
        lines = []
        lines.append("="*70)
        lines.append("RAILWAY TRACK TAMPERING DETECTION REPORT")
        lines.append("="*70)
        lines.append(f"Timestamp: {detection_result['timestamp']}")
        lines.append("")
        
        status = "✓ TRACK NORMAL" if not detection_result['tampering_detected'] else "⚠ TAMPERING DETECTED"
        lines.append(f"Status: {status}")
        lines.append(f"Confidence: {detection_result['confidence_score']*100:.1f}%")
        lines.append(f"Severity: {detection_result['severity'].upper()}")
        lines.append("")
        
        lines.append(f"Anomalies Detected: {detection_result['anomaly_count']}")
        lines.append("-"*70)
        
        for idx, anomaly in enumerate(detection_result['anomalies'], 1):
            lines.append("")
            lines.append(f"{idx}. {anomaly['type'].replace('_', ' ').title()}")
            lines.append(f"   {anomaly['description']}")
            lines.append(f"   Confidence: {anomaly['confidence']*100:.1f}% | Severity: {anomaly['severity']}")
            if 'recommended_action' in anomaly:
                lines.append(f"   Action: {anomaly['recommended_action']}")
        
        lines.append("")
        lines.append("="*70)
        
        return "\n".join(lines)
    
    def save_json(self, data: Dict, output_path: str):
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)


class AlertGenerator:
    def generate_alert(self, detection_result: Dict) -> Dict:
        severity = detection_result['severity']
        anomaly_count = detection_result['anomaly_count']
        
        priority_map = {
            'low': 'INFO',
            'medium': 'WARNING',
            'high': 'ALERT',
            'critical': 'CRITICAL'
        }
        
        title = f"Track Tampering {priority_map.get(severity, 'ALERT')}"
        
        anomaly_types = [a['type'].replace('_', ' ').title() for a in detection_result['anomalies']]
        
        message = f"Detected {anomaly_count} anomalies: {', '.join(anomaly_types)}"
        
        alert = {
            'title': title,
            'message': message,
            'severity': severity,
            'priority': priority_map.get(severity, 'ALERT'),
            'timestamp': detection_result['timestamp'],
            'requires_immediate_action': severity in ['high', 'critical']
        }
        
        return alert
