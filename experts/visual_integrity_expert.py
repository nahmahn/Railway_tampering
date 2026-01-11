"""
Visual Track Integrity Expert - IMPLEMENTED

Uses YOLO for object detection, Grounding DINO for zero-shot detection,
and SAM for segmentation to detect track tampering.

Based on hackathon-hack4delhi.ipynb implementation.

INPUT TYPES:
- Images: JPG, JPEG, PNG, BMP, TIFF (CCTV frames, drone captures)
- Videos: MP4, AVI, MOV, MKV (CCTV recordings, drone videos)
- JSON: Metadata, annotations, camera calibration

OUTPUT:
- Return ExpertResult with detections, segmentation results, and alerts
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import os
import numpy as np
from datetime import datetime

# Optional imports - gracefully handle missing dependencies
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

try:
    import torch
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    GROUNDING_DINO_AVAILABLE = True
except ImportError:
    GROUNDING_DINO_AVAILABLE = False

try:
    from segment_anything import sam_model_registry, SamPredictor
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False


class DetectionType(Enum):
    """Types of detections from visual analysis."""
    OBSTACLE = "obstacle"
    CRACK = "crack"
    DEFORMATION = "deformation"
    MISSING_COMPONENT = "missing_component"
    VEGETATION = "vegetation"
    DEBRIS = "debris"
    TAMPERING = "tampering"
    PERSON = "person"
    ANIMAL = "animal"
    VEHICLE = "vehicle"
    TRAIN = "train"
    STONE = "stone"
    FOREIGN_OBJECT = "foreign_object"
    UNKNOWN = "unknown"


@dataclass
class Detection:
    """A single detection from visual analysis."""
    detection_type: DetectionType
    confidence: float
    bounding_box: Optional[Tuple[int, int, int, int]] = None  # x1, y1, x2, y2
    segmentation_mask: Optional[Any] = None
    frame_number: Optional[int] = None
    timestamp: Optional[float] = None
    label: str = ""
    on_track: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VisualIntegrityResult:
    """Result structure for visual track integrity analysis."""
    file_path: str
    file_type: str
    analysis_status: str
    
    # Detection results
    detections: List[Detection] = field(default_factory=list)
    detection_summary: Dict[str, int] = field(default_factory=dict)
    
    # Segmentation results
    track_segmentation_score: float = 0.0
    rail_visibility: float = 0.0
    
    # Anomaly scores
    tampering_probability: float = 0.0
    obstruction_probability: float = 0.0
    damage_probability: float = 0.0
    
    # Tampering status
    person_detected: bool = False
    person_on_track: bool = False
    foreign_object_on_track: bool = False
    tampering_types: List[str] = field(default_factory=list)
    
    # Video-specific
    frame_count: int = 0
    fps: float = 0.0
    temporal_anomalies: List[Dict[str, Any]] = field(default_factory=list)
    
    # Overall assessment
    risk_level: str = "low"
    alerts: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    confidence: float = 0.0
    timestamp: str = ""


class VisualIntegrityExpert:
    """
    Visual Track Integrity Expert using YOLO, Grounding DINO, and SAM.
    """
    
    def __init__(
        self,
        yolo_model_path: str = "yolov8n.pt",
        sam_checkpoint: str = None,
        device: str = None
    ):
        """Initialize the expert with required models."""
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu") if GROUNDING_DINO_AVAILABLE else "cpu"
        
        # Initialize YOLO
        self.yolo_model = None
        if YOLO_AVAILABLE:
            try:
                self.yolo_model = YOLO(yolo_model_path)
            except Exception as e:
                print(f"Warning: Could not load YOLO model: {e}")
        
        # Initialize Grounding DINO
        self.gd_processor = None
        self.gd_model = None
        if GROUNDING_DINO_AVAILABLE:
            try:
                self.gd_processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
                self.gd_model = AutoModelForZeroShotObjectDetection.from_pretrained(
                    "IDEA-Research/grounding-dino-base"
                ).to(self.device)
            except Exception as e:
                print(f"Warning: Could not load Grounding DINO: {e}")
        
        # Initialize SAM
        self.sam_predictor = None
        self.sam_checkpoint = sam_checkpoint
        if SAM_AVAILABLE and sam_checkpoint and os.path.exists(sam_checkpoint):
            try:
                sam = sam_model_registry["vit_b"](checkpoint=sam_checkpoint)
                sam.to(self.device)
                self.sam_predictor = SamPredictor(sam)
            except Exception as e:
                print(f"Warning: Could not load SAM: {e}")
        
        # Tampering detection prompts for Grounding DINO
        self.tampering_texts = [
            "large stone on railway track",
            "rock obstructing railway rail",
            "object placed on railway track",
            "railway track obstruction",
            "debris on railway track",
            "foreign object on track"
        ]
        
        self.box_threshold = 0.3
    
    def create_track_mask(self, image: np.ndarray) -> np.ndarray:
        """Create a mask for the approximate rail corridor."""
        h, w = image.shape[:2]
        track_mask = np.zeros((h, w), dtype=np.uint8)
        # Approximate rail corridor (center-bottom of image)
        track_mask[int(h * 0.55):h, int(w * 0.25):int(w * 0.75)] = 1
        return track_mask
    
    def overlaps_track(self, box: Tuple, mask: np.ndarray) -> bool:
        """Check if a bounding box overlaps with the track mask."""
        x1, y1, x2, y2 = map(int, box[:4])
        h, w = mask.shape
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return False
        return mask[y1:y2, x1:x2].sum() > 0
    
    def run_yolo_detection(self, image_path: str) -> Tuple[List[Detection], Dict]:
        """Run YOLO detection on an image."""
        detections = []
        raw_results = {}
        
        if not self.yolo_model:
            return detections, raw_results
        
        try:
            yolo_results = self.yolo_model(image_path)
            r = yolo_results[0]
            
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            clss = r.boxes.cls.cpu().numpy()
            names = r.names
            
            raw_results = {
                "boxes": boxes.tolist(),
                "confidences": confs.tolist(),
                "classes": [names[int(c)] for c in clss]
            }
            
            for box, cls, conf in zip(boxes, clss, confs):
                label = names[int(cls)]
                det_type = self._map_yolo_class_to_detection_type(label)
                
                detections.append(Detection(
                    detection_type=det_type,
                    confidence=float(conf),
                    bounding_box=tuple(map(int, box)),
                    label=label,
                    metadata={"source": "yolo"}
                ))
        except Exception as e:
            print(f"YOLO detection error: {e}")
        
        return detections, raw_results
    
    def run_grounding_dino_detection(
        self, 
        image_path: str,
        texts: List[str] = None
    ) -> Tuple[List[Detection], Dict]:
        """Run Grounding DINO for zero-shot object detection."""
        detections = []
        raw_results = {}
        
        if not self.gd_model or not self.gd_processor:
            return detections, raw_results
        
        texts = texts or self.tampering_texts
        
        try:
            image_pil = Image.open(image_path).convert("RGB")
            inputs = self.gd_processor(images=image_pil, text=texts, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.gd_model(**inputs)
            
            results = self.gd_processor.post_process_grounded_object_detection(
                outputs,
                inputs["input_ids"],
                target_sizes=[image_pil.size[::-1]]
            )[0]
            
            gd_boxes = []
            gd_labels = []
            gd_scores = []
            
            for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
                if score >= self.box_threshold:
                    gd_boxes.append(box.cpu().numpy())
                    gd_labels.append(label)
                    gd_scores.append(float(score))
                    
                    detections.append(Detection(
                        detection_type=DetectionType.FOREIGN_OBJECT,
                        confidence=float(score),
                        bounding_box=tuple(map(int, box.cpu().numpy())),
                        label=label,
                        metadata={"source": "grounding_dino"}
                    ))
            
            raw_results = {
                "boxes": [b.tolist() for b in gd_boxes],
                "labels": gd_labels,
                "scores": gd_scores
            }
        except Exception as e:
            print(f"Grounding DINO detection error: {e}")
        
        return detections, raw_results
    
    def run_sam_segmentation(
        self, 
        image: np.ndarray, 
        boxes: List[Tuple]
    ) -> List[np.ndarray]:
        """Run SAM segmentation on detected boxes."""
        masks = []
        
        if not self.sam_predictor or len(boxes) == 0:
            return masks
        
        try:
            self.sam_predictor.set_image(image)
            
            for box in boxes:
                x1, y1, x2, y2 = map(int, box[:4])
                mask_results, _, _ = self.sam_predictor.predict(
                    box=np.array([x1, y1, x2, y2]),
                    multimask_output=False
                )
                masks.append(mask_results[0])
        except Exception as e:
            print(f"SAM segmentation error: {e}")
        
        return masks
    
    def _map_yolo_class_to_detection_type(self, label: str) -> DetectionType:
        """Map YOLO class names to DetectionType enum."""
        mapping = {
            "person": DetectionType.PERSON,
            "train": DetectionType.TRAIN,
            "car": DetectionType.VEHICLE,
            "truck": DetectionType.VEHICLE,
            "bus": DetectionType.VEHICLE,
            "cat": DetectionType.ANIMAL,
            "dog": DetectionType.ANIMAL,
            "cow": DetectionType.ANIMAL,
            "horse": DetectionType.ANIMAL,
            "bird": DetectionType.ANIMAL,
        }
        return mapping.get(label.lower(), DetectionType.UNKNOWN)
    
    def analyze_image(
        self,
        file_path: str,
        annotations: Dict[str, Any] = None,
        metadata: Dict[str, Any] = None
    ) -> VisualIntegrityResult:
        """
        Analyze an image for track integrity issues.
        
        Args:
            file_path: Path to the image file
            annotations: Optional JSON annotations
            metadata: Additional metadata
        
        Returns:
            VisualIntegrityResult with detections and analysis
        """
        result = VisualIntegrityResult(
            file_path=file_path,
            file_type="image",
            analysis_status="processing",
            timestamp=datetime.utcnow().isoformat() + "Z"
        )
        
        if not CV2_AVAILABLE:
            result.analysis_status = "error"
            result.alerts.append("OpenCV not available")
            return result
        
        try:
            # Load image
            img = cv2.imread(file_path)
            if img is None:
                result.analysis_status = "error"
                result.alerts.append(f"Could not load image: {file_path}")
                return result
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]
            
            # Create track mask
            track_mask = self.create_track_mask(img)
            
            # Run YOLO detection
            yolo_detections, yolo_raw = self.run_yolo_detection(file_path)
            
            # Filter person detections
            person_boxes = [
                d.bounding_box for d in yolo_detections 
                if d.detection_type == DetectionType.PERSON
            ]
            result.person_detected = len(person_boxes) > 0
            
            # Check if person is on track
            result.person_on_track = any(
                self.overlaps_track(box, track_mask) for box in person_boxes
            )
            
            # Run Grounding DINO for tampering detection
            gd_detections, gd_raw = self.run_grounding_dino_detection(file_path)
            
            # Check for foreign objects on track
            for det in gd_detections:
                if det.bounding_box:
                    det.on_track = self.overlaps_track(det.bounding_box, track_mask)
                    if det.on_track:
                        result.foreign_object_on_track = True
            
            # Combine detections
            all_detections = yolo_detections + gd_detections
            
            # Mark detections on track
            for det in all_detections:
                if det.bounding_box:
                    det.on_track = self.overlaps_track(det.bounding_box, track_mask)
            
            result.detections = all_detections
            
            # Build detection summary
            for det in all_detections:
                key = det.detection_type.value
                result.detection_summary[key] = result.detection_summary.get(key, 0) + 1
            
            # Determine tampering status
            if result.person_on_track:
                result.tampering_types.append("Human intrusion on railway track")
            
            if result.foreign_object_on_track:
                result.tampering_types.append("Foreign object placement (stones/debris)")
            
            if not result.tampering_types:
                result.tampering_types.append("Normal track")
            
            # Calculate probabilities and risk level
            result.tampering_probability = self._calculate_tampering_probability(result)
            result.obstruction_probability = self._calculate_obstruction_probability(result)
            result.risk_level = self._calculate_risk_level(result)
            result.confidence = self._calculate_confidence(result)
            
            # Generate alerts
            result.alerts = self._generate_alerts(result)
            result.recommendations = self._generate_recommendations(result)
            
            result.analysis_status = "success"
            
        except Exception as e:
            result.analysis_status = "error"
            result.alerts.append(f"Analysis error: {str(e)}")
        
        return result
    
    def analyze_video(
        self,
        file_path: str,
        annotations: Dict[str, Any] = None,
        metadata: Dict[str, Any] = None,
        sample_rate: int = 30
    ) -> VisualIntegrityResult:
        """
        Analyze a video for track integrity issues.
        
        Args:
            file_path: Path to the video file
            annotations: Optional JSON annotations
            metadata: Additional metadata
            sample_rate: Analyze every Nth frame
        
        Returns:
            VisualIntegrityResult with detections and temporal analysis
        """
        result = VisualIntegrityResult(
            file_path=file_path,
            file_type="video",
            analysis_status="processing",
            timestamp=datetime.utcnow().isoformat() + "Z"
        )
        
        if not CV2_AVAILABLE:
            result.analysis_status = "error"
            result.alerts.append("OpenCV not available")
            return result
        
        try:
            cap = cv2.VideoCapture(file_path)
            result.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            result.fps = cap.get(cv2.CAP_PROP_FPS)
            
            frame_idx = 0
            all_frame_detections = []
            tampering_frames = []
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_idx % sample_rate == 0:
                    # Save temporary frame for analysis
                    temp_path = f"/tmp/frame_{frame_idx}.jpg"
                    cv2.imwrite(temp_path, frame)
                    
                    # Analyze frame
                    frame_result = self.analyze_image(temp_path)
                    
                    # Tag detections with frame number
                    for det in frame_result.detections:
                        det.frame_number = frame_idx
                        det.timestamp = frame_idx / result.fps if result.fps > 0 else 0
                        all_frame_detections.append(det)
                    
                    # Track tampering frames
                    if frame_result.person_on_track or frame_result.foreign_object_on_track:
                        tampering_frames.append({
                            "frame": frame_idx,
                            "timestamp": frame_idx / result.fps if result.fps > 0 else 0,
                            "person_on_track": frame_result.person_on_track,
                            "foreign_object": frame_result.foreign_object_on_track,
                            "tampering_types": frame_result.tampering_types
                        })
                    
                    # Clean up temp file
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                
                frame_idx += 1
            
            cap.release()
            
            result.detections = all_frame_detections
            result.temporal_anomalies = tampering_frames
            
            # Aggregate results
            result.person_on_track = any(t.get("person_on_track") for t in tampering_frames)
            result.foreign_object_on_track = any(t.get("foreign_object") for t in tampering_frames)
            
            # Collect unique tampering types
            all_types = set()
            for t in tampering_frames:
                all_types.update(t.get("tampering_types", []))
            result.tampering_types = list(all_types) if all_types else ["Normal track"]
            
            # Calculate metrics
            result.tampering_probability = self._calculate_tampering_probability(result)
            result.obstruction_probability = self._calculate_obstruction_probability(result)
            result.risk_level = self._calculate_risk_level(result)
            result.confidence = self._calculate_confidence(result)
            
            result.alerts = self._generate_alerts(result)
            result.recommendations = self._generate_recommendations(result)
            
            result.analysis_status = "success"
            
        except Exception as e:
            result.analysis_status = "error"
            result.alerts.append(f"Video analysis error: {str(e)}")
        
        return result
    
    def _calculate_tampering_probability(self, result: VisualIntegrityResult) -> float:
        """Calculate overall tampering probability."""
        prob = 0.0
        if result.person_on_track:
            prob += 0.4
        if result.foreign_object_on_track:
            prob += 0.5
        if result.person_detected and not result.person_on_track:
            prob += 0.1
        return min(prob, 1.0)
    
    def _calculate_obstruction_probability(self, result: VisualIntegrityResult) -> float:
        """Calculate obstruction probability."""
        if result.foreign_object_on_track:
            return 0.9
        if result.person_on_track:
            return 0.7
        return 0.0
    
    def _calculate_risk_level(self, result: VisualIntegrityResult) -> str:
        """Determine risk level based on detections."""
        if result.foreign_object_on_track and result.person_on_track:
            return "critical"
        if result.foreign_object_on_track:
            return "high"
        if result.person_on_track:
            return "high"
        if result.person_detected:
            return "medium"
        return "low"
    
    def _calculate_confidence(self, result: VisualIntegrityResult) -> float:
        """Calculate overall confidence score."""
        if not result.detections:
            return 0.5
        avg_conf = sum(d.confidence for d in result.detections) / len(result.detections)
        return avg_conf
    
    def _generate_alerts(self, result: VisualIntegrityResult) -> List[str]:
        """Generate alerts based on analysis results."""
        alerts = []
        
        if result.person_on_track:
            alerts.append("🚨 CRITICAL: Human intrusion detected on railway track")
        
        if result.foreign_object_on_track:
            alerts.append("🚨 CRITICAL: Foreign object detected on railway track")
        
        if result.person_detected and not result.person_on_track:
            alerts.append("⚠️ WARNING: Person detected near railway track")
        
        if result.file_type == "video" and result.temporal_anomalies:
            alerts.append(f"⚠️ Tampering detected in {len(result.temporal_anomalies)} frames")
        
        return alerts
    
    def _generate_recommendations(self, result: VisualIntegrityResult) -> List[str]:
        """Generate recommendations based on analysis results."""
        recommendations = []
        
        if result.risk_level == "critical":
            recommendations.extend([
                "Immediately halt train operations in this section",
                "Dispatch security and maintenance teams",
                "Notify railway control center",
                "Document incident with additional imagery"
            ])
        elif result.risk_level == "high":
            recommendations.extend([
                "Reduce train speed in affected section",
                "Deploy inspection team",
                "Monitor with additional cameras"
            ])
        elif result.risk_level == "medium":
            recommendations.extend([
                "Increase monitoring frequency",
                "Schedule routine inspection"
            ])
        else:
            recommendations.append("Continue normal monitoring")
        
        return recommendations
    
    def to_dict(self, result: VisualIntegrityResult) -> Dict[str, Any]:
        """Convert result to dictionary for JSON serialization."""
        return {
            "file_path": result.file_path,
            "file_type": result.file_type,
            "analysis_status": result.analysis_status,
            "timestamp": result.timestamp,
            "detections": [
                {
                    "type": d.detection_type.value,
                    "confidence": d.confidence,
                    "bounding_box": d.bounding_box,
                    "label": d.label,
                    "on_track": d.on_track,
                    "frame_number": d.frame_number,
                    "timestamp": d.timestamp
                }
                for d in result.detections
            ],
            "detection_summary": result.detection_summary,
            "tampering": {
                "person_detected": result.person_detected,
                "person_on_track": result.person_on_track,
                "foreign_object_on_track": result.foreign_object_on_track,
                "tampering_types": result.tampering_types,
                "tampering_probability": result.tampering_probability,
                "obstruction_probability": result.obstruction_probability
            },
            "risk_assessment": {
                "risk_level": result.risk_level,
                "confidence": result.confidence
            },
            "video_info": {
                "frame_count": result.frame_count,
                "fps": result.fps,
                "temporal_anomalies": result.temporal_anomalies
            } if result.file_type == "video" else None,
            "alerts": result.alerts,
            "recommendations": result.recommendations
        }


# Module-level functions for backward compatibility
_expert_instance = None

def get_expert() -> VisualIntegrityExpert:
    """Get or create the singleton expert instance."""
    global _expert_instance
    if _expert_instance is None:
        _expert_instance = VisualIntegrityExpert()
    return _expert_instance


def analyze_image(
    file_path: str,
    annotations: Dict[str, Any] = None,
    metadata: Dict[str, Any] = None
) -> VisualIntegrityResult:
    """Analyze an image for track integrity issues."""
    return get_expert().analyze_image(file_path, annotations, metadata)


def analyze_video(
    file_path: str,
    annotations: Dict[str, Any] = None,
    metadata: Dict[str, Any] = None,
    sample_rate: int = 30
) -> VisualIntegrityResult:
    """Analyze a video for track integrity issues."""
    return get_expert().analyze_video(file_path, annotations, metadata, sample_rate)


def analyze_batch(
    file_paths: List[str],
    annotations: Dict[str, Any] = None
) -> List[VisualIntegrityResult]:
    """Analyze multiple image/video files."""
    expert = get_expert()
    results = []
    for fp in file_paths:
        ext = fp.lower().split('.')[-1]
        if ext in ['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'tif']:
            results.append(expert.analyze_image(fp, annotations))
        elif ext in ['mp4', 'avi', 'mov', 'mkv', 'webm']:
            results.append(expert.analyze_video(fp, annotations))
    return results


# ==================== Example Usage ====================

if __name__ == "__main__":
    print("=" * 60)
    print("Visual Track Integrity Expert - Test")
    print("=" * 60)
    
    expert = VisualIntegrityExpert()
    print(f"YOLO Available: {YOLO_AVAILABLE}")
    print(f"Grounding DINO Available: {GROUNDING_DINO_AVAILABLE}")
    print(f"SAM Available: {SAM_AVAILABLE}")
    print(f"Device: {expert.device}")
