import argparse
import json
import time
from pathlib import Path
import sys
import numpy as np

from point_cloud_processor import PointCloudProcessor
from tampering_detector import TamperingDetector
from output_formatter import OutputFormatter, AlertGenerator


class TamperingDetectionPipeline:
    def __init__(self, config_path: str = None):
        config = {}
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
        
        self.processor = PointCloudProcessor(config.get('processing', {}))
        self.detector = TamperingDetector(config.get('detection', {}))
        self.formatter = OutputFormatter()
        self.alert_gen = AlertGenerator()
    
    def process_point_cloud(self, input_file: str, metadata: dict = None) -> dict:
        start_time = time.time()
        
        print(f"\n{'='*70}")
        print(f"Processing: {Path(input_file).name}")
        print(f"{'='*70}\n")
        
        print("Step 1/5: Loading point cloud data...")
        points = self.processor.load_point_cloud(input_file)
        
        print("Step 2/5: Preprocessing point cloud...")
        points = self.processor.preprocess(points)
        
        print("Step 3/5: Segmenting rails and extracting features...")
        segments = self.processor.segment_rails(points)
        features = self.processor.extract_features(points, segments)
        
        print("Step 4/5: Analyzing for tampering...")
        detection_result = self.detector.detect_tampering(features, metadata)
        
        print("Step 5/5: Formatting results...\n")
        processing_time = time.time() - start_time
        
        output = self.formatter.format_detection_output(
            detection_result,
            input_file=input_file,
            processing_time=processing_time
        )
        
        print(self.formatter.format_summary(detection_result))
        
        if detection_result['tampering_detected']:
            alert = self.alert_gen.generate_alert(detection_result)
            output['alert'] = alert
            
            print(f"\n{'!'*70}")
            print(f"🚨 {alert['title']}")
            print(f"{'!'*70}")
            print(f"Message: {alert['message']}")
            print(f"{'!'*70}\n")
        
        print(f"Processing completed in {processing_time:.2f} seconds")
        
        return output
    
    def process_batch(self, input_dir: str, output_dir: str = "./results"):
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        extensions = ['.npy', '.pcd', '.ply', '.las', '.laz']
        files = []
        for ext in extensions:
            files.extend(input_path.glob(f'*{ext}'))
        
        if not files:
            print(f"No point cloud files found in {input_dir}")
            return
        
        print(f"\nFound {len(files)} files to process")
        results = []
        
        for idx, file_path in enumerate(files, 1):
            print(f"\n[{idx}/{len(files)}] Processing {file_path.name}...")
            
            try:
                result = self.process_point_cloud(str(file_path))
                
                output_file = output_path / f"{file_path.stem}_result.json"
                self.formatter.save_json(result, output_file)
                
                results.append({
                    'file': file_path.name,
                    'status': result['result']['status'],
                    'confidence': result['result']['confidence_score'],
                    'severity': result['result']['severity_level']
                })
                
            except Exception as e:
                print(f"Error processing {file_path.name}: {e}")
                results.append({
                    'file': file_path.name,
                    'status': 'ERROR',
                    'error': str(e)
                })
        
        summary = {
            'total_files': len(files),
            'processed': len([r for r in results if r['status'] != 'ERROR']),
            'tampering_detected': len([r for r in results if r.get('status') == 'TAMPERED']),
            'results': results
        }
        
        summary_file = output_path / 'batch_summary.json'
        self.formatter.save_json(summary, summary_file)
        
        print(f"\n{'='*70}")
        print(f"Batch processing complete!")
        print(f"Results saved to: {output_path}")
        print(f"{'='*70}")
    
    def create_sample_data(self, output_dir: str = "data/samples"):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print("Creating sample railway track point cloud data...")
        
        normal_points = self._generate_normal_track()
        normal_file = output_path / "normal_track.npy"
        np.save(normal_file, normal_points)
        print(f"Created normal track sample: {normal_file}")
        
        tampered_points = self._generate_tampered_track()
        tampered_file = output_path / "tampered_track.npy"
        np.save(tampered_file, tampered_points)
        print(f"Created tampered track sample: {tampered_file}")
        
        metadata = {
            "normal_track": {
                "description": "Normal railway track section",
                "gauge": "standard (1.435m)",
                "location": "Test section A"
            },
            "tampered_track": {
                "description": "Tampered railway track section with intentional gauge deviation",
                "gauge": "standard (1.435m) - tampered",
                "location": "Test section B"
            }
        }
        
        metadata_file = output_path / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Created metadata: {metadata_file}\n")
        
        return output_path
    
    def _generate_normal_track(self) -> np.ndarray:
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
        
        points = np.vstack([left_rail, right_rail, sleepers, ground])
        
        return points
    
    def _generate_tampered_track(self) -> np.ndarray:
        np.random.seed(123)
        
        length = 30
        gauge = 1.435
        rail_height = 0.15
        tamper_start = 15
        tamper_end = 25
        
        x = np.linspace(0, length, 300)
        gauge_variation = np.zeros_like(x)
        tamper_mask = (x >= tamper_start) & (x <= tamper_end)
        gauge_variation[tamper_mask] = 0.08 * np.sin((x[tamper_mask] - tamper_start) * np.pi / (tamper_end - tamper_start))
        
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
        
        debris_x = np.random.uniform(tamper_start, tamper_end, 200)
        debris_y = np.random.uniform(-0.5, 0.5, 200)
        debris_z = np.random.uniform(rail_height + 0.05, rail_height + 0.15, 200)
        debris = np.column_stack([debris_x, debris_y, debris_z])
        
        points = np.vstack([left_rail, right_rail, sleepers, ground, debris])
        
        return points


def main():
    parser = argparse.ArgumentParser(description='Railway Track Tampering Detection - LIDAR Module')
    
    parser.add_argument('--input', '-i', type=str, help='Input point cloud file path')
    parser.add_argument('--metadata', '-m', type=str, help='Metadata JSON file path')
    parser.add_argument('--output', '-o', type=str, help='Output JSON file path (default: stdout)')
    parser.add_argument('--batch', '-b', type=str, help='Process all files in directory')
    parser.add_argument('--batch-output', type=str, default='./results', help='Output directory for batch processing')
    parser.add_argument('--config', '-c', type=str, help='Configuration file path')
    parser.add_argument('--demo', action='store_true', help='Run demo with sample data')
    
    args = parser.parse_args()
    
    if args.demo:
        print("Running demo with sample data...")
        pipeline = TamperingDetectionPipeline(args.config)
        sample_dir = pipeline.create_sample_data()
        
        for sample_file in ['normal_track.npy', 'tampered_track.npy']:
            file_path = sample_dir / sample_file
            if file_path.exists():
                result = pipeline.process_point_cloud(str(file_path))
                
                output_file = sample_dir / f"{sample_file.replace('.npy', '_result.json')}"
                pipeline.formatter.save_json(result, output_file)
                print(f"Output saved to: {output_file}")
        
        return
    
    if args.batch:
        pipeline = TamperingDetectionPipeline(args.config)
        pipeline.process_batch(args.batch, args.batch_output)
        return
    
    if not args.input:
        parser.print_help()
        print("\nError: --input required (or use --demo or --batch)")
        sys.exit(1)
    
    metadata = None
    if args.metadata:
        with open(args.metadata, 'r') as f:
            metadata = json.load(f)
    
    pipeline = TamperingDetectionPipeline(args.config)
    result = pipeline.process_point_cloud(args.input, metadata)
    
    if args.output:
        pipeline.formatter.save_json(result, args.output)
    else:
        print(f"\n{'='*70}")
        print("JSON OUTPUT:")
        print(f"{'='*70}")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
