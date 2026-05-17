"""
Main Pipeline - Run Preprocessing Techniques Investigation

Workflow:
1. Capture person's face from camera
2. Apply preprocessing techniques
3. Compare with database reference images
4. Evaluate accuracy scores
5. Record performance metrics
"""

import cv2
import os
import sys
import time
import json
from datetime import datetime
from pathlib import Path
import logging

# Import custom modules
from camera_utils import CameraCapture, capture_test_image
from preprocessing_techniques import PreprocessingPipeline
from evaluation import AccuracyCalculator
from performance_monitor import PerformanceMonitor, Timer
from config import CAPTURED_DIR, REFERENCE_IMAGES_DIR, RESULTS_DIR, DATABASE_DIR, LOGS_DIR

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / 'preprocessing_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class PreprocessingInvestigation:
    """Main pipeline for preprocessing techniques investigation"""
    
    def __init__(self):
        self.pipeline = PreprocessingPipeline()
        self.evaluator = AccuracyCalculator()
        self.monitor = PerformanceMonitor()
        self.results = []
    
    def test_preprocessing_on_condition(self, test_image, reference_images, condition_name, 
                                       preprocessing_technique_name=None):
        """Test a preprocessing technique on a specific condition"""
        try:
            with Timer(f"Preprocessing - {condition_name}"):
                # Select preprocessing technique
                if preprocessing_technique_name == "low_light":
                    processed = self.pipeline.preprocess_low_light(test_image)
                elif preprocessing_technique_name == "motion_blur":
                    processed = self.pipeline.preprocess_motion_blur(test_image)
                elif preprocessing_technique_name == "sensor_noise":
                    processed = self.pipeline.preprocess_sensor_noise(test_image)
                elif preprocessing_technique_name == "extreme_angle":
                    processed = self.pipeline.preprocess_extreme_angle(test_image)
                else:
                    processed = self.pipeline.preprocess_normal(test_image)
                
                # Compare with reference images
                accuracies = []
                for ref_img in reference_images:
                    accuracy = self.evaluator.compare_faces(ref_img, processed, method='ssim')
                    accuracies.append(accuracy)
                
                avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0
                
                return {
                    'condition': condition_name,
                    'preprocessing': preprocessing_technique_name or 'normal',
                    'accuracy': avg_accuracy,
                    'accuracies': accuracies,
                    'num_comparisons': len(accuracies)
                }
        except Exception as e:
            logger.error(f"Error testing preprocessing: {e}")
            return None
    
    def run_full_investigation(self, test_person_id="subject_001"):
        """Run complete preprocessing investigation"""
        try:
            logger.info("=" * 60)
            logger.info("STARTING PREPROCESSING TECHNIQUES INVESTIGATION")
            logger.info("=" * 60)
            
            # Load reference images for test person
            reference_dir = REFERENCE_IMAGES_DIR / test_person_id
            if not reference_dir.exists():
                logger.error(f"Reference images not found for {test_person_id}")
                logger.info("Please run database capture first")
                return False
            
            reference_images = []
            for img_file in reference_dir.glob("*.jpg"):
                try:
                    img = cv2.imread(str(img_file))
                    if img is not None:
                        reference_images.append(img)
                except Exception as e:
                    logger.warning(f"Failed to load reference image {img_file}: {e}")
            
            if not reference_images:
                logger.error("No valid reference images found")
                return False
            
            logger.info(f"Loaded {len(reference_images)} reference images")
            
            # Capture test images for different conditions
            conditions = {
                'normal': "Normal lighting and angle",
                'low_light': "Low lighting conditions (simulate with gamma)",
                'motion_blur': "Motion blur simulation",
                'sensor_noise': "High sensor noise simulation",
                'extreme_angle': "Extreme face angle"
            }
            
            preprocessing_techniques = [
                'normal',
                'low_light',
                'motion_blur',
                'sensor_noise',
                'extreme_angle'
            ]
            
            # Capture test image
            print("\nCapturing test image for analysis...")
            camera = CameraCapture()
            if not camera.initialize_camera():
                logger.error("Failed to initialize camera")
                return False
            
            camera.live_preview(duration_seconds=3)
            ret, test_frame = camera.cap.read()
            camera.release_camera()
            
            if not ret:
                logger.error("Failed to capture test frame")
                return False
            
            logger.info("Test frame captured successfully")
            
            # Test each preprocessing technique
            logger.info("\nTesting preprocessing techniques...")
            logger.info("-" * 60)
            
            for condition_name, condition_desc in conditions.items():
                logger.info(f"\nCondition: {condition_name}")
                logger.info(f"Description: {condition_desc}")
                
                # Test preprocessing technique
                result = self.test_preprocessing_on_condition(
                    test_frame,
                    reference_images,
                    condition_name,
                    condition_name
                )
                
                if result:
                    self.results.append(result)
                    logger.info(f"Accuracy: {result['accuracy']:.2f}%")
                    logger.info(f"Comparisons: {result['num_comparisons']}")
            
            # Save results
            logger.info("\n" + "=" * 60)
            logger.info("INVESTIGATION COMPLETED")
            logger.info("=" * 60)
            
            self._save_results()
            self._print_summary()
            
            return True
        
        except Exception as e:
            logger.error(f"Investigation failed: {e}")
            return False
    
    def _save_results(self):
        """Save investigation results"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = RESULTS_DIR / f"preprocessing_investigation_{timestamp}.json"
            
            with open(filepath, 'w') as f:
                json.dump(self.results, f, indent=2)
            
            logger.info(f"Results saved to {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            return None
    
    def _print_summary(self):
        """Print investigation summary"""
        try:
            if not self.results:
                logger.warning("No results to summarize")
                return
            
            print("\n" + "=" * 60)
            print("PREPROCESSING TECHNIQUES INVESTIGATION SUMMARY")
            print("=" * 60)
            
            print(f"\nTotal Tests: {len(self.results)}")
            print(f"\nResults by Technique:")
            print("-" * 60)
            
            for result in self.results:
                print(f"\nTechnique: {result['preprocessing']}")
                print(f"Condition: {result['condition']}")
                print(f"Accuracy: {result['accuracy']:.2f}%")
                print(f"Comparisons: {result['num_comparisons']}")
            
            # Overall statistics
            accuracies = [r['accuracy'] for r in self.results]
            print("\n" + "-" * 60)
            print("Overall Statistics:")
            print(f"Mean Accuracy: {sum(accuracies)/len(accuracies):.2f}%")
            print(f"Max Accuracy: {max(accuracies):.2f}%")
            print(f"Min Accuracy: {min(accuracies):.2f}%")
            
            # Best performing technique
            best_result = max(self.results, key=lambda x: x['accuracy'])
            print(f"\nBest Performing Technique: {best_result['preprocessing']}")
            print(f"Accuracy: {best_result['accuracy']:.2f}%")
            
            print("=" * 60 + "\n")
        
        except Exception as e:
            logger.error(f"Failed to print summary: {e}")


def setup_database(person_id="subject_001", num_samples=5):
    """Capture reference images for database"""
    try:
        logger.info(f"Setting up database for {person_id}")
        
        person_dir = REFERENCE_IMAGES_DIR / person_id
        person_dir.mkdir(parents=True, exist_ok=True)
        
        camera = CameraCapture()
        if not camera.initialize_camera():
            logger.error("Failed to initialize camera")
            return False
        
        print(f"\nCapturing {num_samples} reference images for {person_id}")
        print("Position your face in front of the camera")
        print("Frames will be captured automatically\n")
        
        captured_paths = camera.capture_multiple_frames(
            count=num_samples,
            person_name=person_id
        )
        
        camera.release_camera()
        
        # Move captured images to reference directory
        captured_person_dir = CAPTURED_DIR / person_id
        if captured_person_dir.exists():
            for img_file in captured_person_dir.glob("*.jpg"):
                import shutil
                shutil.move(str(img_file), str(person_dir / img_file.name))
        
        logger.info(f"Database setup completed for {person_id}")
        print(f"\n✓ Database setup completed: {len(list(person_dir.glob('*.jpg')))} images captured\n")
        
        return True
    except Exception as e:
        logger.error(f"Database setup failed: {e}")
        return False


def main():
    """Main entry point"""
    print("\n" + "=" * 60)
    print("FACIAL RECOGNITION - PREPROCESSING TECHNIQUES INVESTIGATION")
    print("=" * 60)
    
    # Check if database exists
    person_id = "subject_001"
    ref_dir = REFERENCE_IMAGES_DIR / person_id
    
    if not ref_dir.exists() or not list(ref_dir.glob("*.jpg")):
        print("\nNo database found. Setting up reference images...")
        if not setup_database(person_id, num_samples=5):
            print("Failed to setup database")
            return
    
    # Run investigation
    investigation = PreprocessingInvestigation()
    investigation.run_full_investigation(person_id)


if __name__ == "__main__":
    main()
