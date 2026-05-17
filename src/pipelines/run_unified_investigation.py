"""
Unified Investigation Pipeline - Run Both Preprocessing & Models in Sequence

Workflow:
1. Capture your face once from camera
2. Run preprocessing pipeline (BEFORE/AFTER comparison)
3. Run models pipeline (BEFORE/AFTER comparison)
4. Combine results into unified JSON with both sections
5. Organize files into separate folders (preprocessing/ and models/)
"""

import cv2
import os
import sys
import numpy as np
from datetime import datetime
from pathlib import Path
import logging
import json
import shutil

# Add parent directories to path BEFORE imports
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import custom modules
from core.camera_utils import CameraCapture
from core.colors import Colors, print_header, print_subheader, print_success, print_error, print_info
from core.config import CAPTURED_DIR, LOGS_DIR

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / 'unified_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class UnifiedInvestigation:
    """Unified pipeline that runs both preprocessing and models investigations"""

    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.investigation_folder = CAPTURED_DIR / self.timestamp
        self.preprocessing_folder = self.investigation_folder / "preprocessing"
        self.models_folder = self.investigation_folder / "models"
        
        # Create folder structure
        self.investigation_folder.mkdir(parents=True, exist_ok=True)
        self.preprocessing_folder.mkdir(parents=True, exist_ok=True)
        self.models_folder.mkdir(parents=True, exist_ok=True)
        
        self.captured_face = None
        self.preprocessing_results = None
        self.models_results = None

    def capture_your_face(self):
        """Capture your face from camera once - shared by both pipelines"""
        try:
            logger.info("\n" + "="*70)
            logger.info("CAPTURING YOUR FACE - SHARED FOR BOTH INVESTIGATIONS")
            logger.info("="*70)
            logger.info("Position your face in front of the camera...")
            logger.info("Press SPACE to capture, ESC to cancel")

            camera = CameraCapture()
            if not camera.initialize_camera():
                logger.error("Failed to initialize camera")
                return None

            captured_frame = None

            # Show live camera feed
            while True:
                ret, frame = camera.cap.read()
                if not ret:
                    logger.error("Failed to read frame")
                    break

                # Display instructions
                cv2.putText(frame, "Press SPACE to capture, ESC to cancel",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, "Will be used for BOTH pipelines",
                            (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

                cv2.imshow("Camera - Capture Your Face (Shared)", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == 32:  # SPACE
                    captured_frame = frame.copy()
                    logger.info("[OK] Face captured!")
                    break
                elif key == 27:  # ESC
                    logger.info("Capture cancelled")
                    break

            cv2.destroyAllWindows()
            camera.release_camera()

            return captured_frame
        except Exception as e:
            logger.error(f"Error capturing face: {e}")
            return None

    def run_preprocessing_investigation(self):
        """Run preprocessing pipeline on captured face"""
        try:
            print_subheader("Running PREPROCESSING Investigation")
            
            from pipelines.run_preprocessing_with_capture import PreprocessingWithCapture
            
            # Create temporary investigation object
            investigation = PreprocessingWithCapture()
            
            # Override timestamp and captured dir to use our unified folder
            investigation.timestamp = self.timestamp
            
            # Temporarily override CAPTURED_DIR for preprocessing to use our preprocessing folder
            original_run_investigation = investigation.run_investigation
            
            # Run the preprocessing investigation
            success = investigation.run_preprocessing_on_captured_face(
                self.captured_face,
                self.preprocessing_folder
            )
            
            if success:
                print_success("Preprocessing investigation completed")
                
                # Load preprocessing results
                preprocessing_json = self.preprocessing_folder / "comparison_results.json"
                if preprocessing_json.exists():
                    with open(preprocessing_json, 'r') as f:
                        self.preprocessing_results = json.load(f)
                    logger.info("[OK] Preprocessing results loaded")
                    return True
            else:
                print_error("Preprocessing investigation failed")
                return False
                
        except Exception as e:
            print_error(f"Error running preprocessing investigation: {e}")
            logger.error(f"Preprocessing error: {e}")
            import traceback
            traceback.print_exc()
            return False

    def run_models_investigation(self):
        """Run models pipeline on captured face"""
        try:
            print_subheader("Running MODELS Investigation")
            
            from pipelines.run_recommended_models import RecommendedModelsWithCapture
            
            # Create temporary investigation object
            investigation = RecommendedModelsWithCapture()
            
            # Override timestamp to use our unified timestamp
            investigation.timestamp = self.timestamp
            
            # Run the models investigation
            success = investigation.run_models_on_captured_face(
                self.captured_face,
                self.models_folder
            )
            
            if success:
                print_success("Models investigation completed")
                
                # Load models results
                models_json = self.models_folder / "comparison_results.json"
                if models_json.exists():
                    with open(models_json, 'r') as f:
                        self.models_results = json.load(f)
                    logger.info("[OK] Models results loaded")
                    return True
            else:
                print_error("Models investigation failed")
                return False
                
        except Exception as e:
            print_error(f"Error running models investigation: {e}")
            logger.error(f"Models error: {e}")
            import traceback
            traceback.print_exc()
            return False

    def combine_results(self):
        """Combine preprocessing and models results into unified JSON"""
        try:
            combined = {
                'timestamp': self.timestamp,
                'folder': str(self.investigation_folder),
                'investigation_type': 'UNIFIED_PREPROCESSING_AND_MODELS',
                
                'preprocessing': self.preprocessing_results,
                'models': self.models_results,
                
                'folder_structure': {
                    'preprocessing_folder': 'preprocessing/',
                    'models_folder': 'models/',
                    'combined_results': 'combined_results.json'
                }
            }
            
            results_file = self.investigation_folder / "combined_results.json"
            with open(results_file, 'w') as f:
                json.dump(combined, f, indent=2)
                f.flush()
            
            logger.info(f"[OK] Combined results saved: {results_file}")
            return results_file
            
        except Exception as e:
            logger.error(f"Failed to combine results: {e}")
            import traceback
            traceback.print_exc()
            return None

    def run_unified_investigation(self):
        """Run complete unified investigation"""
        try:
            # Print header
            print_header("UNIFIED INVESTIGATION - PREPROCESSING & MODELS")
            print_info("Single face capture → Run both pipelines → Combined JSON")

            # Step 1: Capture face once
            print_subheader("Step 1: Capturing your face from camera (ONCE)")
            self.captured_face = self.capture_your_face()
            if self.captured_face is None:
                print_error("Failed to capture face")
                return False
            print_success(f"Face captured - Shape: {self.captured_face.shape}")
            print_success(f"Investigation folder: {self.investigation_folder}")

            # Step 2: Run preprocessing investigation
            print_subheader("Step 2: PREPROCESSING INVESTIGATION")
            if not self.run_preprocessing_investigation():
                print_error("Preprocessing investigation failed")
                return False

            # Step 3: Run models investigation
            print_subheader("Step 3: MODELS INVESTIGATION")
            if not self.run_models_investigation():
                print_error("Models investigation failed")
                return False

            # Step 4: Combine results
            print_subheader("Step 4: Combining results into unified JSON")
            combined_file = self.combine_results()
            if combined_file is None:
                print_error("Failed to combine results")
                return False
            print_success("Results combined successfully")

            # Final summary
            print_header("UNIFIED INVESTIGATION COMPLETED SUCCESSFULLY!")
            print_success(f"📁 Investigation folder: {self.investigation_folder}")
            print_success(f"📁 Preprocessing results: {self.investigation_folder}/preprocessing/")
            print_success(f"📁 Models results: {self.investigation_folder}/models/")
            print_success(f"📄 Combined results: {self.investigation_folder}/combined_results.json")
            print_info(f"\nFolder Structure:")
            print_info(f"  {self.timestamp}/")
            print_info(f"  ├── preprocessing/")
            print_info(f"  │   ├── 01_before_preprocessing.jpg")
            print_info(f"  │   ├── 02_after_preprocessing.jpg")
            print_info(f"  │   └── comparison_results.json")
            print_info(f"  ├── models/")
            print_info(f"  │   ├── 01_before_preprocessing.jpg")
            print_info(f"  │   ├── 02_after_preprocessing.jpg")
            print_info(f"  │   └── comparison_results.json")
            print_info(f"  └── combined_results.json")

            return True

        except Exception as e:
            print_error(f"Investigation failed: {e}")
            logger.error(f"Investigation failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Main entry point"""
    try:
        investigation = UnifiedInvestigation()
        success = investigation.run_unified_investigation()

        if success:
            print("\n[OK] Unified investigation completed successfully!")
        else:
            print("\n[FAIL] Unified investigation failed. Check logs for details.")

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
