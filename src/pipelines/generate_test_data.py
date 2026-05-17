"""
Test Data Generator - Create synthetic face-like images for testing
Bypasses the need for live camera capture
"""

import cv2
import numpy as np
from pathlib import Path
import logging
from datetime import datetime

from config import REFERENCE_IMAGES_DIR, CAPTURED_DIR, LOGS_DIR

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / 'test_data_generator.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def create_synthetic_face_image(width=640, height=480, variation=0):
    """Generate a synthetic face-like image using simple numpy operations"""
    try:
        # Create base image with skin tone
        image = np.ones((height, width, 3), dtype=np.uint8)
        
        # Skin tone (BGR) 
        skin_base = [180, 140, 120]
        for c in range(3):
            image[:, :, c] = np.clip(skin_base[c] + variation, 0, 255)
        
        # Add simple gradient for face contour using numpy
        y_coords, x_coords = np.ogrid[:height, :width]
        distance = np.sqrt((x_coords - width/2)**2 + (y_coords - height/2)**2)
        
        # Face area - lighten
        face_mask = distance < width * 0.3
        for c in range(3):
            image[face_mask, c] = np.clip(image[face_mask, c] + 20, 0, 255)
        
        # Edge - darken
        edge_mask = (distance >= width * 0.3) & (distance < width * 0.35)
        for c in range(3):
            image[edge_mask, c] = np.clip(image[edge_mask, c] - 20, 0, 255)
        
        # Add simple noise
        noise = np.random.normal(0, variation + 5, image.shape)
        image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        
        # Draw simple geometric features (eyes, nose, mouth)
        # Left eye center
        eye_y = int(height * 0.35)
        left_eye_x = int(width * 0.35)
        
        # Simple eye as dark circle using numpy
        eye_dist = np.sqrt((x_coords - left_eye_x)**2 + (y_coords - eye_y)**2)
        eye_mask = eye_dist < 15
        image[eye_mask] = [50, 50, 50]
        
        # Right eye
        right_eye_x = int(width * 0.65)
        eye_dist_r = np.sqrt((x_coords - right_eye_x)**2 + (y_coords - eye_y)**2)
        eye_mask_r = eye_dist_r < 15
        image[eye_mask_r] = [50, 50, 50]
        
        # Nose as simple vertical line
        nose_x = int(width * 0.5)
        nose_start_y = int(height * 0.45)
        nose_end_y = int(height * 0.55)
        image[nose_start_y:nose_end_y, nose_x-1:nose_x+2] = [100, 80, 60]
        
        # Mouth as simple horizontal line
        mouth_y = int(height * 0.65)
        mouth_left_x = int(width * 0.35)
        mouth_right_x = int(width * 0.65)
        image[mouth_y:mouth_y+3, mouth_left_x:mouth_right_x] = [100, 50, 50]
        
        return image.astype(np.uint8)
    except Exception as e:
        logger.error(f"Failed to create synthetic face image: {e}")
        return None


def generate_database_images(person_id="001", num_images=5):
    """Generate test database images for a person"""
    try:
        person_dir = REFERENCE_IMAGES_DIR / person_id
        person_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Generating {num_images} test images for {person_id}...")
        print(f"\nGenerating test database images for {person_id}...")
        
        for i in range(num_images):
            # Create synthetic face with slight variations
            variation = np.random.randint(0, 30)
            face_image = create_synthetic_face_image(variation=variation)
            
            if face_image is not None:
                # Save image
                filename = f"frame_{i}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                filepath = person_dir / filename
                
                cv2.imwrite(str(filepath), face_image)
                print(f"  ✓ Generated: {filename}")
                logger.info(f"Generated test image: {filepath}")
            else:
                logger.error(f"Failed to generate image {i}")
        
        print(f"✓ Database setup completed: {len(list(person_dir.glob('*.jpg')))} images generated\n")
        logger.info(f"Database generation completed for {person_id}")
        return True
    except Exception as e:
        logger.error(f"Failed to generate database images: {e}")
        return False


def generate_test_images(person_id="001", num_images=3, condition="normal"):
    """Generate test images for investigation"""
    try:
        test_dir = CAPTURED_DIR / person_id
        test_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Generating {num_images} test images for condition: {condition}...")
        
        for i in range(num_images):
            variation = np.random.randint(10, 50)
            face_image = create_synthetic_face_image(variation=variation)
            
            if face_image is not None:
                # Apply condition-specific transformations
                if condition == "low_light":
                    face_image = cv2.convertScaleAbs(face_image, alpha=0.6, beta=0)
                elif condition == "motion_blur":
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
                    face_image = cv2.morphologyEx(face_image, cv2.MORPH_OPEN, kernel)
                elif condition == "sensor_noise":
                    noise = np.random.normal(0, 50, face_image.shape)
                    face_image = np.clip(face_image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
                elif condition == "extreme_angle":
                    M = cv2.getRotationMatrix2D((320, 240), 25, 1.0)
                    face_image = cv2.warpAffine(face_image, M, (640, 480))
                
                # Save image
                filename = f"test_{condition}_{i}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                filepath = test_dir / filename
                cv2.imwrite(str(filepath), face_image)
                logger.info(f"Generated test image: {filepath}")
        
        return True
    except Exception as e:
        logger.error(f"Failed to generate test images: {e}")
        return False


def setup_complete_test_environment(person_id="001"):
    """Setup complete test environment with all needed data"""
    try:
        print("\n" + "="*80)
        print("GENERATING COMPLETE TEST ENVIRONMENT")
        print("="*80)
        
        # Generate reference database
        print("\n1. Generating reference database images...")
        if not generate_database_images(person_id, num_images=5):
            logger.error("Failed to generate database images")
            return False
        
        # Generate test images for each condition
        conditions = ['normal', 'low_light', 'motion_blur', 'sensor_noise', 'extreme_angle']
        
        print("\n2. Generating test images for different conditions...")
        for condition in conditions:
            print(f"\n   Condition: {condition}")
            if not generate_test_images(person_id, num_images=2, condition=condition):
                logger.error(f"Failed to generate test images for {condition}")
        
        print("\n" + "="*80)
        print("✓ TEST ENVIRONMENT SETUP COMPLETE")
        print("="*80)
        print(f"\nDatabase: {REFERENCE_IMAGES_DIR / person_id}")
        print(f"Test Images: {CAPTURED_DIR / person_id}")
        print("\nYou can now run the investigation pipelines!")
        print("="*80 + "\n")
        
        return True
    except Exception as e:
        logger.error(f"Failed to setup test environment: {e}")
        return False


def main():
    """Main entry point"""
    import sys
    
    if len(sys.argv) > 1:
        person_id = sys.argv[1]
    else:
        person_id = "001"
    
    setup_complete_test_environment(person_id)


if __name__ == "__main__":
    main()
