"""
Preprocessing Techniques for Facial Recognition
Implements techniques for different challenging conditions
"""

import cv2
import numpy as np
from scipy import ndimage
from scipy.ndimage import gaussian_filter
import logging
from pathlib import Path
import sys

# Add parent to path for imports
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from core.config import LOGS_DIR

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / 'preprocessing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class PreprocessingTechniques:
    """Collection of preprocessing techniques for facial recognition"""
    
    @staticmethod
    def histogram_equalization(image):
        """Histogram Equalization - for low light conditions"""
        try:
            if len(image.shape) == 3:
                # Convert to HSV, equalize V channel, convert back
                hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                h, s, v = cv2.split(hsv)
                v = cv2.equalizeHist(v)
                hsv_equalized = cv2.merge([h, s, v])
                result = cv2.cvtColor(hsv_equalized, cv2.COLOR_HSV2BGR)
            else:
                result = cv2.equalizeHist(image)
            
            logger.debug("Histogram equalization applied")
            return result
        except Exception as e:
            logger.error(f"Histogram equalization failed: {e}")
            return image
    
    @staticmethod
    def adaptive_histogram_equalization(image, clip_limit=2.0, tile_size=8):
        """Adaptive Histogram Equalization (CLAHE) - enhanced low light handling"""
        try:
            if len(image.shape) == 3:
                lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                
                clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
                l = clahe.apply(l)
                
                lab_clahe = cv2.merge([l, a, b])
                result = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
            else:
                clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
                result = clahe.apply(image)
            
            logger.debug("Adaptive histogram equalization applied")
            return result
        except Exception as e:
            logger.error(f"Adaptive histogram equalization failed: {e}")
            return image
    
    @staticmethod
    def gamma_correction(image, gamma=0.5):
        """Gamma Correction - adjust brightness for low light conditions"""
        try:
            # Build a lookup table mapping pixel values [0, 255] to adjusted gamma values
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
            
            result = cv2.LUT(image, table)
            logger.debug(f"Gamma correction applied (gamma={gamma})")
            return result
        except Exception as e:
            logger.error(f"Gamma correction failed: {e}")
            return image
    
    @staticmethod
    def laplacian_sharpening(image, intensity=1.0):
        """Laplacian Sharpening - sharpen blurred faces"""
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            sharpened = gray - intensity * laplacian
            sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
            
            # Convert back to BGR if needed
            if len(image.shape) == 3:
                result = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
            else:
                result = sharpened
            
            logger.debug(f"Laplacian sharpening applied (intensity={intensity})")
            return result
        except Exception as e:
            logger.error(f"Laplacian sharpening failed: {e}")
            return image
    
    @staticmethod
    def wiener_filter(image, noise_variance=None):
        """Wiener Filter - reduce sensor noise"""
        try:
            if len(image.shape) == 3:
                # Apply to each channel
                channels = cv2.split(image)
                filtered_channels = []
                for channel in channels:
                    filtered_channel = cv2.medianBlur(channel, 5)
                    filtered_channels.append(filtered_channel)
                result = cv2.merge(filtered_channels)
            else:
                result = cv2.medianBlur(image, 5)
            
            logger.debug("Wiener filter applied")
            return result
        except Exception as e:
            logger.error(f"Wiener filter failed: {e}")
            return image
    
    @staticmethod
    def gaussian_blur(image, kernel_size=5, sigma=1.0):
        """Gaussian Blur - smooth image"""
        try:
            result = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
            logger.debug(f"Gaussian blur applied (kernel_size={kernel_size}, sigma={sigma})")
            return result
        except Exception as e:
            logger.error(f"Gaussian blur failed: {e}")
            return image
    
    @staticmethod
    def non_local_means_denoising(image):
        """Non-local Means Denoising - advanced noise reduction"""
        try:
            if len(image.shape) == 3:
                result = cv2.fastNlMeansDenoisingColored(
                    image,
                    h=10,
                    hForColorComponents=10,
                    templateWindowSize=7,
                    searchWindowSize=21
                )
            else:
                result = cv2.fastNlMeansDenoising(
                    image,
                    h=10,
                    templateWindowSize=7,
                    searchWindowSize=21
                )
            
            logger.debug("Non-local means denoising applied")
            return result
        except Exception as e:
            logger.error(f"Non-local means denoising failed: {e}")
            return image
    
    @staticmethod
    def motion_blur_kernel(size=15, angle=45):
        """Create motion blur kernel"""
        kernel = cv2.getRotationMatrix2D((size / 2, size / 2), angle, 1.0)
        kernel = cv2.warpAffine(np.eye(size), kernel, (size, size))
        kernel = kernel / kernel.sum()
        return kernel
    
    @staticmethod
    def motion_blur_reduction(image, kernel_size=15):
        """Motion Blur Reduction - inverse motion blur filter"""
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Apply unsharp mask to reduce motion blur
            gaussian = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
            sharpened = cv2.addWeighted(gray, 2.0, gaussian, -1.0, 0)
            sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
            
            if len(image.shape) == 3:
                result = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
            else:
                result = sharpened
            
            logger.debug("Motion blur reduction applied")
            return result
        except Exception as e:
            logger.error(f"Motion blur reduction failed: {e}")
            return image
    
    @staticmethod
    def affine_transformation(image, angle=20, scale=1.0):
        """Affine Transformation - handle extreme angles"""
        try:
            rows, cols = image.shape[:2]
            
            # Get rotation matrix
            center = (cols / 2, rows / 2)
            M = cv2.getRotationMatrix2D(center, angle, scale)
            
            result = cv2.warpAffine(image, M, (cols, rows))
            logger.debug(f"Affine transformation applied (angle={angle}, scale={scale})")
            return result
        except Exception as e:
            logger.error(f"Affine transformation failed: {e}")
            return image
    
    @staticmethod
    def perspective_transformation(image, shift_percent=0.1):
        """3D-like perspective transformation - handle extreme angles"""
        try:
            rows, cols = image.shape[:2]
            
            pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
            pts2 = np.float32([
                [50 + shift_percent * cols, 50],
                [200 + shift_percent * cols, 50],
                [50, 200 + shift_percent * rows]
            ])
            
            M = cv2.getAffineTransform(pts1, pts2)
            result = cv2.warpAffine(image, M, (cols, rows))
            
            logger.debug(f"Perspective transformation applied (shift={shift_percent})")
            return result
        except Exception as e:
            logger.error(f"Perspective transformation failed: {e}")
            return image
    
    @staticmethod
    def contrast_enhancement(image, alpha=1.5, beta=0):
        """Enhance contrast - general improvement"""
        try:
            result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
            logger.debug(f"Contrast enhancement applied (alpha={alpha})")
            return result
        except Exception as e:
            logger.error(f"Contrast enhancement failed: {e}")
            return image
    
    @staticmethod
    def normalize_image(image):
        """Normalize image values to [0, 1] range"""
        try:
            if image.dtype == np.uint8:
                result = image.astype(np.float32) / 255.0
            else:
                result = cv2.normalize(image, None, 0, 1, cv2.NORM_MINMAX)
            
            logger.debug("Image normalization applied")
            return result
        except Exception as e:
            logger.error(f"Image normalization failed: {e}")
            return image


class PreprocessingPipeline:
    """Composite preprocessing for different conditions"""
    
    def __init__(self):
        self.techniques = PreprocessingTechniques()
    
    def preprocess_low_light(self, image):
        """Pipeline for low light conditions"""
        try:
            # Histogram equalization or CLAHE
            image = self.techniques.adaptive_histogram_equalization(image)
            
            # Optional gamma correction
            image = self.techniques.gamma_correction(image, gamma=0.7)
            
            # Contrast enhancement
            image = self.techniques.contrast_enhancement(image, alpha=1.3)
            
            logger.info("Low light preprocessing completed")
            return image
        except Exception as e:
            logger.error(f"Low light preprocessing failed: {e}")
            return image
    
    def preprocess_motion_blur(self, image):
        """Pipeline for motion blur"""
        try:
            image = self.techniques.motion_blur_reduction(image)
            image = self.techniques.laplacian_sharpening(image, intensity=0.8)
            image = self.techniques.contrast_enhancement(image, alpha=1.2)
            
            logger.info("Motion blur preprocessing completed")
            return image
        except Exception as e:
            logger.error(f"Motion blur preprocessing failed: {e}")
            return image
    
    def preprocess_sensor_noise(self, image):
        """Pipeline for sensor noise"""
        try:
            image = self.techniques.non_local_means_denoising(image)
            image = self.techniques.gaussian_blur(image, kernel_size=3, sigma=0.5)
            
            logger.info("Sensor noise preprocessing completed")
            return image
        except Exception as e:
            logger.error(f"Sensor noise preprocessing failed: {e}")
            return image
    
    def preprocess_extreme_angle(self, image):
        """Pipeline for extreme angles"""
        try:
            image = self.techniques.perspective_transformation(image, shift_percent=0.05)
            image = self.techniques.contrast_enhancement(image, alpha=1.2)
            image = self.techniques.laplacian_sharpening(image, intensity=0.5)
            
            logger.info("Extreme angle preprocessing completed")
            return image
        except Exception as e:
            logger.error(f"Extreme angle preprocessing failed: {e}")
            return image
    
    def preprocess_normal(self, image):
        """
        REVISED: Fixed SSIM Drop Logic
        Replaces Global Equalization with Adaptive CLAHE and adds edge-preserving smoothing.
        """
        try:
            # 1. FIX: Use CLAHE (Adaptive Equalization) instead of Global HE.
            # This processes the image in tiles, preserving local facial structure.
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            
            if len(image.shape) == 3:
                # Process in LAB color space to equalize only Lightness
                lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                l = clahe.apply(l)
                image = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
            else:
                image = clahe.apply(image)

            # 2. FIX: Add Bilateral Filter
            # Removes camera 'grain' (which hurts SSIM) while keeping eyes/nose sharp.
            image = cv2.bilateralFilter(image, d=5, sigmaColor=75, sigmaSpace=75)

            # 3. FIX: Milder Contrast Enhancement
            # Alpha 1.05 provides enhancement without the pixel clipping of 1.1.
            image = self.techniques.contrast_enhancement(image, alpha=1.05)
            
            logger.info("SSIM-Optimized Normal preprocessing completed")
            return image
        except Exception as e:
            logger.error(f"Normal preprocessing failed: {e}")
            return image


if __name__ == "__main__":
    # Test preprocessing
    pipeline = PreprocessingPipeline()
    print("Preprocessing techniques module loaded successfully")
