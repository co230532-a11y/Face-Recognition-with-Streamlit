"""
Evaluation Module - Calculate accuracy and match scores
"""

import cv2
import numpy as np
from pathlib import Path
import json
import logging
from datetime import datetime
import sys

# Add parent to pathF
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from core.config import LOGS_DIR, RESULTS_DIR


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / 'evaluation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class EvaluationMetrics:
    """Calculate evaluation metrics for face recognition"""
    
    @staticmethod
    def euclidean_distance(vector1, vector2):
        """Calculate Euclidean distance between two vectors"""
        return np.sqrt(np.sum((vector1 - vector2) ** 2))
    
    @staticmethod
    def cosine_similarity(vector1, vector2):
        """Calculate cosine similarity between two vectors"""
        dot_product = np.dot(vector1, vector2)
        norm1 = np.linalg.norm(vector1)
        norm2 = np.linalg.norm(vector2)
        
        if norm1 == 0 or norm2 == 0:
            return 0
        
        return dot_product / (norm1 * norm2)
    
    @staticmethod
    def histogram_intersection(hist1, hist2):
        """Calculate histogram intersection similarity"""
        intersection = 0
        for i in range(len(hist1)):
            intersection += min(hist1[i], hist2[i])
        
        return intersection / (1 + len(hist1))
    
    @staticmethod
    def structural_similarity(img1, img2):
        """Calculate Structural Similarity Index (SSIM)"""
        try:
            # Ensure images are same size
            if img1.shape != img2.shape:
                img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
            
            # Convert to grayscale if needed
            if len(img1.shape) == 3:
                gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            else:
                gray1 = img1
            
            if len(img2.shape) == 3:
                gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            else:
                gray2 = img2
            
            # Simple SSIM calculation
            gray1 = gray1.astype(np.float32) / 255.0
            gray2 = gray2.astype(np.float32) / 255.0
            
            mse = np.mean((gray1 - gray2) ** 2)
            max_pixel = 1.0
            psnr = 20 * np.log10(max_pixel / np.sqrt(mse)) if mse > 0 else 100
            
            # Convert to 0-100 similarity score
            similarity = min(100, max(0, psnr / 2))
            
            return similarity
        except Exception as e:
            logger.error(f"SSIM calculation failed: {e}")
            return 0
    
    @staticmethod
    def feature_vector_extraction(image):
        """Extract a simple feature vector from image"""
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Normalize
            gray = gray.astype(np.float32) / 255.0
            
            # Simple feature extraction: histogram
            hist = cv2.calcHist([gray], [0], None, [256], [0, 1])
            hist = hist.flatten() / hist.sum()
            
            return hist
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return None


class AccuracyCalculator:
    """Calculate accuracy scores for recognition results"""
    
    def __init__(self):
        self.results = []
    
    def compare_faces(self, reference_image, test_image, method='ssim', distance_threshold=0.6):
        """Compare two face images and return similarity score (0-100%)"""
        try:
            metrics = EvaluationMetrics()
            
            if method == 'ssim':
                score = metrics.structural_similarity(reference_image, test_image)
            
            elif method == 'histogram':
                hist_ref = metrics.feature_vector_extraction(reference_image)
                hist_test = metrics.feature_vector_extraction(test_image)
                
                if hist_ref is not None and hist_test is not None:
                    similarity = metrics.cosine_similarity(hist_ref, hist_test)
                    score = (similarity + 1) / 2 * 100  # Convert [-1, 1] to [0, 100]
                else:
                    score = 0
            
            elif method == 'euclidean':
                vec_ref = metrics.feature_vector_extraction(reference_image)
                vec_test = metrics.feature_vector_extraction(test_image)
                
                if vec_ref is not None and vec_test is not None:
                    distance = metrics.euclidean_distance(vec_ref, vec_test)
                    score = max(0, 100 - distance * 100)
                else:
                    score = 0
            
            else:
                score = metrics.structural_similarity(reference_image, test_image)
            
            score = max(0, min(100, score))  # Clamp to [0, 100]
            
            logger.debug(f"Face comparison ({method}): {score:.2f}%")
            return score
        
        except Exception as e:
            logger.error(f"Face comparison failed: {e}")
            return 0
    
    def record_result(self, person_id, condition, preprocessing_technique, model_name, 
                     accuracy_score, processing_time, memory_usage):
        """Record a test result"""
        try:
            result = {
                'timestamp': datetime.now().isoformat(),
                'person_id': person_id,
                'condition': condition,
                'preprocessing': preprocessing_technique,
                'model': model_name,
                'accuracy': round(accuracy_score, 2),
                'processing_time_ms': round(processing_time * 1000, 2),
                'memory_usage_mb': round(memory_usage, 2)
            }
            
            self.results.append(result)
            logger.info(f"Result recorded: {person_id}, {condition}, {accuracy_score:.2f}%")
            
            return result
        except Exception as e:
            logger.error(f"Failed to record result: {e}")
            return None
    
    def save_results(self, filename="evaluation_results.json"):
        """Save results to file"""
        try:
            filepath = RESULTS_DIR / filename
            
            with open(filepath, 'w') as f:
                json.dump(self.results, f, indent=2)
            
            logger.info(f"Results saved to {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            return None
    
    def load_results(self, filename="evaluation_results.json"):
        """Load results from file"""
        try:
            filepath = RESULTS_DIR / filename
            
            if not filepath.exists():
                logger.warning(f"Results file not found: {filepath}")
                return []
            
            with open(filepath, 'r') as f:
                self.results = json.load(f)
            
            logger.info(f"Results loaded from {filepath}")
            return self.results
        except Exception as e:
            logger.error(f"Failed to load results: {e}")
            return []
    
    def get_statistics(self):
        """Calculate statistics from results"""
        try:
            if not self.results:
                logger.warning("No results to analyze")
                return {}
            
            accuracies = np.array([r['accuracy'] for r in self.results])
            
            stats = {
                'total_tests': len(self.results),
                'mean_accuracy': float(np.mean(accuracies)),
                'std_accuracy': float(np.std(accuracies)),
                'min_accuracy': float(np.min(accuracies)),
                'max_accuracy': float(np.max(accuracies)),
                'median_accuracy': float(np.median(accuracies)),
                'mean_processing_time_ms': float(np.mean([r['processing_time_ms'] for r in self.results])),
                'mean_memory_usage_mb': float(np.mean([r['memory_usage_mb'] for r in self.results]))
            }
            
            logger.info(f"Statistics calculated: Mean Accuracy = {stats['mean_accuracy']:.2f}%")
            return stats
        except Exception as e:
            logger.error(f"Statistics calculation failed: {e}")
            return {}
    
    def get_condition_statistics(self):
        """Get statistics grouped by condition"""
        try:
            stats_by_condition = {}
            
            for result in self.results:
                condition = result['condition']
                if condition not in stats_by_condition:
                    stats_by_condition[condition] = []
                
                stats_by_condition[condition].append(result['accuracy'])
            
            condition_stats = {}
            for condition, accuracies in stats_by_condition.items():
                acc_array = np.array(accuracies)
                condition_stats[condition] = {
                    'count': len(accuracies),
                    'mean': float(np.mean(acc_array)),
                    'std': float(np.std(acc_array)),
                    'min': float(np.min(acc_array)),
                    'max': float(np.max(acc_array))
                }
            
            logger.info(f"Condition statistics calculated for {len(condition_stats)} conditions")
            return condition_stats
        except Exception as e:
            logger.error(f"Condition statistics calculation failed: {e}")
            return {}
    
    def get_model_comparison(self):
        """Compare accuracy across models"""
        try:
            model_stats = {}
            
            for result in self.results:
                model = result['model']
                if model not in model_stats:
                    model_stats[model] = []
                
                model_stats[model].append(result['accuracy'])
            
            comparison = {}
            for model, accuracies in model_stats.items():
                acc_array = np.array(accuracies)
                comparison[model] = {
                    'count': len(accuracies),
                    'mean': float(np.mean(acc_array)),
                    'std': float(np.std(acc_array)),
                    'min': float(np.min(acc_array)),
                    'max': float(np.max(acc_array))
                }
            
            logger.info(f"Model comparison calculated for {len(comparison)} models")
            return comparison
        except Exception as e:
            logger.error(f"Model comparison failed: {e}")
            return {}
    
    def cosine_similarity(self, vector1, vector2):
        """Calculate cosine similarity between two vectors"""
        return EvaluationMetrics.cosine_similarity(vector1, vector2)
    
    def euclidean_distance(self, vector1, vector2):
        """Calculate Euclidean distance between two vectors"""
        return EvaluationMetrics.euclidean_distance(vector1, vector2)


if __name__ == "__main__":
    print("Evaluation module loaded successfully")
