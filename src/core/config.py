"""
Configuration settings for Facial Recognition Investigation
"""

import os
import cv2
from pathlib import Path

# Project directories - go up from src/core to project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATABASE_DIR = PROJECT_ROOT / "database"
REFERENCE_IMAGES_DIR = DATABASE_DIR / "reference_images"
CAPTURED_DIR = PROJECT_ROOT / "captured"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories if they don't exist
for directory in [DATABASE_DIR, REFERENCE_IMAGES_DIR, CAPTURED_DIR, RESULTS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Camera settings
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# Face detection settings
FACE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

# Image processing settings
HISTOGRAM_BINS = 256
GAMMA_VALUES = [0.5, 1.0, 1.5, 2.0]  # For varying lighting conditions
WIENER_FILTER_SIZE = (5, 5)

# Model settings
CONFIDENCE_THRESHOLD = 0.5
DISTANCE_THRESHOLD = 0.6

# Testing conditions
TEST_CONDITIONS = {
    'normal': {'description': 'Normal lighting and angle'},
    'low_light': {'description': 'Low lighting conditions', 'gamma': 0.5},
    'motion_blur': {'description': 'Motion blur simulation'},
    'sensor_noise': {'description': 'High sensor noise'},
    'extreme_angle': {'description': 'Extreme face angles'}
}

# Statistical test settings
SIGNIFICANCE_LEVEL = 0.05
SAMPLE_SIZE_MIN = 30

# Performance monitoring
MEMORY_THRESHOLD_MB = 500  # Alert if memory usage exceeds this
CPU_THRESHOLD_PERCENT = 80  # Alert if CPU usage exceeds this

# Results storage
RESULTS_FORMAT = 'json'  # 'json', 'csv', or 'both'

# Logging
LOG_LEVEL = 'INFO'
LOG_FILE = LOGS_DIR / 'investigation.log'
