# Facial Recognition Under Varying Conditions Investigation

## Project Overview

This comprehensive investigation studies facial recognition performance under varying environmental conditions, comparing **Preprocessing Techniques** against **Recommended Models**, with rigorous statistical testing.

### Investigation Goals

1. **Accuracy Scores**: Measure recognition accuracy under different conditions
2. **PC Performance Effects**: Monitor CPU, memory, and processing time
3. **Statistical Analysis**: Determine significant differences and relationships

## Project Structure

```
Statistics/
├── database/
│   └── reference_images/          # Reference face images for database
├── captured/                       # Captured test images during investigation
├── results/                        # Investigation results (JSON)
├── logs/                           # Application logs
│
├── config.py                       # Configuration settings
├── camera_utils.py                 # Camera capture and face detection
├── preprocessing_techniques.py     # Preprocessing pipeline
├── recognition_models.py           # Face recognition models
├── evaluation.py                   # Accuracy evaluation metrics
├── performance_monitor.py          # Performance monitoring
├── statistical_tests.py            # Statistical testing suite
│
├── run_preprocessing_techniques.py  # Preprocessing techniques pipeline
├── run_recommended_models.py        # Recommended models pipeline
├── main_analysis.py                # Comprehensive statistical analysis
├── main.py                         # Main control interface
│
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## Testing Conditions

### Environmental Conditions

1. **Normal**: Standard lighting and face angle
2. **Low Light**: Challenging lighting conditions
3. **Motion Blur**: Blurred/moving faces
4. **Sensor Noise**: High noise in camera feed
5. **Extreme Angles**: Non-frontal face positions

### Preprocessing Techniques

1. **Histogram Equalization**: Enhance contrast
2. **Adaptive Histogram Equalization (CLAHE)**: Advanced contrast enhancement
3. **Gamma Correction**: Adjust brightness
4. **Laplacian Sharpening**: Sharpen blurred images
5. **Wiener Filter**: Reduce noise
6. **Non-local Means Denoising**: Advanced denoising
7. **Motion Blur Reduction**: Reduce motion blur effects
8. **Affine Transformations**: Handle extreme angles
9. **Perspective Transformation**: 3D-like view adjustment

### Recommended Models

1. **Eigenfaces**: Principal Component Analysis-based recognition
2. **LBPH**: Local Binary Patterns Histograms
3. **Fisherface**: Fisher Discriminant Analysis
4. **SIFT Matching**: Scale-Invariant Feature Transform
5. **ORB Matching**: Oriented FAST and Rotated BRIEF
6. **Ensemble**: Combined predictions from multiple models

## Statistical Tests

### Test I: Independent t-test

**Hypothesis**: Preprocessing Techniques vs. Recommended Models

- Tests if two independent groups have significantly different means
- Output: t-statistic, p-value, Cohen's d effect size

### Test II: Paired t-test

**Hypothesis**: Same technique before vs. after preprocessing

- Tests if dependent measurements are significantly different
- Output: t-statistic, p-value, mean differences

### Test III: One-way ANOVA

**Hypothesis**: Compare three or more approaches

- Tests if means differ significantly across multiple groups
- Groups: Preprocessing vs. Models vs. Hybrid approaches
- Output: F-statistic, p-value, Eta-squared effect size

### Test IV: Simple Linear Regression

**Hypothesis**: One model accuracy predicts another

- Determines if one variable predicts another
- Example: Eigenfaces accuracy → Ensemble accuracy
- Output: Slope, intercept, R², RMSE

### Test V: Multiple Linear Regression

**Hypothesis**: Multiple models combined predict accuracy

- Multiple predictor variables → Single response
- Example: Eigenfaces, LBPH, Fisherface → Ensemble accuracy
- Output: Coefficients, R², Adjusted R²

### Test VI: Logistic Regression

**Hypothesis**: Predict high vs. low accuracy

- Binary classification: What predicts high/low accuracy?
- Output: Accuracy, Sensitivity, Specificity, Precision

## Installation & Setup

### Prerequisites

- Python 3.8+
- Webcam/Camera device
- Windows/Linux/macOS

### Installation Steps

1. **Clone/Extract Project**

   ```bash
   cd Statistics
   ```

2. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Verify Installation**
   ```bash
   python -c "import cv2, numpy, scipy, sklearn, psutil; print('All dependencies installed!')"
   ```

## Usage Guide

### Quick Start

```bash
python main.py
```

This launches the interactive menu interface.

### Step-by-Step Workflow

#### 1. Setup Database

```bash
python main.py
# Select Option 1: "Setup Database"
# Follow prompts to capture reference images
```

**What it does:**

- Initializes camera connection
- Captures 5 reference images of a person
- Stores images in `database/reference_images/subject_001/`

**Expected output:**

- Console feedback on captured frames
- Log file: `logs/camera.log`

#### 2. Run Preprocessing Techniques Pipeline

```bash
python main.py
# Select Option 2: "Preprocessing Pipeline"
```

**What it does:**

- Loads reference images from database
- Captures test image from camera
- Applies 5 preprocessing techniques
- Compares each preprocessed image with references
- Calculates accuracy scores

**Expected output:**

- Results: `results/preprocessing_investigation_TIMESTAMP.json`
- Logs: `logs/preprocessing_pipeline.log`
- Performance metrics: `logs/performance.log`

**Sample Output:**

```json
[
  {
    "condition": "low_light",
    "preprocessing": "low_light",
    "accuracy": 85.42,
    "accuracies": [84.2, 86.5, 85.3],
    "num_comparisons": 3
  },
  ...
]
```

#### 3. Run Recommended Models Pipeline

```bash
python main.py
# Select Option 3: "Models Pipeline"
```

**What it does:**

- Trains all 6 models on reference images
- Captures test image from camera
- Tests each model on all conditions
- Records confidence scores and predictions
- Evaluates model accuracy

**Expected output:**

- Results: `results/models_investigation_TIMESTAMP.json`
- Logs: `logs/models_pipeline.log`

**Sample Output:**

```json
[
  {
    "model": "eigenfaces",
    "condition": "normal",
    "predicted_label": 0,
    "confidence": 92.5,
    "correct_prediction": true,
    "expected_label": 0
  },
  ...
]
```

#### 4. Run Statistical Analysis

```bash
python main.py
# Select Option 4: "Statistical Analysis"
```

**What it does:**

- Loads all investigation results
- Performs all 6 statistical tests
- Generates comprehensive analysis report
- Displays findings and conclusions

**Expected output:**

- Analysis: `results/statistical_analysis_TIMESTAMP.json`
- Console output with detailed test results
- Logs: `logs/analysis.log`

#### 5. View Results

```bash
python main.py
# Select Option 5: "View Results"
```

Displays all generated result files for review.

#### 6. System Information

```bash
python main.py
# Select Option 6: "System Information"
```

Shows CPU, memory, and project structure information.

## Running Individual Modules

### Direct Pipeline Execution

```bash
# Preprocessing Techniques
python run_preprocessing_techniques.py

# Recommended Models
python run_recommended_models.py

# Statistical Analysis
python main_analysis.py
```

## Output Files

### Results Directory (`results/`)

| File                                 | Description                     |
| ------------------------------------ | ------------------------------- |
| `preprocessing_investigation_*.json` | Preprocessing technique results |
| `models_investigation_*.json`        | Face recognition model results  |
| `statistical_analysis_*.json`        | Statistical test results        |
| `evaluation_results.json`            | Detailed accuracy scores        |
| `performance_metrics.json`           | CPU/Memory performance data     |

### Logs Directory (`logs/`)

| File                | Description                  |
| ------------------- | ---------------------------- |
| `camera.log`        | Camera operations log        |
| `preprocessing.log` | Preprocessing techniques log |
| `models.log`        | Recognition models log       |
| `evaluation.log`    | Accuracy evaluation log      |
| `statistics.log`    | Statistical tests log        |
| `performance.log`   | Performance monitoring log   |
| `main_control.log`  | Main control script log      |

### Database Directory (`database/`)

```
database/
└── reference_images/
    └── subject_001/
        ├── frame_0_20240511_120530.jpg
        ├── frame_1_20240511_120532.jpg
        └── ...
```

### Captured Directory (`captured/`)

```
captured/
└── subject_001/
    └── test_capture_*.jpg
```

## Understanding Results

### Accuracy Score Interpretation

- **90-100%**: Excellent match
- **80-89%**: Good match
- **70-79%**: Fair match
- **60-69%**: Poor match
- **<60%**: No match

### Statistical Test Interpretation

#### p-value

- **p < 0.05**: Statistically significant difference (reject null hypothesis)
- **p ≥ 0.05**: No significant difference (fail to reject null hypothesis)

#### Effect Size (Cohen's d)

- **0.2**: Small effect
- **0.5**: Medium effect
- **0.8**: Large effect

#### R² (Coefficient of Determination)

- **0.7-1.0**: Strong relationship
- **0.4-0.7**: Moderate relationship
- **0.0-0.4**: Weak relationship

## Performance Metrics

The system tracks:

1. **Processing Time**: Milliseconds per operation
2. **Memory Usage**: MB used per test
3. **CPU Utilization**: Percentage of CPU usage
4. **Thresholds**:
   - Warning if memory > 500 MB
   - Warning if CPU > 80%

## Troubleshooting

### Camera Issues

**Problem**: "Failed to open camera"

**Solutions**:

1. Check camera connection
2. Verify camera permissions
3. Close other applications using camera
4. Check `config.py` CAMERA_INDEX (default: 0)

### Import Errors

**Problem**: "ModuleNotFoundError"

**Solutions**:

```bash
pip install --upgrade -r requirements.txt
pip install opencv-python opencv-contrib-python
```

### Face Detection Issues

**Problem**: "No face detected"

**Solutions**:

1. Ensure adequate lighting
2. Position face directly in front of camera
3. Ensure face is clearly visible
4. Try from different angles

### Memory Issues

**Problem**: "MemoryError" or system slowdown

**Solutions**:

1. Close unnecessary applications
2. Reduce reference image dataset
3. Use lower resolution images

## Performance Tips

1. **Optimal Camera Distance**: 30-60 cm from camera
2. **Lighting**: Well-lit environment (>500 lux)
3. **Face Position**: Front-facing, neutral expression
4. **Image Quality**: High resolution (640x480+)
5. **System Resources**: Close other applications

## Configuration

Edit `config.py` to customize:

```python
# Camera settings
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# Performance thresholds
MEMORY_THRESHOLD_MB = 500
CPU_THRESHOLD_PERCENT = 80

# Statistical settings
SIGNIFICANCE_LEVEL = 0.05
SAMPLE_SIZE_MIN = 30
```

## Advanced Usage

### Custom Preprocessing Pipeline

```python
from preprocessing_techniques import PreprocessingPipeline

pipeline = PreprocessingPipeline()
processed_image = pipeline.preprocess_low_light(image)
```

### Direct Model Testing

```python
from recognition_models import EigenfacesRecognizer

model = EigenfacesRecognizer()
model.train(training_images, labels)
label, confidence = model.predict(test_image)
```

### Custom Statistical Tests

```python
from statistical_tests import StatisticalTests

tests = StatisticalTests()
result = tests.independent_ttest(group1, group2)
```

## Project Architecture

### Module Dependencies

```
main.py
├── run_preprocessing_techniques.py
├── run_recommended_models.py
└── main_analysis.py
    ├── preprocessing_techniques.py
    ├── recognition_models.py
    ├── evaluation.py
    ├── statistical_tests.py
    ├── camera_utils.py
    ├── performance_monitor.py
    └── config.py
```

### Data Flow

```
Camera Capture
    ↓
Face Detection
    ↓
Image Storage
    ↓
Preprocessing
    ↓
Model Inference
    ↓
Accuracy Evaluation
    ↓
Performance Tracking
    ↓
Statistical Analysis
    ↓
Results Report
```

## License

This project is provided for educational and research purposes.

## Support & Documentation

- **Logs**: Check `logs/` directory for detailed error messages
- **Results**: Review `results/` for detailed output data
- **Configuration**: Adjust `config.py` for custom settings
- **Console Output**: Provides real-time progress updates

## Key References

### Face Recognition

- Eigenfaces: Turk & Pentland (1991)
- Fisherface: Belhumeur et al. (1997)
- LBPH: Ahonen et al. (2006)

### Image Processing

- Histogram Equalization: Gonzalez & Woods
- SIFT: Lowe (2004)
- ORB: Rublee et al. (2011)

### Statistics

- t-test: Student (1908)
- ANOVA: Fisher (1921)
- Logistic Regression: Cox (1958)

## Future Enhancements

- [ ] GPU acceleration with CUDA
- [ ] Real-time performance dashboard
- [ ] Deep learning model integration (ResNet, VGGFace)
- [ ] Multi-person recognition
- [ ] 3D face alignment
- [ ] Liveness detection
- [ ] Cloud deployment
- [ ] Web interface

---

**Last Updated**: May 11, 2026

**Author**: Facial Recognition Research Team

**Status**: Production Ready
"# Preprocessing-Technique" 
"# Facial-Recognition-with-Streamlit" 
