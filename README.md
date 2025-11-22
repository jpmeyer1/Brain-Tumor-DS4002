# Brain Tumors | DS4002 | ThunderPandas
## This repository supports the Brain Tumor Identification project by Team Thunder Pandas. The goal of the project is to develop and evaluate a YOLO-based deep learning model capable of accurately and efficiently detecting and localizing brain tumors in MRI scans. This repository is organized for reproducibility and clarity, following best practices for open data science projects. It contains all scripts, data documentation, outputs, and supporting files needed to understand, reproduce, and extend the analyses.

# Software and Platform
This project was developed and tested using Python 3.10+ on both Google Colab (Linux GPU environment) and a local Windows 11 machine with CUDA-enabled GPU support. The baseline and enhanced YOLOv11 models were trained using the Ultralytics framework, along with standard scientific and medical imaging libraries.

**Primary software and libraries:**

Core Software
- Python 3.10+
- PyTorch ≥ 2.0.0
- Ultralytics YOLOv11 (ultralytics ≥ 8.0.0)
- Google Colab (Linux GPU) or Windows 11 (CUDA-enabled)

Essential Machine Learning & Computer Vision Libraries
- torchvision ≥ 0.15.0
- opencv-python ≥ 4.8.0
- albumentations ≥ 1.3.0
- scikit-learn ≥ 1.3.0

Scientific Computing
- numpy ≥ 1.24.0
- scipy ≥ 1.10.0
- pandas ≥ 2.0.0
  
Visualization
- matplotlib ≥ 3.7.0
- seaborn ≥ 0.12.0
  
Image & Data Utilities
- Pillow ≥ 9.5.0
- PyYAML ≥ 6.0

All dependencies are fully listed in requirements.txt and requirements_enhanced.txt.

# Documentation

```
├── README.md
├── LICENSE.md
├── DATA/
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   ├── labels/
│   │   ├── train/
│   │   └── val/
│   ├── Metadata.md
│   ├── Brain1.png
│   ├── Brain2.png
│   ├── Bounding Box Distribution.png
│   └── Tumor Class Distribution.png
├── SCRIPTS/
│   ├── setup/
│   │   ├── requirements.txt
│   │   └── requirements_enhanced.txt
│   ├── data/
│   │   ├── dataset.yaml
│   │   └── data_preprocessing.py
│   ├── baseline/
│   │   ├── train_yolo.py
│   │   ├── yolov11_brain_tumor_analysis.py
│   │   └── run_analysis.sh
│   ├── enhanced/
│   │   ├── yolov11_enhanced_brain_tumor.py
│   │   └── run_enhanced_analysis.sh
│   ├── evaluation/
│   │   ├── evaluate_model.py
│   │   └── create_enhanced_summary.py
│   └── inference/
│       └── tumor_detector.py
├── OUTPUT/
│   ├── runs/detect/val
│   ├── yolov11_brain_tumor_final/
│   │   └── weights/
│   ├── yolov11_enhanced_attention/
│   │   └── weights/
│   ├── class_distribution_analysis.png
│   ├── dataset_eda_overview.png
│   ├── eda_results.json
│   ├── enhanced_brain_tumor_data.yaml
│   ├── enhanced_results_summary.json
│   ├── enhanced_yolov11_best.pt
│   ├── preprocessing_pipeline.png
│   ├── sample_images_with_annotations.png
│   └── training_log.txt
```

# Reproduction

To reproduce the results in this repository, follow the steps below. The project includes both a baseline YOLOv11 model and an enhanced YOLOv11 model incorporating attention mechanisms and advanced medical image preprocessing.

1. **Clone this repository:**

```
git clone https://github.com/jpmeyer1/Brain-Tumor-DS4002.git
cd Brain-Tumor-DS4002
```

2. **Set up the Python environment:**

Two requirements files are provided:
- requirements.txt — baseline model
- requirements_enhanced.txt — enhanced attention-based model

To install all dependencies (baseline and enhanced):

```
pip install -r SCRIPTS/setup/requirements.txt
pip install -r SCRIPTS/setup/requirements_enhanced.txt
```

3. **Run all scripts in order:**

All scripts are located in the SCRIPTS/ directory and organized by function.

**A. Baseline YOLOv11 Model**

1. Preprocess images: SCRIPTS/data/data_preprocessing.py
2. Run the baseline training pipeline: SCRIPTS/baseline/run_analysis.sh
3. Evaluate baseline model performance: SCRIPTS/evaluation/evaluate_model.py

**B. Enhanced YOLOv11 Model**

1. Run the enhanced model training pipeline: SCRIPTS/enhanced/run_enhanced_analysis.sh

**C. Generate Summary Comparison**

```
python SCRIPTS/evaluation/create_enhanced_summary.py
```

4. **View results:**

All generated results—including model weights, evaluation plots, confusion matrices, logs, and inference outputs—are automatically saved in the OUTPUT/ directory
