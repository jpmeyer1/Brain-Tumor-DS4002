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
│   ├── 
│   └── 
```

# Reproduction
