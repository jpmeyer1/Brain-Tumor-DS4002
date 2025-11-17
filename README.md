# Brain Tumor Detection using YOLO11n

## DS4002 Project 3 - Deep Learning for Medical Imaging

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO11-green.svg)](https://ultralytics.com/)
[![License](https://img.shields.io/badge/license-AGPL--3.0-blue.svg)](LICENSE)

## 📋 Executive Summary

This project implements an automated brain tumor detection system using a fine-tuned **YOLO11n** (YOLOv11-Nano) deep learning model on MRI brain scans. The goal is to achieve **≥85% mean average precision (mAP)** while maintaining real-time inference speed for clinical decision support.

**Key Features:**
- 🧠 Automated tumor detection and localization in MRI scans
- ⚡ Real-time inference capability
- 📊 Comprehensive evaluation metrics (mAP, Precision, Recall, F1-Score)
- 🎯 Transfer learning from pretrained YOLO11n
- 🔄 Advanced data augmentation pipeline
- 📈 Complete exploratory data analysis

---

## 🎯 Research Question

**Can deep learning object detection models accurately locate brain tumors in medical imaging scans with sufficient precision and speed to serve as effective decision support tools for radiologists?**

### Hypothesis

*A YOLO-based deep learning model trained on MRI brain scans will achieve at least an 85% mean average precision in detecting and localizing brain tumors, outperforming traditional image classification methods in both accuracy and real-time inference speed.*

---

## 📁 Project Structure

```
Brain-Tumor-DS4002/
├── config/
│   ├── training_config.yaml      # Model training hyperparameters
│   └── dataset_template.yaml     # Dataset configuration template
├── data/
│   ├── brain-tumor/              # Raw dataset (download from Ultralytics)
│   │   ├── train/
│   │   │   ├── images/
│   │   │   └── labels/
│   │   └── test/
│   │       ├── images/
│   │       └── labels/
│   └── processed/                # Preprocessed dataset
├── scripts/
│   ├── preprocess_data.py        # Data preprocessing and validation
│   ├── train_model.py            # Model training script
│   ├── evaluate_model.py         # Comprehensive evaluation
│   └── detect_tumor.py           # Inference tool
├── notebooks/
│   └── exploratory_analysis.ipynb # EDA notebook
├── models/
│   └── brain_tumor_yolo11n/      # Trained model checkpoints
├── results/
│   ├── eda/                      # EDA outputs
│   ├── evaluation/               # Evaluation metrics and plots
│   └── detections/               # Detection visualizations
├── utils/                        # Utility functions
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## 🚀 Quick Start

### 1. Environment Setup

```powershell
# Clone the repository
git clone https://github.com/jpmeyer1/Brain-Tumor-DS4002.git
cd Brain-Tumor-DS4002

# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Dataset

Download the **Ultralytics Brain Tumor Dataset**:

```powershell
# Using Ultralytics CLI
yolo download dataset brain-tumor

# Or manually download from:
# https://docs.ultralytics.com/datasets/detect/brain-tumor/
```

Expected structure:
- Training: 893 labeled MRI images
- Testing: 223 labeled MRI images
- Format: YOLO format (images + txt annotations)

### 3. Data Preprocessing

```powershell
python scripts/preprocess_data.py `
    --data-yaml data/brain-tumor/data.yaml `
    --output-dir data/processed `
    --img-size 640
```

This will:
- ✅ Validate all images and labels
- ✅ Normalize image sizes to 640×640
- ✅ Remove corrupted files
- ✅ Generate statistics
- ✅ Create processed dataset config

### 4. Exploratory Data Analysis

```powershell
jupyter notebook notebooks/exploratory_analysis.ipynb
```

The EDA notebook provides:
- 📊 Class distribution (positive vs negative cases)
- 📏 Bounding box statistics (size, location, area)
- 🗺️ Tumor location heatmaps
- 🖼️ Sample visualizations

### 5. Model Training

```powershell
python scripts/train_model.py `
    --config config/training_config.yaml
```

**Training Configuration Highlights:**
- **Model:** YOLO11n (lightweight, fast)
- **Epochs:** 100
- **Batch Size:** 16
- **Image Size:** 640×640
- **Optimizer:** AdamW
- **Learning Rate:** 0.001 → 0.00001 (cosine decay)
- **Augmentation:** Rotation, scaling, flipping, mosaic, mixup

**Hardware Requirements:**
- GPU: NVIDIA GPU with ≥8GB VRAM (recommended)
- CPU: Fallback supported but slower
- RAM: ≥16GB

**Training Time:**
- ~2-4 hours on NVIDIA RTX 3080
- ~10-15 hours on CPU

### 6. Model Evaluation

```powershell
python scripts/evaluate_model.py `
    --model models/brain_tumor_yolo11n/weights/best.pt `
    --data-yaml data/processed/dataset.yaml `
    --output-dir results/evaluation `
    --measure-time
```

**Evaluation Outputs:**
- 📈 mAP@0.5, mAP@0.5:0.95
- 📊 Precision, Recall, F1-Score
- 🎯 Confusion matrix
- ⏱️ Inference time (ms per image)
- 📉 Confidence distribution plots
- 📄 Comprehensive evaluation report

### 7. Inference on New Images

**Single Image:**
```powershell
python scripts/detect_tumor.py `
    --model models/brain_tumor_yolo11n/weights/best.pt `
    --source path/to/image.jpg `
    --output-dir results/detections
```

**Batch Processing:**
```powershell
python scripts/detect_tumor.py `
    --model models/brain_tumor_yolo11n/weights/best.pt `
    --source path/to/images/ `
    --output-dir results/detections
```

**Video Processing:**
```powershell
python scripts/detect_tumor.py `
    --model models/brain_tumor_yolo11n/weights/best.pt `
    --source path/to/video.mp4 `
    --output-dir results/detections `
    --video `
    --display
```

---

## 📊 Dataset Details

### Ultralytics Brain Tumor Dataset

| **Attribute** | **Details** |
|---------------|-------------|
| **Source** | Ultralytics (MRI/CT scans) |
| **Training Samples** | 893 images |
| **Test Samples** | 223 images |
| **Classes** | 2 (Negative: 0, Positive: 1) |
| **Format** | YOLO format (normalized bbox) |
| **License** | AGPL-3.0 |
| **Resolution** | Variable (normalized to 640×640) |

### Data Dictionary

| **Column** | **Description** | **Example Values** |
|------------|-----------------|-------------------|
| `image` | MRI brain scan image (JPG/PNG) | - |
| `image_title` | Image filename | `00054_145.jpg` |
| `label_title` | Label filename | `00054_145.txt` |
| `class` | 0 = Negative, 1 = Positive (tumor) | `0`, `1` |
| `x_center` | Bounding box X-center (normalized) | `0.344484` |
| `y_center` | Bounding box Y-center (normalized) | `0.342723` |
| `width` | Bounding box width (normalized) | `0.221831` |
| `height` | Bounding box height (normalized) | `0.176056` |

### Ethical Considerations

⚠️ **Critical Clinical Considerations:**
- **False Negatives:** Missed tumors could delay treatment → Minimize FN rate
- **False Positives:** Unnecessary stress/procedures → Balance precision
- **Privacy:** All data must be de-identified (HIPAA compliant)
- **Consent:** Patient consent required for dataset usage
- **Clinical Validation:** Model requires medical validation before deployment

---

## 🧪 Methodology

### Analysis Framework

```
1. Preprocess Data → 2. Train YOLO11n → 3. Evaluate Model → 4. Deploy Tool
```

### Step-by-Step Pipeline

#### **Step 1: Data Preprocessing**
- Load and validate 893 training + 223 test MRI images
- Normalize pixel values and resize to 640×640
- Parse YOLO format labels
- Remove corrupted/missing data
- Generate dataset statistics

#### **Step 2: Data Augmentation**
- **Geometric:** Rotation (±15°), scaling (±50%), flipping (50% horizontal)
- **Color:** HSV adjustments (hue, saturation, brightness)
- **Advanced:** Mosaic (100%), Mixup (10%)
- **Purpose:** Improve generalization and prevent overfitting

#### **Step 3: Transfer Learning**
- Initialize with pretrained YOLO11n weights (COCO dataset)
- Fine-tune detection head on medical imaging domain
- Progressive unfreezing of layers
- Optimize for tumor localization

#### **Step 4: Training (100 Epochs)**
- Optimizer: AdamW
- Learning rate: Cosine decay (0.001 → 0.00001)
- Early stopping: Patience of 50 epochs
- Monitor: mAP, loss curves, validation metrics

#### **Step 5: Evaluation**
- **Metrics:** mAP@0.5, mAP@0.5:0.95, Precision, Recall, F1-Score
- **Focus:** Minimize false negatives (critical for patient safety)
- **Speed:** Measure inference time for real-time capability

#### **Step 6: Deployment**
- Python tool for batch/single image inference
- Outputs: Bounding boxes + confidence scores
- Real-time visualization
- CSV summary reports

---

## 📈 Expected Results

### Performance Targets

| **Metric** | **Target** | **Rationale** |
|-----------|-----------|---------------|
| mAP@0.5 | **≥85%** | Hypothesis validation |
| Recall | **≥90%** | Minimize false negatives |
| Precision | **≥80%** | Control false positives |
| Inference Time | **<50ms** | Real-time capability |
| F1-Score | **≥85%** | Balanced performance |

### Hypothesis Evaluation

✅ **Success Criteria:** mAP@0.5 ≥ 85%  
❌ **Failure Criteria:** mAP@0.5 < 85%

---

## 🛠️ Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```yaml
# In config/training_config.yaml, reduce batch size:
batch_size: 8  # Instead of 16
```

**2. Slow Training on CPU**
```powershell
# Enable GPU (if available)
# Check: python -c "import torch; print(torch.cuda.is_available())"
```

**3. Import Errors**
```powershell
pip install --upgrade ultralytics torch torchvision
```

**4. Dataset Not Found**
```powershell
# Verify paths in config/dataset_template.yaml
# Ensure images and labels are in correct directories
```

---

## 📚 References

1. **Ultralytics** (2023). Brain Tumor Detection Dataset. *Ultralytics Documentation*.  
   https://docs.ultralytics.com/datasets/detect/brain-tumor/

2. **Redmon, J., Divvala, S., Girshick, R., & Farhadi, A.** (2016). You only look once: Unified, real-time object detection. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, 779-788.

3. **Litjens, G., Kooi, T., Bejnordi, B. E., et al.** (2017). A survey on deep learning in medical image analysis. *Medical Image Analysis*, 42, 60-88.

4. **Ultralytics** (2024). YOLOv11 Documentation.  
   https://docs.ultralytics.com/

---

## 📄 License

This project is licensed under the **GNU Affero General Public License v3 (AGPL-3.0)** via Ultralytics.

⚠️ **Important:**
- Free for research and open-source projects
- Commercial use requires separate license from Ultralytics
- Modified versions must be open-sourced under AGPL-3.0

---

## 👥 Contributors

**DS4002 Project Team**
- Research Design & Implementation
- Data Analysis & Visualization
- Model Training & Evaluation
- Documentation & Reporting

---

## 🙏 Acknowledgments

- **Ultralytics** for the Brain Tumor Dataset and YOLO framework
- **PyTorch** team for deep learning infrastructure
- **Medical imaging community** for dataset contributions

---

## 📞 Contact & Support

For questions or issues:
1. Check the [Troubleshooting](#-troubleshooting) section
2. Review Ultralytics documentation: https://docs.ultralytics.com/
3. Open an issue on GitHub repository

---

## 🔄 Project Status

- [x] Project setup and structure
- [x] Data preprocessing pipeline
- [x] Exploratory data analysis
- [x] Model training implementation
- [x] Evaluation framework
- [x] Inference tool
- [ ] Dataset acquisition (user action required)
- [ ] Model training execution
- [ ] Results validation
- [ ] Final report and presentation

**Next Steps:**
1. Download the Ultralytics Brain Tumor Dataset
2. Run preprocessing script
3. Execute training pipeline
4. Evaluate model performance
5. Validate hypothesis (mAP ≥ 85%)

---

**Last Updated:** November 17, 2025  
**Version:** 1.0.0
