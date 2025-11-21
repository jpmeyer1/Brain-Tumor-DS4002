# YOLOv11 Brain Tumor Detection - Remote Server Setup

## Files Created:
- `yolov11_brain_tumor_analysis.py` - Main analysis script
- `run_analysis.sh` - Automated runner script  
- `requirements.txt` - Python dependencies
- `dataset.yaml` - Dataset configuration

## To run on remote server (ddz2sb@realai01):

### Step 1: Transfer files to remote server
```bash
# From your local machine, upload the entire project
scp -r /u/ddz2sb/Brain-Tumor-DS4002 ddz2sb@realai01:~/
```

### Step 2: Connect to remote server
```bash
ssh ddz2sb@realai01
```

### Step 3: Navigate to project and run
```bash
cd Brain-Tumor-DS4002/SCRIPTS

# Install dependencies (first time only)
pip install -r requirements.txt

# Run the complete analysis
./run_analysis.sh
```

## Alternative: Run Python script directly
```bash
cd Brain-Tumor-DS4002/SCRIPTS
python3 yolov11_brain_tumor_analysis.py
```

## What the script does:
1. **Preprocessing Demo** - Shows log transform, histogram equalization, edge detection
2. **Model Training** - Trains YOLOv11 on your brain tumor dataset
3. **Evaluation** - Measures mAP, precision, recall, F1-score, inference time
4. **Visualization** - Creates charts and plots saved to OUTPUT folder
5. **Clinical Analysis** - Assesses readiness for medical deployment

## Output Files (saved to OUTPUT/):
- `preprocessing_pipeline.png` - Preprocessing visualization
- `training_metrics.png` - Training results charts  
- `performance_metrics.png` - Final performance metrics
- `clinical_analysis.png` - Clinical deployment analysis
- `analysis_results.json` - Complete results in JSON format
- `training_log.txt` - Full training log
- Model weights in subdirectories

## Expected Runtime:
- GPU: 30-60 minutes
- CPU: 2-4 hours

## GPU Requirements:
- Recommended: 8GB+ VRAM
- Minimum: 4GB VRAM
- Will automatically use CPU if no GPU available