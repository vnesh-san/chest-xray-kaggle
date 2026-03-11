# VinBigData Chest X-ray Abnormalities Detection

YOLOv8-based object detection pipeline for the [VinBigData Chest X-ray Abnormalities Detection](https://www.kaggle.com/competitions/vinbigdata-chest-xray-abnormalities-detection) Kaggle competition.

**Task:** Detect and localize 14 types of thoracic abnormalities in chest X-ray DICOM images.

## Competition Details

- **Data:** ~18,000 DICOM training images annotated by 17 Vietnamese radiologists (≥3 reads per image); ~3,000 test images
- **14 findings:** Aortic enlargement, Atelectasis, Calcification, Cardiomegaly, Consolidation, ILD, Infiltration, Lung Opacity, Nodule/Mass, Other lesion, Pleural effusion, Pleural thickening, Pneumothorax, Pulmonary fibrosis
- **Class 14:** "No finding" — images where all radiologists saw nothing abnormal
- **Metric:** mAP averaged over IoU thresholds 0.4 → 0.75 (step 0.05)
- **Submission:** CSV with `image_id, PredictionString` where each prediction is `class_id confidence xmin ymin xmax ymax`

## Approach

1. **DICOM preprocessing** — GPU-accelerated windowing + normalization via `DicomPreprocessorGPU` (learnable window center/width)
2. **15-fold stratified cross-validation** — ~14k train / ~1k val per fold; all 15k images seen across folds
3. **YOLOv8x fine-tuning** — pretrained backbone, 50 epochs per fold, 640px input, mosaic augmentation
4. **"No finding" logic** — if no box exceeds confidence threshold at inference, output class 14 with full-image box

## Repository Structure

```
├── notebookb256487d17.ipynb   # Full end-to-end training + submission notebook
├── requirements.txt           # Python dependencies
├── data/
│   └── raw/                   # Downloaded competition data (DICOMs, CSVs)
└── src/                       # (planned) modular pipeline
    ├── data/                  # DICOM preprocessing, label conversion
    ├── models/                # Model wrappers
    ├── training/              # Training loop
    ├── evaluation/            # mAP metrics, failure analysis
    └── inference/             # Submission generation, WBF ensemble
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Download competition data (requires Kaggle API token)
export KAGGLE_API_TOKEN=<your_token>
kaggle competitions download \
  -c vinbigdata-chest-xray-abnormalities-detection \
  -p data/raw/

# Unzip
unzip data/raw/vinbigdata-chest-xray-abnormalities-detection.zip -d data/raw/

# Run the notebook end-to-end
jupyter notebook notebookb256487d17.ipynb
```

## Kaggle Auth (new token format)

This project uses the new Kaggle API token format (`KGAT_...`). Set the token via:

```bash
export KAGGLE_API_TOKEN=KGAT_<your_token>
# or write it to:
echo -n "KGAT_<your_token>" > ~/.kaggle/access_token
chmod 600 ~/.kaggle/access_token
```

## Notebook Pipeline

| Cell | What it does |
|---|---|
| 1 | Install deps, verify CUDA/GPU |
| 2 | Imports, paths, global config (15 folds, 640px, 50 epochs) |
| 3 | `DicomPreprocessorGPU` + `DicomDetectionTrainer` (YOLO subclass) |
| 4 | DICOM → PNG conversion + YOLO label writing (parallel, 4 workers) |
| 5 | 15-fold stratified CV training loop |
| 6 | Test inference + `submission.csv` generation |

## Key Design Decisions

- **GPU windowing:** DICOM images have 12-bit pixel values; a learnable sigmoid windowing layer enhances contrast before the detector sees the image
- **15-fold CV:** maximizes data use for a 15k-image dataset; each fold yields a separate checkpoint for ensembling
- **No finding handling:** class 14 is background during training; at inference a fallback prediction is added if the detector finds nothing above threshold
- **Class imbalance:** rare classes (Calcification, Pneumothorax, ILD) are handled by focal loss inside YOLOv8
