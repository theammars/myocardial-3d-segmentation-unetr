# 🫀 Myocardial Perfusion SPECT Segmentation
### 3D Cardiac Segmentation using UNETR + MONAI | 5-Fold Cross Validation

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange?logo=pytorch)
![MONAI](https://img.shields.io/badge/MONAI-1.3+-teal)
![License](https://img.shields.io/badge/License-MIT-green)

> **Thesis Project** — Automated 3D Myocardium segmentation from SPECT imaging using Vision Transformer architecture (UNETR) with a Nested Cross-Validation methodology (80/20 Holdout + 5-Fold CV).

---

## 📊 Final Results

| Metric | Mean ± Std |
|--------|-----------|
| 🎲 Dice Score (F1) | **~0.91 ± 0.02** |
| 📐 IoU Score | **~0.84 ± 0.03** |
| 👁️ Sensitivity | **~0.90 ± 0.02** |
| 🛡️ Specificity | **~0.99 ± 0.01** |

---

## 🗂️ Repository Structure

```
myocardial-spect-segmentation/
│
├── 📓 Myocardial_Segmentation_IDSC.ipynb          # Full pipeline (Training + Evaluation + Visualization)
├── 📄 README.md               # This guide
├── 📄 requirements.txt        # Library dependencies
│
├── 📁 sample_data/            # Sample of patients (dataset preview)
│   ├── dicom/                 # Raw .dcm files
│   └── masks/                 # Ground Truth .nii.gz files
│
└── 📁 assets/                 # Visualization screenshots
    ├── overlay_sample.png
    └── model_segmentation_comparation.png
```

---

## 📦 Dataset

This project uses the **Myocardial Perfusion Scintigraphy Image Database (IDSC)** from PhysioNet:

> 🔗 [https://physionet.org/content/myocardial-perfusion-scintigraphy-image-database/1.0.0/](https://physionet.org/content/myocardial-perfusion-scintigraphy-image-database/1.0.0/)

The dataset contains **100 patients** in the following format:
- **SPECT Image**: Multi-frame DICOM (`.dcm`) — Shape: `(50, 70, 70)`
- **Ground Truth Mask**: NIfTI (`.nii.gz`) — Shape: `(70, 70, 50)`

---

## ⚙️ Path Configuration (MUST BE SET BEFORE RUNNING!)

Open the notebook, find the **`# ⚙️ GLOBAL CONFIG`** cell at the top, and update the following paths to match your environment:

```python
# ============================================================
# ⚙️ GLOBAL CONFIG — UPDATE THESE PATHS BEFORE RUNNING!
# ============================================================

# [1] Path to folder containing .dcm files (SPECT Images)
dicom_path = "/content/drive/MyDrive/IDSC/dicom/"

# [2] Path to folder containing .nii.gz files (Ground Truth Masks)
mask_path  = "/content/drive/MyDrive/IDSC/masks/"

# [3] Temporary model storage (Colab RAM — fast access)
TEMP_MODEL_DIR = "/content/"

# [4] Backup model storage on Google Drive (insurance against disconnects)
GDRIVE_MODEL_DIR = "/content/drive/MyDrive/IDSC/models/"
# ============================================================
```

> [!IMPORTANT]
> If running **locally (not on Google Colab)**, change all paths above to absolute local paths. Example: `dicom_path = "C:/data/IDSC/dicom/"`

---

## 🚀 How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare the Dataset
- Download the dataset from PhysioNet (link above)
- Extract `.dcm` and `.nii.gz` files into their respective folders
- Update paths in the **Global Config** cell of the notebook

### 3. Run Notebook Cells in Order

| # | Cell | Description |
|---|------|-------------|
| 1 | Global Config | Set all paths & random seed |
| 2 | Install & Import | Install MONAI & libraries |
| 3 | Mount Drive | Mount Google Drive (Colab only) |
| 4 | EDA | Explore data distribution |
| 5 | Pairing | Validate DICOM ↔ Mask file pairs |
| 6 | Transforms | Define training augmentations |
| 7 | Resize Function | 3D volume resize utility |
| 8 | **Save Clean Data** | ⚠️ Preprocessing + Transpose Fix |
| 9 | Verify Alignment | Visual check on 5 samples |
| 10 | Train/Test Split | 80 Dev / 20 Test |
| 11 | **K-Fold Training** | 5-Fold × 200 Epochs |
| 12 | Individual Evaluation | Audit each of the 5 fold models |
| 13 | Ensemble Inference | The Avengers (5 Models) + TTA + LCC |
| 14 | 2D Visualization | Overlay Ground Truth vs Prediction |
| 15 | 3D Hologram | Volumetric visualization |

---

## 🔬 Methodology

```
100 Patients
    │
    ├── 80 Patients → Development Set
    │       │
    │       └── 5-Fold Cross Validation
    │               ├── Fold 1: 64 Train / 16 Val
    │               ├── Fold 2: 64 Train / 16 Val
    │               ├── Fold 3: 64 Train / 16 Val
    │               ├── Fold 4: 64 Train / 16 Val
    │               └── Fold 5: 64 Train / 16 Val
    │
    └── 20 Patients → Hidden Test Set (NEVER seen during Training)
```

**Final Inference Pipeline:**
`5 Individual Models` → `Ensemble (Average Probability)` → `Threshold 0.5` → `LCC Post-processing` → **Final Prediction**

---

## 🏗️ Model Architecture

| Parameter | Value |
|-----------|-------|
| Model | UNETR (UNet Transformer) |
| Input Size | 96 × 96 × 96 |
| Feature Size | 16 |
| Hidden Size | 768 |
| Num Heads | 12 |
| Loss Function | DiceCELoss (squared_pred=True) |
| Optimizer | AdamW (lr=1e-4, wd=1e-5) |
| Scheduler | CosineAnnealingLR |
| Epochs | 200 per Fold |

---

## 🐛 Key Fix: Axis Transpose

This dataset has a **spatial axis mismatch** between DICOM and NIfTI files:
- DICOM via `pydicom.pixel_array` → shape `(Z, Y, X)`
- NIfTI via `nibabel.get_fdata()` → shape `(X, Y, Z)`

**Fix applied in Cell 8 (Save Clean Data):**
```python
image = np.transpose(image, (2, 1, 0))  # (Z,Y,X) → (X,Y,Z)
```

> [!WARNING]
> Without this fix, the Ground Truth mask and SPECT image are **spatially misaligned**, causing the Dice Score to plateau at ~70% regardless of how many epochs are trained.

---

## 📚 References

- Hatamizadeh, A., et al. (2022). *UNETR: Transformers for 3D Medical Image Segmentation*. WACV 2022.
- MONAI Consortium. (2023). *MONAI: Medical Open Network for AI*.
- PhysioNet. *Myocardial Perfusion Scintigraphy Image Database (IDSC) v1.0.0*.

---

## 👤 Author

📧 Feel free to open an issue for questions!
