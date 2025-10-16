# Complete Pipeline Commands - Tabular + Vision + Text/OCR

Execute these commands in order to build the complete multi-modal detection system.

---

## Phase 1: Install Dependencies

### Step 1.1: Core Dependencies (Already Installed ✅)
```bash
pip install pandas numpy scikit-learn joblib
pip install pillow imagehash tqdm
```

### Step 1.2: PyTorch (Required for Vision Models)
```bash
# For CUDA (if you have NVIDIA GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# OR for CPU only (slower but works everywhere)
pip install torch torchvision
```

### Step 1.3: CLIP for Visual Similarity
```bash
pip install open-clip-torch
```

### Step 1.4: OCR (Optional - for text extraction from screenshots)
```bash
# Windows: Download Tesseract from https://github.com/UB-Mannheim/tesseract/wiki
# After installing Tesseract, run:
pip install pytesseract

# Linux/Mac:
# sudo apt-get install tesseract-ocr  # Linux
# brew install tesseract              # Mac
# pip install pytesseract
```

---

## Phase 2: Data Extraction & Feature Engineering

### Step 2.1: Extract Tabular + Text Features
```bash
python data_prep/load_cse_data.py
```
**Output:**
- `data/cse_benign.csv` (35 tabular features)
- `data/cse_text.csv` (text data for BERT)

### Step 2.2: Extract Visual Features (Phash + OCR)
```bash
python data_prep/extract_visual_features.py --screenshots CSE/out/screenshots --output data/cse_visual_features.csv
```
**Output:**
- `data/cse_visual_features.csv` (phash, OCR text, OCR stats)

**Note:** OCR will be skipped if Tesseract not installed. Phash works without it.

### Step 2.3: Fix Domain Names in Visual Features
```bash
python data_prep/fix_visual_domains.py
```
**Output:**
- Updates `data/cse_visual_features.csv` with correct domain names

### Step 2.4: Merge All Features
```bash
python data_prep/merge_all_features.py
```
**Output:**
- `data/cse_all_features.csv` (41 features: tabular + visual + text)

---

## Phase 3: Model Training

### Step 3.1: Train Tabular Anomaly Detector (with ALL features)
```bash
python models/tabular/train_anomaly.py --csv data/cse_all_features.csv --outdir models/tabular/anomaly_all --contamination 0.05
```
**Output:**
- `models/tabular/anomaly_all/anomaly_detector.joblib`
- `models/tabular/anomaly_all/metadata.json`

**Uses:** 39 features (tabular + favicon hash + screenshot phash + text + OCR stats)

### Step 3.2: Build CLIP Visual Similarity Index
```bash
python models/vision/build_cse_index.py --img_dir CSE/out/screenshots --outdir models/vision/cse_index --model ViT-B-32
```
**Output:**
- `models/vision/cse_index/cse_embeddings.npy` (CLIP embeddings for 124 CSE screenshots)
- `models/vision/cse_index/cse_metadata.json` (domain mappings)
- `models/vision/cse_index/index_stats.json` (statistics)

**Purpose:** Semantic visual similarity (catches "similar but not identical" pages)

### Step 3.3: Train Vision Autoencoder (Optional)
```bash
python models/vision/train_cse_autoencoder.py --img_dir CSE/out/screenshots --outdir models/vision/autoencoder --epochs 50 --batch_size 16 --lr 1e-4
```
**Output:**
- `models/vision/autoencoder/autoencoder_best.pth`

**Purpose:** Detect visually anomalous screenshots (high reconstruction error = phishing)

---

## Phase 4: Build Detection Databases

### Step 4.1: Create Favicon Hash Database
```bash
python -c "
import pandas as pd
df = pd.read_csv('data/cse_all_features.csv')
favicon_db = df[['registrable', 'favicon_md5', 'favicon_sha256']].dropna()
favicon_db = favicon_db[favicon_db['favicon_md5'] != '']
favicon_db.to_csv('data/cse_favicon_db.csv', index=False)
print(f'Created favicon DB with {len(favicon_db)} entries')
"
```
**Output:**
- `data/cse_favicon_db.csv`

### Step 4.2: Create Screenshot Phash Database
```bash
python -c "
import pandas as pd
df = pd.read_csv('data/cse_visual_features.csv')
phash_db = df[['registrable', 'screenshot_phash']].dropna()
phash_db.to_csv('data/cse_phash_db.csv', index=False)
print(f'Created phash DB with {len(phash_db)} entries')
"
```
**Output:**
- `data/cse_phash_db.csv`

---

## Phase 5: Validation

### Step 5.1: Run Pipeline Validation
```bash
python test_pipeline.py
```
**Expected:** 4/4 tests passing

### Step 5.2: Test Detection (Manual)
```python
# Create test script: test_detection.py
import pandas as pd
import joblib
import numpy as np

# Load models
anomaly_model = joblib.load("models/tabular/anomaly_all/anomaly_detector.joblib")
clip_embeddings = np.load("models/vision/cse_index/cse_embeddings.npy")

# Load databases
favicon_db = pd.read_csv("data/cse_favicon_db.csv")
phash_db = pd.read_csv("data/cse_phash_db.csv")

print("All models and databases loaded successfully!")
print(f"Tabular features: {len(anomaly_model.feature_names_in_)} features")
print(f"CLIP embeddings: {len(clip_embeddings)} CSE screenshots")
print(f"Favicon database: {len(favicon_db)} entries")
print(f"Phash database: {len(phash_db)} entries")
```

Run:
```bash
python test_detection.py
```

---

## Summary of Outputs

### Data Files:
```
data/
  ├── cse_benign.csv              # 207 × 35 tabular features
  ├── cse_text.csv                # 207 × 3 text data
  ├── cse_visual_features.csv     # 85 × 7 visual features
  ├── cse_all_features.csv        # 207 × 41 merged features
  ├── cse_favicon_db.csv          # Favicon hash lookup
  └── cse_phash_db.csv            # Screenshot phash lookup
```

### Model Files:
```
models/
  ├── tabular/anomaly_all/
  │   ├── anomaly_detector.joblib     # IsolationForest (39 features)
  │   └── metadata.json
  ├── vision/cse_index/
  │   ├── cse_embeddings.npy          # CLIP embeddings
  │   ├── cse_metadata.json
  │   └── index_stats.json
  └── vision/autoencoder/
      └── autoencoder_best.pth        # Vision autoencoder
```

---

## Troubleshooting

### Issue: PyTorch installation fails
**Solution:**
```bash
# Try CPU version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Issue: CLIP model download slow
**Solution:** It will download ~350MB model first time. Be patient or use faster internet.

### Issue: Tesseract not found
**Solution:**
- Windows: Download installer from https://github.com/UB-Mannheim/tesseract/wiki
- Add to PATH: `C:\Program Files\Tesseract-OCR`
- OCR is optional - pipeline works without it

### Issue: CUDA out of memory
**Solution:**
```bash
# Reduce batch size in autoencoder training
python models/vision/train_cse_autoencoder.py --batch_size 8
```

---

## Quick Start (All Commands)

Copy-paste this entire block:

```bash
# Phase 1: Dependencies
pip install torch torchvision open-clip-torch pillow imagehash pytesseract tqdm

# Phase 2: Data Extraction
python data_prep/load_cse_data.py
python data_prep/extract_visual_features.py
python data_prep/fix_visual_domains.py
python data_prep/merge_all_features.py

# Phase 3: Model Training
python models/tabular/train_anomaly.py --csv data/cse_all_features.csv --outdir models/tabular/anomaly_all
python models/vision/build_cse_index.py
python models/vision/train_cse_autoencoder.py

# Phase 4: Build Databases
python -c "import pandas as pd; df=pd.read_csv('data/cse_all_features.csv'); favicon_db=df[['registrable','favicon_md5','favicon_sha256']].dropna(); favicon_db[favicon_db['favicon_md5']!=''].to_csv('data/cse_favicon_db.csv',index=False); print('Favicon DB created')"
python -c "import pandas as pd; df=pd.read_csv('data/cse_visual_features.csv'); phash_db=df[['registrable','screenshot_phash']].dropna(); phash_db.to_csv('data/cse_phash_db.csv',index=False); print('Phash DB created')"

# Phase 5: Validation
python test_pipeline.py
```

---

## Execution Time Estimates

- Data extraction: ~2 minutes
- Tabular training: ~10 seconds
- CLIP index building: ~2-5 minutes (first time downloads model)
- Autoencoder training: ~10-20 minutes (50 epochs, GPU) or ~60 minutes (CPU)

**Total: ~15-30 minutes for complete pipeline**

---

## Next Steps After Training

See `FINAL_STATUS.md` for:
- Detection workflow
- Integration with ChromaDB
- Example usage code
