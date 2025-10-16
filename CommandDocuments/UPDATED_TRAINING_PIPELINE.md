# Updated Multi-Modal Training Pipeline (CSE-Only)

## What Changed

Now extracts **all 37+ features** including:
- ✅ Favicon hashes (MD5/SHA256)
- ✅ OCR text from screenshots
- ✅ Screenshot perceptual hashes (phash)
- ✅ Document text for BERT

## Complete Training Pipeline

### Step 1: Extract ALL Features

```bash
# 1. Extract tabular + text + favicons
python data_prep/load_cse_data.py
# Output: data/cse_benign.csv, data/cse_text.csv

# 2. Extract visual features (OCR + phash)
pip install pytesseract pillow imagehash
python data_prep/extract_visual_features.py
# Output: data/cse_visual_features.csv

# 3. Merge everything
python data_prep/merge_all_features.py
# Output: data/cse_all_features.csv
```

---

### Step 2: Train All Models

#### 2A. Tabular Anomaly Detector (Updated with 37 features)
```bash
python models/tabular/train_anomaly.py \
    --csv data/cse_all_features.csv \
    --outdir models/tabular/anomaly \
    --contamination 0.05
```

**Now uses:** All 37 features including favicon hashes, OCR stats

---

#### 2B. Vision Models

**Option 1: CLIP Index (Similarity Search)**
```bash
python models/vision/build_cse_index.py \
    --img_dir CSE/out/screenshots \
    --outdir models/vision/cse_index
```

**Option 2: Autoencoder (Anomaly Detection)**
```bash
python models/vision/train_cse_autoencoder.py \
    --img_dir CSE/out/screenshots \
    --outdir models/vision/autoencoder \
    --epochs 50
```

**How it works:**
- Trains on CSE screenshots only
- High reconstruction error = anomaly (phishing)

---

#### 2C. Text Model (Optional - when you get phishing data)
```bash
# Requires labeled phishing for training
python models/text/train_mbxt.py \
    --train_csv data/train_text.csv \
    --val_csv data/val_text.csv \
    --outdir models/text/out
```

---

### Step 3: Build Hash Indexes

#### Favicon Hash Database
```python
import pandas as pd

df = pd.read_csv("data/cse_all_features.csv")
favicon_db = df[['registrable', 'favicon_md5', 'favicon_sha256']].dropna()
favicon_db.to_csv("data/cse_favicon_db.csv", index=False)
```

#### Screenshot Phash Database
```python
df = pd.read_csv("data/cse_visual_features.csv")
phash_db = df[['domain', 'screenshot_phash']].dropna()
phash_db.to_csv("data/cse_phash_db.csv", index=False)
```

---

## Multi-Signal Detection

### Detection Workflow

```python
# Load all models
anomaly_detector = joblib.load("models/tabular/anomaly/anomaly_detector.joblib")
clip_index = np.load("models/vision/cse_index/cse_embeddings.npy")
autoencoder = load_autoencoder("models/vision/autoencoder/autoencoder_best.pth")

# Load hash databases
favicon_db = pd.read_csv("data/cse_favicon_db.csv")
phash_db = pd.read_csv("data/cse_phash_db.csv")

# Detect
verdict = detect_multi_signal(
    tabular_features,
    screenshot_path,
    favicon_hash,
    domain
)
```

### Detection Rules (Priority Order)

1. **Favicon Match + Domain Mismatch** → PHISHING (confidence: 0.95)
   ```python
   if favicon_md5 in favicon_db AND domain != matched_cse_domain:
       return "PHISHING"  # Stolen favicon
   ```

2. **Screenshot Phash Match + Domain Mismatch** → PHISHING (confidence: 0.92)
   ```python
   if phash_similarity > 0.95 AND domain != matched_cse_domain:
       return "PHISHING"  # Exact visual clone
   ```

3. **CLIP Similarity + Domain Mismatch** → PHISHING (confidence: 0.88)
   ```python
   if clip_similarity > 0.85 AND domain != matched_cse_domain:
       return "PHISHING"  # Visual mimicry
   ```

4. **Autoencoder Anomaly** → SUSPICIOUS (confidence: 0.70)
   ```python
   if reconstruction_error > threshold:
       return "SUSPICIOUS"  # Unusual screenshot
   ```

5. **Tabular Anomaly** → SUSPICIOUS (confidence: 0.65)
   ```python
   if IsolationForest == -1:
       return "SUSPICIOUS"  # Unusual features
   ```

6. **OCR Keywords + Credential Forms** → SUSPICIOUS (confidence: 0.60)
   ```python
   if ("login" in ocr_text OR "verify" in ocr_text) AND has_credential_form:
       return "SUSPICIOUS"  # Phishing indicators
   ```

---

## Why This Has Low False Positives

### Multiple Independent Checks
Only flags as **PHISHING** when:
- Visual/favicon matches CSE site **AND**
- Domain is different

Random sites won't trigger both conditions.

### Confidence-Based Verdicts
- **High (>0.85):** Visual match + domain mismatch
- **Medium (0.60-0.85):** Anomaly detected
- **Low (<0.60):** No strong signals

---

## Quick Start

```bash
# Full pipeline
python data_prep/load_cse_data.py
python data_prep/extract_visual_features.py
python data_prep/merge_all_features.py

python models/tabular/train_anomaly.py
python models/vision/build_cse_index.py
python models/vision/train_cse_autoencoder.py

# Test
python integration/chromadb_similarity_detection.py
```

---

## Model Summary

| Model | Input | Output | Use Case |
|-------|-------|--------|----------|
| Tabular Anomaly | 37 features | Anomaly score | Detect unusual features |
| CLIP Index | Screenshot | Similarity scores | Find visual matches |
| Autoencoder | Screenshot | Reconstruction error | Detect unusual visuals |
| Favicon Hash | Favicon MD5/SHA256 | Exact match | Brand impersonation |
| Screenshot Phash | Screenshot phash | Similarity | Detect clones |

All models trained on **CSE benign data only** → Low false positives!
