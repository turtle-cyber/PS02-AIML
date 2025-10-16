# Training with Your CSE Data + ChromaDB

## Your Setup
- ✅ CSE folder with legitimate pages (dump_all.jsonl, screenshots, PDFs)
- ✅ ChromaDB with suspicious domains (rule-scorer verdicts)

## Training Steps (15 minutes)

### Step 1: Load CSE Benign Data
```bash
python data_prep/load_cse_data.py
# Reads: CSE/dump_all.jsonl
# Saves: data/cse_benign.csv (all legitimate)
```

### Step 2: Extract Phishing from ChromaDB
```bash
python data_prep/prepare_training_data.py
# Queries ChromaDB for risk_score >= 70
# Saves: data/chromadb_phishing.csv
```

### Step 3: Combine Data
```bash
python data_prep/combine_training_data.py
# Combines benign + phishing
# Saves: data/train_features.csv, data/val_features.csv
```

### Step 4: Train XGBoost
```bash
python models/tabular/train_xbg.py \
  --csv data/train_features.csv \
  --outdir models/tabular/out
```

### Step 5: Run ML Pipeline
```bash
python integration/chromadb_to_ml.py
# Scores all ChromaDB records without ML verdict
```

## What You Get
- XGBoost model trained on YOUR actual data
- CSE pages = benign labels
- High-risk ChromaDB domains = phishing labels
- No external datasets needed!

## Expected Results
- Benign samples: ~100-500 (CSE pages)
- Phishing samples: ~500-2000 (ChromaDB high-risk)
- Model accuracy: 85-95% (depending on data quality)

## Quick Test
```bash
# Run all at once
cd data_prep
python load_cse_data.py && \
python prepare_training_data.py && \
python combine_training_data.py && \
cd .. && \
python models/tabular/train_xbg.py --csv data/train_features.csv --outdir models/tabular/out
```

Done!
