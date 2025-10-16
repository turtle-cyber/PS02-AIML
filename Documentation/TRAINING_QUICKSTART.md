# Quick Start: Training Without Labeled Phishing Data

## Your Situation
- ✅ CSE pages (legitimate)
- ✅ Parked domains
- ❌ No labeled phishing

## Solution: Use Rule-Based Verdicts + CSE Data

### Step 1: Prepare Data (10 min)
```bash
python data_prep/prepare_training_data.py
# Uses ChromaDB verdicts as labels:
# - benign: CSE pages + low risk_score
# - phishing: high risk_score (>=70)
```

### Step 2: Train XGBoost (30 min)
```bash
python models/tabular/train_xbg.py --csv data/train_features.csv --outdir models/tabular/out
```

### Step 3: Run Pipeline (5 min)
```bash
python integration/chromadb_to_ml.py
```

## Alternative: Use PhishTank

1. Download: https://phishtank.com/developer_info.php
2. Crawl URLs → Extract features
3. Train with benign=CSE + phishing=PhishTank

## Minimal Working Example

**If you just want it working today:**
```bash
# Use rule-scorer verdicts as training labels
python data_prep/prepare_training_data.py
python models/tabular/train_xbg.py --csv data/train_features.csv --outdir models/tabular/out
python integration/chromadb_to_ml.py
```

Done! Tabular model trained and running.
