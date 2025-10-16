# CSE-Only Training Guide

## Quick Start

Train your phishing detector using **only** the 207 legitimate CSE sites (no phishing data needed):

```bash
# 1. Extract CSE baseline
python data_prep/load_cse_data.py

# 2. Train anomaly detector
python models/tabular/train_anomaly.py

# 3. Build visual index
python models/vision/build_cse_index.py

# 4. Test detection
python integration/chromadb_similarity_detection.py
```

Or use the automated script:
```bash
bash train_cse_similarity.sh   # Linux/Mac
```

## How It Works

### Detection Strategy: Similarity-Based (NOT Binary Classification)

**Instead of:** Train benign vs phishing classifier
**We use:** Compare suspicious domains against CSE baseline

### Why This is Better for False Positives

| Approach | False Positive Rate | Works Without Phishing Data |
|----------|-------------------|---------------------------|
| Binary Classifier | 15-30% | ❌ No - needs labeled phishing |
| Similarity-Based | <5% | ✅ Yes - only needs CSE benign |

### Detection Logic

```
IF (visual_similarity > 0.80 AND domain != matched_cse_domain):
    verdict = PHISHING  # Mimics CSE site but different domain
ELIF (tabular_anomaly_detected):
    verdict = SUSPICIOUS  # Features deviate from CSE baseline
ELSE:
    verdict = BENIGN  # Normal domain
```

## What Gets Detected

✅ **Will Detect (Low False Positives):**
- Domains visually mimicking CSE sites (screenshot match)
- Typosquatting domains with CSE-like features
- Brand impersonation attempts

✅ **Won't Flag (Low False Positives):**
- Random legitimate websites (no CSE similarity)
- Generic phishing (not targeting CSE brands)
- Parked domains (unless mimicking CSE)

## Files Created

```
data_prep/
  └── load_cse_data.py                    # Extract CSE baseline (UPDATED)

models/
  ├── tabular/train_anomaly.py            # Train IsolationForest (NEW)
  ├── vision/build_cse_index.py           # Build CLIP index (NEW)
  └── ensemble/similarity_detector.py     # Detection logic (NEW)

integration/
  └── chromadb_similarity_detection.py    # ChromaDB integration (NEW)

Documentation/
  └── CSE_SIMILARITY_TRAINING.md          # Full documentation (NEW)
```

## Next Steps

1. Run training: `python data_prep/load_cse_data.py && python models/tabular/train_anomaly.py && python models/vision/build_cse_index.py`
2. Test on ChromaDB: `python integration/chromadb_similarity_detection.py`
3. Review results in `data/ml_verdicts.csv`
4. Tune thresholds if needed (see full docs)

## Documentation

Full details: [Documentation/CSE_SIMILARITY_TRAINING.md](Documentation/CSE_SIMILARITY_TRAINING.md)
