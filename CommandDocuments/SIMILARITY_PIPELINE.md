# Similarity-Based Phishing Detection Pipeline

## Overview
Uses **benign CSE baseline data** to detect phishing through anomaly and similarity detection.

## Training Pipeline

### 1. Train Anomaly Detector
```bash
python models/tabular/train_anomaly.py --csv data/cse_benign_numeric.csv --outdir models/saved/anomaly
```

### 2. Build CLIP Visual Index
```bash
python models/vision/build_clip_index.py
```

### 3. (Optional) Train Visual Autoencoder
```bash
python models/vision/train_cse_autoencoder.py
```

## Live Detection Pipeline

### Start API Server
```bash
python api/predict.py
```

### API Endpoints

**Health Check:**
```bash
curl http://localhost:8000/health
```

**Scan URL:**
```bash
curl -X POST http://localhost:8000/scan \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://example.com",
    "domain": "example.com",
    "features": {...},
    "favicon_hash": "abc123",
    "phash": "def456",
    "registrar": "GoDaddy"
  }'
```

**Response:**
```json
{
  "url": "https://example.com",
  "label": "benign|suspicious|phishing",
  "confidence": 0.85,
  "risk_score": 0.23,
  "signals": {
    "anomaly_score": 0.1,
    "favicon_match": 0.0,
    "phash_match": 0.0,
    "visual_similarity": 0.0
  },
  "matched_cse_domain": null,
  "reasons": ["No anomalies detected"]
}
```

## Detection Signals

1. **Tabular Anomaly** - Features deviate from CSE baseline
2. **Favicon Match** - Favicon hash matches CSE domain but URL differs
3. **Phash Match** - Screenshot visually similar to CSE but URL differs
4. **Visual Similarity** - CLIP embeddings match CSE domain (TODO)

## Integration with Crawler

```python
# In your crawler
from api.predict import ScanRequest
import requests

# Extract features from page
features = extract_features(url)
favicon_hash = hash_favicon(favicon)
phash = compute_phash(screenshot)

# Send to API
response = requests.post("http://localhost:8000/scan", json={
    "url": url,
    "domain": domain,
    "features": features,
    "favicon_hash": favicon_hash,
    "phash": phash,
    "registrar": whois_data.get("registrar")
})

verdict = response.json()
# Store in ChromaDB
```

## Model Files Required

- `models/saved/anomaly/anomaly_detector.joblib`
- `models/vision/cse_index/cse_embeddings.npy`
- `models/vision/cse_index/cse_metadata.json`
- `data/cse_favicon_db.csv`
- `data/cse_phash_db.csv`
