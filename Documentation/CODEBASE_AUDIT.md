# Complete Codebase Audit - What Exists vs What's Missing

## 📋 Audit Date: January 2025

This document provides a comprehensive audit of the ps02_stage1 repository, identifying what exists, what's implemented, what's empty (stub files), and what needs to be done according to your phishing detection needs.

---

## 🔍 Repository Structure

```
ps02_stage1/
├── api/                       # Prediction API
├── common/                    # Shared utilities
├── models/
│   ├── ensemble/             # Ensemble methods
│   ├── explainability/       # SHAP explanations
│   ├── monitor/              # Drift detection
│   ├── tabular/              # Tabular models (XGBoost, RF, LR)
│   ├── text/                 # Text models (XLM-RoBERTa)
│   └── vision/               # Vision models (ResNet18, CLIP, DINO)
├── query_vi.py               # ChromaDB query utility
└── docs/                     # Documentation
```

---

## ✅ IMPLEMENTED & WORKING

### **API Layer**
| File | Status | Lines | Description |
|------|--------|-------|-------------|
| `api/predict.py` | ✅ COMPLETE | 500 | Production API with error handling, logging, health checks |

### **Ensemble Models**
| File | Status | Lines | Description |
|------|--------|-------|-------------|
| `models/ensemble/calibrate_heads.py` | ✅ COMPLETE | 26 | Isotonic regression calibration |
| `models/ensemble/ensemble_softvote.py` | ✅ COMPLETE | 39 | Soft voting ensemble |
| `models/ensemble/optimize_weights.py` | ✅ COMPLETE | 280 | Grid search weight optimization |
| `models/ensemble/optimize_thresholds.py` | ✅ COMPLETE | 400 | PR curve threshold optimization |
| `models/ensemble/validate_calibration.py` | ✅ COMPLETE | 400 | Calibration validation with ECE |

### **Explainability**
| File | Status | Lines | Description |
|------|--------|-------|-------------|
| `models/explainability/shap_explain.py` | ✅ COMPLETE | 350 | SHAP explanations for predictions |

### **Monitoring**
| File | Status | Lines | Description |
|------|--------|-------|-------------|
| `models/monitor/isolation_forest.py` | ✅ COMPLETE | 22 | Drift detection with Isolation Forest |

### **Tabular Models**
| File | Status | Lines | Description |
|------|--------|-------|-------------|
| `models/tabular/eval.py` | ✅ COMPLETE | 71 | Model evaluation with cross-validation |
| `models/tabular/tune_xgboost.py` | ✅ COMPLETE | 250 | XGBoost hyperparameter tuning |

### **Text Models**
| File | Status | Lines | Description |
|------|--------|-------|-------------|
| `models/text/train_mbxt.py` | ✅ COMPLETE | 68 | XLM-RoBERTa training script |
| `models/text/infer_mbxt.py` | ❌ EMPTY | 1 | **NEEDS IMPLEMENTATION** |

### **Vision Models**
| File | Status | Lines | Description |
|------|--------|-------|-------------|
| `models/vision/vis_probe_resnet18.py` | ✅ COMPLETE | 350 | ResNet18 training for screenshots |
| `models/vision/infer_resnet18.py` | ✅ COMPLETE | 120 | ResNet18 inference |
| `models/vision/build_clip_index.py` | ✅ COMPLETE | 47 | CLIP brand similarity |
| `models/vision/dino_embed.py` | ✅ COMPLETE | 39 | DINO layout similarity |

### **Utilities**
| File | Status | Lines | Description |
|------|--------|-------|-------------|
| `query_vi.py` | ✅ COMPLETE | 420 | ChromaDB query tool with advanced filtering |

---

## ❌ EMPTY/STUB FILES (NEED IMPLEMENTATION)

### **Common Utilities** - ⚠️ **CRITICAL GAP**
| File | Status | Purpose | Priority |
|------|--------|---------|----------|
| `common/config.py` | ❌ EMPTY (1 line) | Shared configuration management | **P0** |
| `common/fe.py` | ❌ EMPTY (1 line) | Feature extraction/engineering | **P0** |
| `common/io.py` | ❌ EMPTY (1 line) | File I/O utilities | **P1** |
| `common/metrics.py` | ❌ EMPTY (1 line) | Shared metrics functions | **P1** |
| `common/viz.py` | ❌ EMPTY (1 line) | Visualization utilities | **P2** |

### **Tabular Training Scripts** - ⚠️ **MISSING**
| File | Status | Purpose | Priority |
|------|--------|---------|----------|
| `models/tabular/train_xbg.py` | ❌ EMPTY (1 line) | XGBoost training | **P0** |
| `models/tabular/train_rf.py` | ❌ EMPTY (1 line) | Random Forest training | **P1** |
| `models/tabular/train_lr.py` | ❌ EMPTY (1 line) | Logistic Regression training | **P2** |

### **Text Inference** - ⚠️ **MISSING**
| File | Status | Purpose | Priority |
|------|--------|---------|----------|
| `models/text/infer_mbxt.py` | ❌ EMPTY (1 line) | XLM-RoBERTa inference | **P0** |

---

## 🔴 CRITICAL MISSING COMPONENTS

Based on your need to connect the **Internet Crawler (ChromaDB)** with your **ML Models**, here are the missing pieces:

### **1. ChromaDB → ML Integration Layer** - ❌ **MISSING**
**Problem:** No code to extract data from ChromaDB and feed it to ML models

**What's needed:**
- Script to query ChromaDB for suspicious pages
- Convert ChromaDB metadata → tabular feature vector
- Extract text for text model
- Load screenshots for vision model
- Orchestrate all modalities

**Priority:** **P0 - CRITICAL**

### **2. Feature Engineering Pipeline** - ❌ **EMPTY**
**Problem:** `common/fe.py` is empty

**What's needed:**
- Extract URL features from ChromaDB metadata
- Extract HTML features from document text
- Feature alignment with trained models
- Handle missing features gracefully

**Priority:** **P0 - CRITICAL**

### **3. Basic Tabular Model Training** - ❌ **EMPTY**
**Problem:** `train_xbg.py`, `train_rf.py`, `train_lr.py` are all empty

**What's needed:**
- Basic training scripts for XGBoost, RF, LR
- Note: `eval.py` exists (cross-validation)
- Note: `tune_xgboost.py` exists (hyperparameter tuning)
- But basic training is missing!

**Priority:** **P0 - CRITICAL**

### **4. Text Model Inference** - ❌ **EMPTY**
**Problem:** `infer_mbxt.py` is empty

**What's needed:**
- Load trained XLM-RoBERTa model
- Tokenize input text
- Generate phishing probability

**Priority:** **P0 - CRITICAL**

### **5. End-to-End Pipeline Orchestration** - ❌ **MISSING**
**Problem:** No script to tie everything together

**What's needed:**
```
ChromaDB Query → Feature Extraction → Multi-Modal Inference → Ensemble → Update ChromaDB
```

**Priority:** **P0 - CRITICAL**

---

## 📊 Completion Status

### **Overall Repository Status**
- **Total Python Files:** 25
- **Fully Implemented:** 15 (60%)
- **Empty/Stub Files:** 8 (32%)
- **Missing Critical Files:** 2 (8%)

### **By Component**
| Component | Implemented | Empty/Missing | Completion |
|-----------|-------------|---------------|------------|
| **API** | 1/1 | 0/1 | 100% ✅ |
| **Common Utils** | 0/5 | 5/5 | 0% ❌ |
| **Ensemble** | 5/5 | 0/5 | 100% ✅ |
| **Explainability** | 1/1 | 0/1 | 100% ✅ |
| **Monitor** | 1/1 | 0/1 | 100% ✅ |
| **Tabular** | 2/5 | 3/5 | 40% ⚠️ |
| **Text** | 1/2 | 1/2 | 50% ⚠️ |
| **Vision** | 4/4 | 0/4 | 100% ✅ |
| **Integration** | 0/1 | 1/1 | 0% ❌ |

---

## 🎯 WHAT YOU NEED TO DO NOW

Based on your **actual need** (connecting crawler → ML models), here's the priority order:

### **Phase 1: Critical Foundation (Week 1)**

#### **1. Implement Feature Engineering** (`common/fe.py`) - **P0**
```python
# Extract features from ChromaDB metadata
def extract_url_features(metadata: dict) -> dict:
    """Convert ChromaDB record → tabular features"""
    pass

def extract_text_features(document: str) -> str:
    """Extract text for text model"""
    pass
```

#### **2. Implement Basic Tabular Training** (`train_xbg.py`) - **P0**
```python
# Train XGBoost on labeled data
python models/tabular/train_xbg.py --train_csv data/train.csv --outdir models/tabular/out
```

#### **3. Implement Text Inference** (`infer_mbxt.py`) - **P0**
```python
# Generate text probabilities
python models/text/infer_mbxt.py --model_dir models/text/out --text_csv data/text.csv --out_csv probs.csv
```

#### **4. Create Integration Script** - **P0 NEW**
```python
# models/integration/chromadb_to_ml.py
# Query ChromaDB → Extract features → Run inference → Update ChromaDB
```

### **Phase 2: Complete Utilities (Week 2)**

#### **5. Implement Config Management** (`common/config.py`) - **P1**
```python
# Centralized configuration
class Config:
    MODEL_PATHS = {...}
    FEATURE_COLS = [...]
    THRESHOLDS = {...}
```

#### **6. Implement I/O Utilities** (`common/io.py`) - **P1**
```python
# File loading/saving helpers
def load_model(path): ...
def save_predictions(df, path): ...
```

#### **7. Implement Metrics** (`common/metrics.py`) - **P1**
```python
# Shared metrics functions
def compute_metrics(y_true, y_pred): ...
```

### **Phase 3: Nice-to-Have (Week 3+)**

#### **8. Implement RF/LR Training** (`train_rf.py`, `train_lr.py`) - **P2**
- For comparison with XGBoost

#### **9. Implement Visualization** (`common/viz.py`) - **P2**
```python
# Plotting helpers
def plot_confusion_matrix(...): ...
def plot_feature_importance(...): ...
```

---

## 🚀 RECOMMENDED IMPLEMENTATION ORDER

Here's the exact sequence to get your system working:

### **Step 1: Feature Engineering** (Day 1-2)
```bash
# Implement common/fe.py
# Test: Extract features from sample ChromaDB record
```

### **Step 2: Tabular Training** (Day 3-4)
```bash
# Implement models/tabular/train_xbg.py
# Train XGBoost on labeled phishing data
python models/tabular/train_xbg.py --train_csv data/train_features.csv --outdir models/tabular/out
```

### **Step 3: Text Inference** (Day 5)
```bash
# Implement models/text/infer_mbxt.py
# Generate text probabilities
python models/text/infer_mbxt.py --model_dir models/text/out --text_csv data/text.csv --out_csv text_probs.csv
```

### **Step 4: Integration Layer** (Day 6-7)
```bash
# Create models/integration/chromadb_to_ml.py
# Connect all pieces: ChromaDB → Features → Models → Predictions
python models/integration/chromadb_to_ml.py --chromadb_host localhost --outdir predictions/
```

### **Step 5: End-to-End Test** (Day 8)
```bash
# Test full pipeline
1. Crawler ingests phishing page → ChromaDB
2. Integration script extracts → Runs ML models
3. Updates ChromaDB with ML verdict
```

---

## 📝 IMPLEMENTATION TEMPLATES

### **Template 1: Feature Engineering (`common/fe.py`)**
```python
import pandas as pd
from typing import Dict, List

# Feature columns expected by trained XGBoost model
REQUIRED_FEATURES = [
    'url_length', 'url_entropy', 'num_subdomains', 'has_ip',
    'domain_age_days', 'is_newly_registered', 'is_self_signed',
    'has_credential_form', 'keyword_count', 'form_count',
    # ... add all features your XGBoost was trained on
]

def extract_features_from_chromadb(metadata: Dict) -> Dict:
    """
    Convert ChromaDB metadata to feature dictionary

    Args:
        metadata: ChromaDB record metadata (30+ fields)

    Returns:
        features: Dict with values for REQUIRED_FEATURES
    """
    features = {}

    # Map ChromaDB fields → model features
    features['url_length'] = metadata.get('url_length', 0)
    features['url_entropy'] = metadata.get('url_entropy', 0.0)
    features['num_subdomains'] = metadata.get('num_subdomains', 0)
    features['domain_age_days'] = metadata.get('domain_age_days', 0)
    features['is_newly_registered'] = int(metadata.get('is_newly_registered', False))
    features['is_self_signed'] = int(metadata.get('is_self_signed', False))
    features['has_credential_form'] = int(metadata.get('has_credential_form', False))
    features['keyword_count'] = metadata.get('keyword_count', 0)
    features['form_count'] = metadata.get('form_count', 0)
    # ... map all required features

    # Fill missing with defaults
    for col in REQUIRED_FEATURES:
        if col not in features:
            features[col] = 0

    return features

def extract_text_from_chromadb(document: str, metadata: Dict) -> str:
    """
    Extract text for text model (URL + HTML content)

    Args:
        document: ChromaDB document field (searchable text)
        metadata: Metadata with URL

    Returns:
        text: Combined text for XLM-RoBERTa
    """
    url = metadata.get('url', '')

    # Combine URL + document text (truncate to model max_length)
    text = f"{url} {document}"[:512]

    return text
```

### **Template 2: Integration Script**
```python
# models/integration/chromadb_to_ml.py
import chromadb
import pandas as pd
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from common.fe import extract_features_from_chromadb, extract_text_from_chromadb
import joblib

def process_chromadb_records():
    # 1. Connect to ChromaDB
    client = chromadb.HttpClient(host='localhost', port=8000)
    collection = client.get_collection("domains")

    # 2. Query for records needing ML scoring
    results = collection.get(
        where={"has_verdict": False},  # No ML verdict yet
        include=["metadatas", "documents"],
        limit=1000
    )

    if not results['ids']:
        print("No records to process")
        return

    # 3. Extract features for tabular model
    features_list = []
    for metadata in results['metadatas']:
        features = extract_features_from_chromadb(metadata)
        features_list.append(features)

    features_df = pd.DataFrame(features_list)

    # 4. Run tabular model
    tabular_model = joblib.load("models/tabular/out/XGBoost.joblib")
    tabular_probs = tabular_model.predict_proba(features_df.values)[:, 1]

    # 5. Run text model (if available)
    # TODO: Implement text inference

    # 6. Run vision model (if screenshots available)
    # TODO: Implement vision inference

    # 7. Run ensemble
    # TODO: Implement ensemble scoring

    # 8. Update ChromaDB with ML verdicts
    # TODO: Implement ChromaDB update

    print(f"Processed {len(results['ids'])} records")

if __name__ == "__main__":
    process_chromadb_records()
```

---

## ✅ ACTION ITEMS SUMMARY

### **MUST DO (Week 1) - Critical for your system to work:**
- [ ] Implement `common/fe.py` (feature engineering)
- [ ] Implement `models/tabular/train_xbg.py` (basic XGBoost training)
- [ ] Implement `models/text/infer_mbxt.py` (text model inference)
- [ ] Create `models/integration/chromadb_to_ml.py` (integration layer)
- [ ] Test end-to-end: Crawler → ChromaDB → ML → Updated ChromaDB

### **SHOULD DO (Week 2) - For production:**
- [ ] Implement `common/config.py` (configuration management)
- [ ] Implement `common/io.py` (I/O utilities)
- [ ] Implement `common/metrics.py` (metrics functions)
- [ ] Create end-to-end orchestration script

### **NICE TO HAVE (Week 3+) - For comparison:**
- [ ] Implement `models/tabular/train_rf.py` (Random Forest)
- [ ] Implement `models/tabular/train_lr.py` (Logistic Regression)
- [ ] Implement `common/viz.py` (visualization utilities)

---

## 📊 Visual Summary

```
YOUR SYSTEM NEEDS:
┌─────────────────────────────────────────────────────────────────┐
│  Internet Crawler (✅ COMPLETE)                                 │
│  ├── CT-Watcher ✅                                              │
│  ├── DNSTwist ✅                                                │
│  ├── Feature Crawler ✅                                         │
│  └── ChromaDB ✅                                                │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
          ┌────────────────────────┐
          │  ❌ MISSING GAP ❌     │  ← YOU ARE HERE
          │  Integration Layer     │
          │  (needs implementation)│
          └────────────┬───────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│  ML Models (⚠️ 60% COMPLETE)                                    │
│  ├── Tabular: tune_xgboost ✅, eval ✅, train_xbg ❌           │
│  ├── Text: train ✅, infer ❌                                   │
│  ├── Vision: train ✅, infer ✅, CLIP ✅, DINO ✅              │
│  ├── Ensemble: weights ✅, thresholds ✅, calibration ✅       │
│  └── Common utils: ❌ ALL EMPTY                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🎯 Bottom Line

**You have:**
- ✅ Excellent crawler with ChromaDB
- ✅ Advanced ML optimization tools (my implementations)
- ✅ Solid model architecture (text, vision, ensemble)

**You're missing:**
- ❌ Feature engineering (`common/fe.py`)
- ❌ Basic model training (`train_xbg.py`)
- ❌ Text model inference (`infer_mbxt.py`)
- ❌ Integration layer (ChromaDB → ML)
- ❌ End-to-end orchestration

**Priority:** Implement the 4 missing P0 items to connect crawler → models → predictions.

**Estimated Time:** 1-2 weeks for critical path.

---

**Last Updated:** January 2025
**Next Review:** After implementing P0 items
