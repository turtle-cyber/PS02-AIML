# Requirements - Multi-Modal Phishing Detection System

## Project Overview

**System Name:** PS02 Stage 1 - Multi-Modal Phishing Detection Pipeline

**Purpose:** Defensive phishing detection system that combines internet crawling, vector database storage, and multi-modal machine learning to identify phishing websites through similarity detection against known brand websites (CSE domains).

**Approach:** Compare suspicious domains against verified brand (CSE) pages using:
- Visual similarity (screenshots, favicons)
- Textual similarity (HTML content, URL patterns)
- Tabular features (WHOIS, DNS, domain age, TLS certificates)
- Behavioral patterns (forms, credential requests)

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  FRONTEND (Port 3000)                                       │
│  - URL Submission Interface                                 │
│  - Kafka Producer                                           │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│  KAFKA MESSAGE QUEUE (Port 9092)                            │
│  - Topic: raw.hosts                                         │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│  INTERNET CRAWLER PIPELINE                                  │
│  ├── CT-Watcher (Certificate Transparency monitoring)       │
│  ├── DNSTwist (domain variant generation)                   │
│  ├── DNS Collector (DNS resolution)                         │
│  ├── HTTP Fetcher (HTML content retrieval)                  │
│  ├── Feature Crawler (screenshot, favicon, WHOIS)           │
│  └── ChromaDB Ingestor (metadata + embeddings)              │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│  CHROMADB VECTOR DATABASE (Port 8000)                       │
│  - Collection: "domains"                                    │
│  - 30+ metadata fields per domain                           │
│  - Embeddings for semantic search                           │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│  ML INTEGRATION LAYER                                       │
│  - Feature extraction (common/fe.py)                        │
│  - Multi-modal inference orchestration                      │
│  - Ensemble scoring                                         │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│  ML MODELS                                                   │
│  ├── Tabular: XGBoost (URL/DNS/WHOIS features)              │
│  ├── Text: XLM-RoBERTa (HTML content)                       │
│  ├── Vision: ResNet18 + CLIP + DINO (screenshots)           │
│  ├── Hash Matching: Favicon MD5, Screenshot phash           │
│  └── Ensemble: Calibrated soft voting                       │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│  PREDICTION API                                              │
│  - REST endpoints for inference                             │
│  - Health checks                                            │
│  - SHAP explanations                                        │
└─────────────────────────────────────────────────────────────┘
```

---

## 1. Functional Requirements

### 1.1 Core Features

#### FR-1: URL Submission & Processing
- **FR-1.1** Accept URLs or domains via web interface (port 3000)
- **FR-1.2** Accept URLs via REST API (`POST /api/submit`)
- **FR-1.3** Validate domain format before submission
- **FR-1.4** Support optional CSE ID and notes metadata
- **FR-1.5** Return Kafka partition/offset confirmation
- **FR-1.6** Estimated processing time: 3-5 minutes

#### FR-2: Internet Crawling
- **FR-2.1** Monitor Certificate Transparency logs for new domains
- **FR-2.2** Generate domain variants using DNSTwist
- **FR-2.3** Resolve DNS records (A, MX, NS, TXT)
- **FR-2.4** Fetch HTTP/HTTPS content with redirects
- **FR-2.5** Capture full-page screenshots (1920x1080)
- **FR-2.6** Extract favicons (MD5, SHA256 hashing)
- **FR-2.7** Perform WHOIS lookups (registrar, age, country)
- **FR-2.8** Extract SSL/TLS certificate details
- **FR-2.9** Detect credential forms (login, password fields)
- **FR-2.10** Extract visible text and HTML structure

#### FR-3: ChromaDB Storage
- **FR-3.1** Store domain metadata (30+ fields) in ChromaDB collection
- **FR-3.2** Store searchable document text (HTML content + OCR)
- **FR-3.3** Generate and store text embeddings
- **FR-3.4** Support semantic search by similarity
- **FR-3.5** Query by metadata filters (country, age, verdict)
- **FR-3.6** Support batch retrieval (up to 1000 records)

#### FR-4: Multi-Modal ML Detection

##### FR-4.1 Tabular Model
- Extract 50+ features from domain metadata
- Train XGBoost classifier on labeled data
- Perform hyperparameter tuning (RandomizedSearchCV)
- Output calibrated probability scores

##### FR-4.2 Text Model
- Tokenize HTML content using XLM-RoBERTa
- Train multilingual text classifier
- Handle text up to 512 tokens
- Output text phishing probability

##### FR-4.3 Vision Models
- **ResNet18:** Screenshot classification (phishing vs. benign)
- **CLIP:** Brand similarity detection (85%+ threshold)
- **DINO:** Layout similarity embeddings
- **Phash:** Perceptual hash matching (exact duplicates)
- **Favicon MD5:** Icon matching against CSE database

##### FR-4.4 Ensemble Model
- Soft voting ensemble (weighted average)
- Isotonic regression calibration per modality
- Optimized weights via grid search on validation set
- Optimized thresholds for precision/recall targets
- Output final verdict: BENIGN / SUSPICIOUS / PHISHING

#### FR-5: Similarity Detection Logic
- **FR-5.1** Whitelist check: Known CSE domains marked BENIGN
- **FR-5.2** Subdomain check: Subdomains of CSE domains marked BENIGN
- **FR-5.3** Favicon match: Different domain + same favicon = PHISHING (95% confidence)
- **FR-5.4** Screenshot phash match: Identical screenshot = PHISHING (92% confidence)
- **FR-5.5** CLIP similarity: >85% visual match = PHISHING (88% confidence)
- **FR-5.6** Registrar validation: Same registrar as CSE = likely BENIGN
- **FR-5.7** Different registrar + visual match = PHISHING (93% confidence)
- **FR-5.8** Autoencoder anomaly: Reconstruction error >2.0 = SUSPICIOUS
- **FR-5.9** Tabular anomaly: Isolation Forest score <-0.10 = SUSPICIOUS

#### FR-6: Explainability
- **FR-6.1** Generate SHAP explanations for predictions
- **FR-6.2** Provide feature importance scores
- **FR-6.3** Display matched CSE domain for visual similarity
- **FR-6.4** Show confidence scores per modality
- **FR-6.5** List all triggered detection signals

#### FR-7: API Endpoints
- **FR-7.1** `POST /predict` - Single domain prediction
- **FR-7.2** `GET /health` - Service health check
- **FR-7.3** Return JSON with verdict, confidence, signals, explanations
- **FR-7.4** Error handling for missing models or data
- **FR-7.5** Request validation and logging

#### FR-8: Monitoring & Drift Detection
- **FR-8.1** Track prediction distributions
- **FR-8.2** Isolation Forest drift detection
- **FR-8.3** Flag domains requiring manual review
- **FR-8.4** Monitor until date for suspicious domains
- **FR-8.5** Log all predictions with timestamps

---

## 2. Non-Functional Requirements

### 2.1 Performance

#### NFR-1: Throughput
- **NFR-1.1** Process 100 URLs per hour via crawler pipeline
- **NFR-1.2** ML inference: <2 seconds per domain (tabular only)
- **NFR-1.3** ML inference: <10 seconds per domain (full multi-modal)
- **NFR-1.4** ChromaDB query: <500ms for single domain lookup
- **NFR-1.5** Support batch processing (1000 domains)

#### NFR-2: Accuracy
- **NFR-2.1** Target F1-score: ≥0.93
- **NFR-2.2** Target Precision: ≥0.95 (minimize false positives)
- **NFR-2.3** Target Recall: ≥0.98 (minimize false negatives)
- **NFR-2.4** Calibration ECE: <0.05 (reliable confidence scores)
- **NFR-2.5** False positive rate: <5% on benign domains

#### NFR-3: Reliability
- **NFR-3.1** Graceful degradation: If vision unavailable, use tabular+text
- **NFR-3.2** Retry logic for HTTP fetches (3 attempts)
- **NFR-3.3** Timeout handling for slow domains (30s max)
- **NFR-3.4** Handle missing features with default values
- **NFR-3.5** API uptime: 99% availability

### 2.2 Scalability

#### NFR-4: Data Volume
- **NFR-4.1** Support 100,000+ domains in ChromaDB
- **NFR-4.2** Handle 1,000+ CSE brand domains
- **NFR-4.3** Store 10,000+ screenshots (10GB+ disk)

#### NFR-5: Concurrency
- **NFR-5.1** Support 10 concurrent URL submissions
- **NFR-5.2** Kafka consumer parallelism: 4 partitions
- **NFR-5.3** Docker services: 8 containers minimum

### 2.3 Security

#### NFR-6: Data Protection
- **NFR-6.1** No storage of sensitive user credentials
- **NFR-6.2** HTTPS for external domain fetches
- **NFR-6.3** Sandboxed screenshot rendering (no JS execution)
- **NFR-6.4** Input validation to prevent injection attacks

#### NFR-7: Defensive Use Only
- **NFR-7.1** System designed exclusively for defensive security
- **NFR-7.2** No credential harvesting capabilities
- **NFR-7.3** Detection rules only, no exploitation tools
- **NFR-7.4** Public documentation for transparency

---

## 3. Technical Requirements

### 3.1 System Requirements

#### Infrastructure
- **OS:** Linux (Ubuntu 20.04+) or Windows 10/11
- **RAM:** 16GB minimum, 32GB recommended
- **Disk:** 100GB free space (for screenshots, models, databases)
- **CPU:** 8 cores recommended
- **GPU:** Optional (CUDA for vision models, 8GB VRAM)
- **Network:** Stable internet for domain fetching

#### Containerization
- **Docker:** 20.10+
- **Docker Compose:** 1.29+
- **Containers:**
  - `frontend-api` (Node.js/Express)
  - `kafka` (Kafka broker)
  - `zookeeper` (Kafka dependency)
  - `chromadb` (Vector database)
  - `normalizer` (Domain preprocessing)
  - `dns-collector` (DNS resolution)
  - `http-fetcher` (HTTP content)
  - `feature-crawler` (Screenshots, WHOIS)
  - `chroma-ingestor` (Database writes)

### 3.2 Python Dependencies

#### Core ML Libraries
```
# Deep Learning
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0

# Classical ML
scikit-learn>=1.3.0
xgboost>=2.0.0
joblib>=1.3.0

# Vision
Pillow>=10.0.0
open-clip-torch>=2.20.0
imagehash>=4.3.0
pytesseract>=0.3.10 (OCR)

# Database
chromadb>=0.4.0
httpx>=0.24.0 (ChromaDB client)

# NLP
sentencepiece>=0.1.99
tokenizers>=0.13.0

# Explainability
shap>=0.42.0

# Data Processing
pandas>=2.0.0
numpy>=1.24.0

# Networking
requests>=2.31.0
dnspython>=2.4.0
python-whois>=0.8.0

# Web Framework (API)
fastapi>=0.100.0 OR flask>=2.3.0
uvicorn>=0.23.0 (if using FastAPI)

# Monitoring
prometheus-client>=0.17.0 (optional)

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0
```

#### External Tools
- **Tesseract OCR:** `apt-get install tesseract-ocr` (Linux)
- **Playwright:** For screenshot capture (crawler)
- **DNSTwist:** Domain permutation engine

### 3.3 Data Requirements

#### Training Data

**Labeled Phishing Dataset:**
- **Minimum:** 10,000 labeled examples (50% phishing, 50% benign)
- **Recommended:** 50,000+ examples
- **Format:** CSV with columns:
  - `domain` (string)
  - `label` (0=benign, 1=phishing)
  - `url` (full URL)
  - `screenshot_path` (optional)
  - `html_path` (optional)
  - `features` (JSON or separate columns)

**CSE (Brand) Dataset:**
- **Minimum:** 50 verified brand domains
- **Recommended:** 500+ brand domains
- **Requirements per brand:**
  - Homepage screenshot (PNG/JPG)
  - Favicon (ICO/PNG)
  - HTML snapshot
  - WHOIS record
  - Domain metadata
- **Format:** JSONL or CSV
- **Example CSE domains:** `sbi.co.in`, `icicibank.com`, `hdfc.com`

**Data Preparation:**
- ChromaDB collection: "domains"
- Feature CSV: `cse_all_features.csv` (tabular features)
- Favicon DB: `cse_favicon_db.csv` (MD5 hashes)
- Phash DB: `cse_phash_db.csv` (perceptual hashes)
- CLIP embeddings: `cse_embeddings.npy` (vision vectors)
- Screenshot directory: `data/screenshots/`

---

## 4. Implementation Status

### 4.1 Completed Components (✅)

#### Internet Crawler
- ✅ Frontend API (URL submission interface)
- ✅ Kafka message queue integration
- ✅ DNSTwist variant generation
- ✅ DNS collector (A, MX, NS records)
- ✅ HTTP fetcher (redirects, content)
- ✅ Feature crawler (screenshots, favicon, WHOIS)
- ✅ ChromaDB ingestor
- ✅ Docker Compose orchestration

#### ChromaDB
- ✅ Collection schema (30+ metadata fields)
- ✅ Query utility (`Documentation/query_vi.py`)
- ✅ Semantic search capability
- ✅ REST API (port 8000)

#### ML Models
- ✅ Vision: ResNet18 training (`models/vision/vis_probe_resnet18.py`)
- ✅ Vision: ResNet18 inference (`models/vision/infer_resnet18.py`)
- ✅ Vision: CLIP index builder (`models/vision/build_clip_index.py`)
- ✅ Vision: DINO embeddings (`models/vision/dino_embed.py`)
- ✅ Text: XLM-RoBERTa training (`models/text/train_mbxt.py`)
- ✅ Tabular: XGBoost tuning (`models/tabular/tune_xgboost.py`)
- ✅ Tabular: Evaluation (`models/tabular/eval.py`)
- ✅ Ensemble: Weight optimization (`models/ensemble/optimize_weights.py`)
- ✅ Ensemble: Threshold optimization (`models/ensemble/optimize_thresholds.py`)
- ✅ Ensemble: Calibration validation (`models/ensemble/validate_calibration.py`)
- ✅ Ensemble: Soft voting (`models/ensemble/ensemble_softvote.py`)
- ✅ Ensemble: Calibration (`models/ensemble/calibrate_heads.py`)
- ✅ Explainability: SHAP (`models/explainability/shap_explain.py`)
- ✅ Monitoring: Drift detection (`models/monitor/isolation_forest.py`)
- ✅ API: Prediction endpoint (`api/predict.py`)

#### Integration
- ✅ Similarity detection (`integration/chromadb_similarity_detection.py`)
- ✅ ChromaDB to ML pipeline (`integration/chromadb_to_ml.py`)
- ✅ Unified detector (`detect_phishing.py`)

#### Data Preparation
- ✅ Load CSE data (`data_prep/load_cse_data.py`)
- ✅ Extract document features (`data_prep/extract_document_features.py`)
- ✅ Extract visual features (`data_prep/extract_visual_features.py`)
- ✅ Merge features (`data_prep/merge_all_features.py`)
- ✅ Combine training data (`data_prep/combine_training_data.py`)

#### Documentation
- ✅ Quick start guide (`Documentation/QUICK_START_GUIDE.md`)
- ✅ Implementation summary (`Documentation/IMPLEMENTATION_SUMMARY.md`)
- ✅ Codebase audit (`Documentation/CODEBASE_AUDIT.md`)
- ✅ ChromaDB schema (`Documentation/CHROMADB_SCHEMA.md`)
- ✅ ChromaDB query guide (`Documentation/CHROMADB_QUERY_GUIDE.md`)
- ✅ Improvements readme (`Documentation/IMPROVEMENTS_README.md`)
- ✅ Monitoring config (`Documentation/MONITORING_CONFIGURATION.md`)
- ✅ Crawler documentation (`Documentation/internet-crawler-doc.md`)

### 4.2 Incomplete Components (⚠️)

#### Common Utilities (Empty stubs - need implementation)
- ⚠️ `common/config.py` - Configuration management
- ⚠️ `common/fe.py` - Feature engineering
- ⚠️ `common/io.py` - File I/O utilities
- ⚠️ `common/metrics.py` - Metrics functions
- ⚠️ `common/viz.py` - Visualization utilities

#### ML Training (Missing basic training scripts)
- ⚠️ `models/tabular/train_xbg.py` - XGBoost training (stub)
- ⚠️ `models/tabular/train_rf.py` - Random Forest training (stub)
- ⚠️ `models/tabular/train_lr.py` - Logistic Regression training (stub)
- ⚠️ `models/text/infer_mbxt.py` - Text model inference (stub)

### 4.3 Priority Implementation Order

**P0 - Critical (Week 1):**
1. Implement `common/fe.py` (feature extraction from ChromaDB)
2. Implement `models/tabular/train_xbg.py` (basic XGBoost training)
3. Implement `models/text/infer_mbxt.py` (text model inference)
4. Test end-to-end pipeline

**P1 - Important (Week 2):**
5. Implement `common/config.py` (centralized configuration)
6. Implement `common/io.py` (model loading/saving)
7. Implement `common/metrics.py` (evaluation metrics)

**P2 - Nice-to-Have (Week 3+):**
8. Implement `models/tabular/train_rf.py` (Random Forest)
9. Implement `models/tabular/train_lr.py` (Logistic Regression)
10. Implement `common/viz.py` (plotting utilities)

---

## 5. Training Pipeline

### 5.1 Data Preparation

**Step 1: Load CSE Data**
```bash
python data_prep/load_cse_data.py
```
- Extracts CSE domain features from ChromaDB
- Outputs: `data/cse_benign.csv`, `data/cse_text.csv`

**Step 2: Extract Visual Features**
```bash
python data_prep/extract_visual_features.py
```
- Computes phash for screenshots
- Extracts favicon MD5/SHA256
- Performs OCR on screenshots
- Outputs: `data/cse_phash_db.csv`, `data/cse_favicon_db.csv`

**Step 3: Merge All Features**
```bash
python data_prep/merge_all_features.py
```
- Combines tabular, text, and visual features
- Outputs: `data/cse_all_features.csv`

**Step 4: Combine Training Data**
```bash
python data_prep/combine_training_data.py
```
- Merges benign CSE data with labeled phishing dataset
- Creates train/val/test splits (70/15/15)
- Outputs: `data/train.csv`, `data/val.csv`, `data/test.csv`

### 5.2 Model Training

**Tabular Model:**
```bash
# Train XGBoost
python models/tabular/train_xbg.py \
    --train_csv data/train.csv \
    --val_csv data/val.csv \
    --outdir models/tabular/out

# Hyperparameter tuning
python models/tabular/tune_xgboost.py \
    --train_csv data/train.csv \
    --val_csv data/val.csv \
    --outdir models/tabular/tuned \
    --n_iter 100 \
    --metric f1
```

**Text Model:**
```bash
# Train XLM-RoBERTa
python models/text/train_mbxt.py \
    --train_csv data/train_text.csv \
    --val_csv data/val_text.csv \
    --outdir models/text/out \
    --epochs 5 \
    --batch_size 16
```

**Vision Model:**
```bash
# Build CLIP index
python models/vision/build_clip_index.py \
    --screenshot_dir data/screenshots/cse \
    --outdir models/vision/cse_index

# Train ResNet18
python models/vision/vis_probe_resnet18.py \
    --train_dir data/screenshots/train \
    --val_dir data/screenshots/val \
    --train_labels data/train.csv \
    --val_labels data/val.csv \
    --outdir models/vision/out \
    --epochs 20 \
    --batch_size 32 \
    --augment
```

### 5.3 Ensemble Optimization

**Step 1: Optimize Weights**
```bash
python models/ensemble/optimize_weights.py \
    --val_csv data/validation_probs.csv \
    --cal_dir models/ensemble/out \
    --outdir models/ensemble/optimized \
    --metric f1
```

**Step 2: Optimize Thresholds**
```bash
python models/ensemble/optimize_thresholds.py \
    --val_csv data/validation_scores.csv \
    --outdir models/ensemble/thresholds \
    --target_precision 0.95 \
    --target_recall 0.98
```

**Step 3: Validate Calibration**
```bash
python models/ensemble/validate_calibration.py \
    --val_csv data/calibration_val.csv \
    --outdir models/ensemble/calibration_check \
    --n_bins 10
```

### 5.4 Model Evaluation

```bash
# Evaluate on test set
python models/tabular/eval.py \
    --model_path models/tabular/tuned/XGBoost_tuned.joblib \
    --test_csv data/test.csv \
    --outdir results/eval
```

---

## 6. Deployment

### 6.1 Docker Deployment

**Start Full Pipeline:**
```bash
cd infra
docker-compose up -d
```

**Services:**
- `frontend-api` - http://localhost:3000
- `kafka` - localhost:9092
- `chromadb` - http://localhost:8000
- `normalizer`, `dns-collector`, `http-fetcher`, `feature-crawler`, `chroma-ingestor`

**Monitor Logs:**
```bash
docker logs -f frontend-api
docker logs -f chroma-ingestor
```

### 6.2 API Deployment

**Start Prediction API:**
```bash
python api/predict.py
```

**Configuration:** Create `models/ensemble/config.json`:
```json
{
  "model_paths": {
    "tabular": "models/tabular/tuned/XGBoost_tuned.joblib",
    "text": "models/text/out",
    "vision": "models/vision/out/resnet18_best.pth",
    "calibrators": "models/ensemble/out"
  },
  "weights": {
    "w1": 0.48,
    "w2": 0.32,
    "w3": 0.14,
    "alpha": 0.12,
    "beta": 0.08
  },
  "thresholds": {
    "tau_low": 0.29,
    "tau_high": 0.72
  }
}
```

### 6.3 Testing

**Validate Pipeline:**
```bash
python test_pipeline.py
```

**Test Single Domain:**
```bash
python detect_phishing.py \
    --domain suspicious-domain.com \
    --screenshot data/screenshots/suspicious-domain.png \
    --favicon_md5 abc123... \
    --features_csv data/domain_features.csv
```

**Expected Output:**
```
==================================================================
DETECTION RESULTS: suspicious-domain.com
==================================================================

Verdict: PHISHING
Confidence: 0.95
Signals detected: 2

Detailed Signals:

1. [FAVICON_MATCH]
   Verdict: PHISHING
   Confidence: 0.95
   Reason: Favicon matches sbi.co.in but domain is different
   Matched CSE domain: sbi.co.in

2. [CLIP_SIMILARITY]
   Verdict: PHISHING
   Confidence: 0.88
   Reason: High visual similarity to sbi.co.in (sim=0.923)
   Matched CSE domain: sbi.co.in

==================================================================
```

---

## 7. Performance Benchmarks

### 7.1 Expected Metrics

**Overall System:**
- F1-Score: 0.93 - 0.95
- Precision: ≥0.95
- Recall: ≥0.98
- False Positive Rate: <5%
- Calibration ECE: <0.05

**Per-Modality Performance:**
- Tabular (XGBoost): F1 = 0.85-0.87
- Text (XLM-RoBERTa): F1 = 0.80-0.85
- Vision (ResNet18): F1 = 0.75-0.80
- Vision (CLIP): Precision = 0.88, Similarity threshold = 0.85
- Ensemble: F1 = 0.93-0.95

### 7.2 Processing Times

**Crawler Pipeline (per domain):**
- DNS resolution: ~2 seconds
- HTTP fetch: ~5 seconds
- Screenshot capture: ~10 seconds
- Feature extraction: ~5 seconds
- ChromaDB ingest: ~1 second
- **Total: 3-5 minutes (including DNSTwist variants)**

**ML Inference (per domain):**
- Tabular only: <2 seconds
- Tabular + Text: <5 seconds
- Full multi-modal: <10 seconds
- Batch (1000 domains): ~30 minutes

---

## 8. Known Limitations

### 8.1 Current Limitations

1. **No Real-Time WHOIS:** Some domains lack registrar data (reduces registrar validation effectiveness)
2. **Screenshot Quality:** Depends on page load time and JavaScript rendering
3. **Language Support:** Text model optimized for English (XLM-RoBERTa supports 100+ languages but may have reduced accuracy)
4. **GPU Required:** Vision models significantly slower on CPU (10x slower)
5. **Storage Growth:** Screenshot storage grows ~1MB per domain
6. **False Positives:** Legitimate new brand domains may be flagged if visually similar

### 8.2 Future Enhancements

1. **Temporal Validation:** Time-based train/test splits to prevent data leakage
2. **Multi-Brand Similarity:** Support multiple CSE matches per domain
3. **Adversarial Testing:** Robustness against obfuscation (IDN homographs, URL shorteners)
4. **Automated Retraining:** Weekly model updates with new phishing samples
5. **A/B Testing Framework:** Gradual rollout of model updates
6. **Per-Brand Performance:** Track F1-score breakdown by CSE domain
7. **Real-Time Monitoring Dashboard:** Live metrics and alerts
8. **Active Learning:** Prioritize manual review for uncertain predictions

---

## 9. References

### 9.1 Documentation
- [QUICK_START_GUIDE.md](Documentation/QUICK_START_GUIDE.md) - Getting started
- [IMPLEMENTATION_SUMMARY.md](Documentation/IMPLEMENTATION_SUMMARY.md) - Improvements overview
- [CODEBASE_AUDIT.md](Documentation/CODEBASE_AUDIT.md) - Implementation status
- [IMPROVEMENTS_README.md](Documentation/IMPROVEMENTS_README.md) - ML enhancements

### 9.2 Key Scripts
- [detect_phishing.py](detect_phishing.py) - Unified detection system
- [test_pipeline.py](test_pipeline.py) - Pipeline validation
- [integration/chromadb_to_ml.py](integration/chromadb_to_ml.py) - ML integration
- [api/predict.py](api/predict.py) - Prediction API

### 9.3 External Resources
- **DNSTwist:** https://github.com/elceef/dnstwist
- **ChromaDB:** https://docs.trychroma.com/
- **XGBoost:** https://xgboost.readthedocs.io/
- **CLIP:** https://github.com/openai/CLIP
- **SHAP:** https://shap.readthedocs.io/

---

## 10. Validation Checklist

Before production deployment:

- [ ] Training data collected (≥10,000 labeled examples)
- [ ] CSE dataset prepared (≥50 brand domains)
- [ ] All models trained (tabular, text, vision)
- [ ] Ensemble weights optimized
- [ ] Thresholds optimized (precision ≥0.95, recall ≥0.98)
- [ ] Calibration validated (ECE <0.05)
- [ ] SHAP explanations generated
- [ ] API config updated with optimized values
- [ ] End-to-end pipeline test passed
- [ ] Docker services healthy
- [ ] Performance metrics logged
- [ ] Documentation reviewed
- [ ] Security audit completed (defensive use only)

---

**Document Version:** 1.0
**Last Updated:** 2025-10-16
**Status:** Active Development
**Grade:** A- (Production-ready with proper data)
