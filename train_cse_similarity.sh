#!/bin/bash
# CSE Similarity-Based Detection Training Pipeline
# This script trains a phishing detector using ONLY legitimate CSE sites as baseline

set -e

echo "=========================================="
echo "CSE Similarity-Based Detection Training"
echo "=========================================="
echo ""

# Step 1: Extract CSE baseline features
echo "[1/3] Extracting CSE baseline features from dump_all.jsonl..."
python data_prep/load_cse_data.py
echo "✅ Step 1 complete"
echo ""

# Step 2: Train tabular anomaly detector
echo "[2/3] Training tabular anomaly detector (IsolationForest)..."
python models/tabular/train_anomaly.py \
    --csv data/cse_benign.csv \
    --outdir models/tabular/anomaly \
    --contamination 0.05
echo "✅ Step 2 complete"
echo ""

# Step 3: Build visual similarity index
echo "[3/3] Building visual similarity index (CLIP embeddings)..."
python models/vision/build_cse_index.py \
    --img_dir CSE/out/screenshots \
    --outdir models/vision/cse_index \
    --model ViT-B-32
echo "✅ Step 3 complete"
echo ""

echo "=========================================="
echo "✅ Training Complete!"
echo "=========================================="
echo ""
echo "Models saved:"
echo "  - Anomaly detector: models/tabular/anomaly/anomaly_detector.joblib"
echo "  - Visual index: models/vision/cse_index/cse_embeddings.npy"
echo ""
echo "Next steps:"
echo "  1. Test on ChromaDB domains:"
echo "     python integration/chromadb_similarity_detection.py"
echo ""
echo "  2. Or test on individual domain:"
echo "     python models/ensemble/similarity_detector.py \\"
echo "       --features_csv test_features.csv \\"
echo "       --screenshot test_screenshot.png \\"
echo "       --domain test-domain.com"
echo ""
