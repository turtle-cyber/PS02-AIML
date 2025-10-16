"""ChromaDB to ML Pipeline Integration"""
import sys, chromadb, pandas as pd, joblib
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from common.fe import extract_features_from_chromadb

def main():
    # Connect to ChromaDB
    client = chromadb.HttpClient(host='localhost', port=8000)
    collection = client.get_collection("domains")

    # Query records without ML verdict
    results = collection.get(
        where={"has_verdict": False},
        include=["metadatas", "documents"],
        limit=1000
    )

    if not results['ids']:
        print("No records to process")
        return

    # Extract features
    features = [extract_features_from_chromadb(m) for m in results['metadatas']]
    df = pd.DataFrame(features)

    # Load tabular model
    model = joblib.load("models/tabular/out/XGBoost.joblib")
    probs = model.predict_proba(df.values)[:,1]

    # TODO: Add text/vision models, ensemble, update ChromaDB
    print(f"Processed {len(results['ids'])} records")
    for i, (id, prob) in enumerate(zip(results['ids'], probs)):
        print(f"{id}: {prob:.4f}")

if __name__ == "__main__":
    main()
