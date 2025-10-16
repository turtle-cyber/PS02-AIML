"""ChromaDB to ML Pipeline - Similarity-Based Detection"""
import sys
import chromadb
import pandas as pd
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from common.fe import extract_features_from_chromadb
from models.ensemble.similarity_detector import CSESimilarityDetector

def main():
    # Connect to ChromaDB
    client = chromadb.HttpClient(host='localhost', port=8000)
    collection = client.get_collection("domains")

    # Load similarity detector
    print("Loading CSE similarity detector...")
    detector = CSESimilarityDetector(
        anomaly_model_path="models/tabular/anomaly/anomaly_detector.joblib",
        visual_index_dir="models/vision/cse_index"
    )

    # Query domains that need ML verdict
    # Focus on domains with features (not inactive/unregistered)
    results = collection.get(
        where={
            "$and": [
                {"has_features": True},           # Has webpage features
                {"has_verdict": True},            # Has rule-based verdict
                {"final_verdict": {"$in": ["suspicious", "parked"]}}  # Needs ML review
            ]
        },
        include=["metadatas"],
        limit=100
    )

    if not results['ids']:
        print("No records to process")
        return

    print(f"\nProcessing {len(results['ids'])} suspicious domains...")

    verdicts = []

    for idx, (domain_id, metadata) in enumerate(zip(results['ids'], results['metadatas'])):
        # Extract features
        features = extract_features_from_chromadb(metadata)
        domain = metadata.get('registrable', 'unknown')

        # Get screenshot path if available
        # Assuming screenshots are named: {domain}_{hash}_full.png
        screenshot_dir = Path("CSE/out/screenshots")  # Update for suspicious domains
        screenshot_path = None
        # In production, you'd query the actual screenshot location

        # Run similarity detection
        result = detector.predict(
            features=features,
            screenshot_path=screenshot_path,
            domain=domain
        )

        verdicts.append({
            'id': domain_id,
            'domain': domain,
            'verdict': result['verdict'],
            'confidence': result['confidence'],
            'tabular_score': result['tabular_anomaly_score'],
            'visual_similarity': result['visual_similarity'],
            'matched_cse': result['matched_cse_domain'],
            'reasons': result['reasons']
        })

        # Print progress
        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx + 1}/{len(results['ids'])}")

    # Summary
    df_verdicts = pd.DataFrame(verdicts)
    print(f"\n{'='*70}")
    print("RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"\nTotal processed: {len(df_verdicts)}")
    print(f"\nVerdict distribution:")
    print(df_verdicts['verdict'].value_counts())
    print(f"\nAverage confidence: {df_verdicts['confidence'].mean():.2f}")

    # Show high-confidence phishing detections
    phishing = df_verdicts[df_verdicts['verdict'] == 'phishing'].sort_values('confidence', ascending=False)
    if len(phishing) > 0:
        print(f"\n🚨 HIGH-CONFIDENCE PHISHING DETECTED ({len(phishing)} domains):")
        for _, row in phishing.head(10).iterrows():
            print(f"\n  Domain: {row['domain']}")
            print(f"    Verdict: {row['verdict']} (confidence: {row['confidence']:.2f})")
            print(f"    Tabular anomaly: {row['tabular_score']:.3f}")
            if row['visual_similarity']:
                print(f"    Visual similarity: {row['visual_similarity']:.3f} (mimics {row['matched_cse']})")
            print(f"    Reasons: {row['reasons']}")

    # Save results
    output_path = Path("data/ml_verdicts.csv")
    df_verdicts.to_csv(output_path, index=False)
    print(f"\n✅ Saved verdicts to {output_path}")

    # TODO: Update ChromaDB with ML verdicts
    print("\n⚠️  NOTE: ChromaDB update not implemented yet")
    print("    To update ChromaDB, call collection.update() with new metadata")

if __name__ == "__main__":
    main()
