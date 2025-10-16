"""XGBoost with Text Features (TF-IDF)"""
import argparse, json, pandas as pd, joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from scipy.sparse import hstack
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--text_cols", default="document_text,ocr_text")
    ap.add_argument("--max_features", type=int, default=500)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load
    df = pd.read_csv(args.csv)
    text_cols = [c.strip() for c in args.text_cols.split(',') if c.strip() in df.columns]

    # Labels (use doc_verdict if available, else create dummy)
    if 'label' in df.columns:
        y = df['label'].map({'benign':0, 'phishing':1})
    elif 'doc_verdict' in df.columns:
        y = df['doc_verdict'].map({'benign':0, 'phishing':1})
    else:
        # No labels - this is CSE baseline only
        print("WARNING: No labels found. This appears to be benign-only CSE data.")
        print("For similarity detection, use isolation forest instead (train_anomaly.py)")
        return

    # Numeric features
    drop_cols = ['url', 'domain', 'label', 'registrable'] + text_cols
    drop_cols = [c for c in drop_cols if c in df.columns]
    X_numeric = df.drop(columns=drop_cols)

    # Convert non-numeric
    for c in X_numeric.columns:
        if not pd.api.types.is_numeric_dtype(X_numeric[c]):
            X_numeric[c] = X_numeric[c].astype('category').cat.codes

    # Text features
    X_text = df[text_cols].fillna('')
    X_text_combined = X_text.apply(lambda row: ' '.join(row.values.astype(str)), axis=1)

    # Split
    X_num_train, X_num_val, X_txt_train, X_txt_val, y_train, y_val = train_test_split(
        X_numeric, X_text_combined, y, test_size=0.2, random_state=42
    )

    # TF-IDF
    tfidf = TfidfVectorizer(max_features=args.max_features, min_df=2, ngram_range=(1,2))
    X_txt_train_vec = tfidf.fit_transform(X_txt_train)
    X_txt_val_vec = tfidf.transform(X_txt_val)

    # Numeric imputer
    imputer = SimpleImputer(strategy='median')
    X_num_train_imp = imputer.fit_transform(X_num_train.values)
    X_num_val_imp = imputer.transform(X_num_val.values)

    # Combine
    X_train_combined = hstack([X_num_train_imp, X_txt_train_vec])
    X_val_combined = hstack([X_num_val_imp, X_txt_val_vec])

    # Train
    clf = XGBClassifier(n_estimators=600, max_depth=6, learning_rate=0.05,
                       subsample=0.8, colsample_bytree=0.8, random_state=42)
    clf.fit(X_train_combined, y_train.values)

    # Evaluate
    y_pred = clf.predict(X_val_combined)
    y_proba = clf.predict_proba(X_val_combined)[:,1]

    metrics = {
        'accuracy': float(accuracy_score(y_val, y_pred)),
        'precision': float(precision_score(y_val, y_pred)),
        'recall': float(recall_score(y_val, y_pred)),
        'f1': float(f1_score(y_val, y_pred)),
        'auc_roc': float(roc_auc_score(y_val, y_proba))
    }

    # Save
    model_bundle = {
        'clf': clf,
        'tfidf': tfidf,
        'imputer': imputer,
        'text_cols': text_cols,
        'numeric_cols': X_numeric.columns.tolist()
    }
    joblib.dump(model_bundle, outdir / "XGBoost_with_text.joblib")
    (outdir / "metrics_text.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
