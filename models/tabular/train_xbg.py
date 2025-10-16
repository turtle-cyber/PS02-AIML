"""XGBoost training script"""
import argparse, json, pandas as pd, joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load data
    df = pd.read_csv(args.csv)
    drop_cols = [c for c in ['url', 'domain', 'label'] if c in df.columns]
    y = df['label'].map({'benign':0, 'phishing':1})
    X = df.drop(columns=drop_cols)

    # Convert non-numeric
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].astype('category').cat.codes

    # Split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train
    model = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('clf', XGBClassifier(n_estimators=600, max_depth=6, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8, random_state=42))
    ])
    model.fit(X_train.values, y_train.values)

    # Evaluate
    y_pred = model.predict(X_val.values)
    y_proba = model.predict_proba(X_val.values)[:,1]

    metrics = {
        'accuracy': float(accuracy_score(y_val, y_pred)),
        'precision': float(precision_score(y_val, y_pred)),
        'recall': float(recall_score(y_val, y_pred)),
        'f1': float(f1_score(y_val, y_pred)),
        'auc_roc': float(roc_auc_score(y_val, y_proba))
    }

    # Save
    joblib.dump(model, outdir / "XGBoost.joblib")
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
