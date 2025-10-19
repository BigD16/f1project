import argparse, json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from src.features import add_grid_pitlane_features, build_baseline_design_matrix

def load_and_merge(data_path: Path) -> pd.DataFrame:
    """
    Load Kaggle F1 CSVs and merge into a single dataframe:
    each row = one driver's result in one race.
    """
    results      = pd.read_csv(data_path / "results.csv")
    races        = pd.read_csv(data_path / "races.csv")
    drivers      = pd.read_csv(data_path / "drivers.csv")
    constructors = pd.read_csv(data_path / "constructors.csv")
    circuits     = pd.read_csv(data_path / "circuits.csv")

    # Merge in correct order so circuitId exists
    df = results.merge(
        races[["raceId","year","round","circuitId","date"]],
        on="raceId", how="left"
    )
    df = df.merge(
        drivers[["driverId","dob","nationality","code","forename","surname"]],
        on="driverId", how="left"
    )
    df = df.merge(
        constructors[["constructorId","name","nationality"]],
        on="constructorId", how="left", suffixes=("", "_constructor")
    )
    df = df.merge(
        circuits[["circuitId","circuitRef","country"]],
        on="circuitId", how="left", suffixes=("", "_circuit")
    )

    # Dates & sort
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values(["date","raceId","driverId"]).reset_index(drop=True)
    return df

def train_baseline(X: pd.DataFrame, y: pd.Series, random_state: int = 42):
    """
    Train & evaluate logistic regression baseline with class_weight='balanced'.
    """
    X, y = shuffle(X, y, random_state=random_state)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.20, random_state=random_state, stratify=y
    )

    model = LogisticRegression(max_iter=500, class_weight="balanced")
    model.fit(Xtr, ytr)

    yhat = model.predict(Xte)
    p = model.predict_proba(Xte)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(yte, yhat)),
        "roc_auc": float(roc_auc_score(yte, p)),
        "f1": float(f1_score(yte, yhat)),
        "classification_report": classification_report(yte, yhat, output_dict=True)
    }

    cm = confusion_matrix(yte, yhat)
    fig = plt.figure(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Non-Podium","Podium"], yticklabels=["Non-Podium","Podium"])
    plt.title("Confusion Matrix — Baseline (LogReg)")
    plt.ylabel("True"); plt.xlabel("Pred"); plt.tight_layout()
    return model, metrics, fig

def main():
    ap = argparse.ArgumentParser(description="Train F1 podium baseline (logistic regression).")
    ap.add_argument("--data_path", required=True, help="Folder containing F1 CSVs.")
    ap.add_argument("--out_dir", default="models", help="Where to save model/metrics/figures.")
    ap.add_argument("--random_state", type=int, default=42)
    args = ap.parse_args()

    data_path = Path(args.data_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading & merging…")
    df = load_and_merge(data_path)

    print("Applying grid/pit-lane features…")
    df = add_grid_pitlane_features(df)

    print("Building design matrix…")
    X, y = build_baseline_design_matrix(df)

    print("Training…")
    model, metrics, fig = train_baseline(X, y, random_state=args.random_state)

    # Save artifacts
    import joblib
    joblib.dump({"model": model, "columns": X.columns.tolist()}, out_dir / "baseline_logreg.joblib")

    with open(out_dir / "metrics_baseline.json", "w") as f:
        json.dump(metrics, f, indent=2)

    fig.savefig(out_dir / "confusion_matrix_baseline.png", dpi=150)

    print("Done. Saved to:", out_dir)

if __name__ == "__main__":
    main()
