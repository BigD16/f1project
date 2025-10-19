# src/train.py
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from src.features import (
    add_grid_pitlane_features,
    build_baseline_design_matrix,
)
from src.features_rolling import (
    add_domain_rolling_features,
    build_tabular_features,
)


# ------------------------------- Data load/merge -------------------------------

def load_and_merge(data_path: Path) -> pd.DataFrame:
    """
    Load Kaggle F1 CSVs and merge into one row per driver-race with race/team/driver/circuit metadata.
    """
    results      = pd.read_csv(data_path / "results.csv")
    races        = pd.read_csv(data_path / "races.csv")
    drivers      = pd.read_csv(data_path / "drivers.csv")
    constructors = pd.read_csv(data_path / "constructors.csv")
    circuits     = pd.read_csv(data_path / "circuits.csv")

    # Merge in an order that ensures circuitId is available
    df = results.merge(
        races[["raceId", "year", "round", "circuitId", "date"]],
        on="raceId", how="left"
    )
    df = df.merge(
        drivers[["driverId", "dob", "nationality", "code", "forename", "surname"]],
        on="driverId", how="left"
    )
    df = df.merge(
        constructors[["constructorId", "name", "nationality"]],
        on="constructorId", how="left", suffixes=("", "_constructor")
    )
    df = df.merge(
        circuits[["circuitId", "circuitRef", "country"]],
        on="circuitId", how="left", suffixes=("", "_circuit")
    )

    # Parse date & sort chronologically (needed for leakage-safe rolling)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values(["date", "raceId", "driverId"]).reset_index(drop=True)
    return df


# ------------------------------- Splitting utils -------------------------------

def time_split_mask(df: pd.DataFrame, train_end=2015, val_end=2018):
    """Return boolean masks for train/val/test by year."""
    tr = df["year"] <= train_end
    va = (df["year"] > train_end) & (df["year"] <= val_end)
    te = df["year"] > val_end
    return tr, va, te


# ------------------------------- Train/Eval -------------------------------

def fit_eval_logreg(X_tr, y_tr, X_va, y_va, X_te, y_te, title_suffix=""):
    """Fit LogisticRegression and return metrics dicts for train/val/test + a test CM figure."""
    model = LogisticRegression(max_iter=500, class_weight="balanced")
    model.fit(X_tr, y_tr)

    def _metrics(X, y):
        yhat = model.predict(X)
        proba = model.predict_proba(X)[:, 1]
        return dict(
            acc=float(accuracy_score(y, yhat)),
            auc=float(roc_auc_score(y, proba)),
            f1=float(f1_score(y, yhat)),
            report=classification_report(y, yhat, output_dict=True),
            cm=confusion_matrix(y, yhat).tolist(),  # store as list for JSON
        )

    m_tr = _metrics(X_tr, y_tr)
    m_va = _metrics(X_va, y_va)
    m_te = _metrics(X_te, y_te)

    # Plot confusion matrix for TEST
    cm = confusion_matrix(y_te, model.predict(X_te))
    fig = plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Non-Podium", "Podium"],
                yticklabels=["Non-Podium", "Podium"])
    plt.title(f"Confusion Matrix — Test (LogReg) {title_suffix}")
    plt.ylabel("True"); plt.xlabel("Pred"); plt.tight_layout()

    return model, {"train": m_tr, "val": m_va, "test": m_te}, fig


# ------------------------------- Main -------------------------------

def main():
    ap = argparse.ArgumentParser(description="F1 podium prediction: baseline & rolling features with time split.")
    ap.add_argument("--data_path", required=True, help="Folder with Kaggle CSVs (results.csv, races.csv, ...)")
    ap.add_argument("--out_dir", default="models", help="Where to save models/metrics/figures")
    ap.add_argument("--split", choices=["random", "time"], default="time")
    ap.add_argument("--train_end_year", type=int, default=2015)
    ap.add_argument("--val_end_year", type=int, default=2018)
    args = ap.parse_args()

    data_path = Path(args.data_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading & merging…")
    df = load_and_merge(data_path)

    print("Applying baseline grid/pit-lane features…")
    df = add_grid_pitlane_features(df)

    print("Building baseline design matrix…")
    X_base, y_base = build_baseline_design_matrix(df)

    print("Adding domain rolling features (leakage-safe)…")
    df_roll = add_domain_rolling_features(df)

    print("Building rich (rolling) design matrix…")
    X_roll, y_roll = build_tabular_features(df_roll)

    # ---------- Split ----------
    if args.split == "time":
        tr_m, va_m, te_m = time_split_mask(df, args.train_end_year, args.val_end_year)

        Xb_tr, yb_tr = X_base[tr_m], y_base[tr_m]
        Xb_va, yb_va = X_base[va_m], y_base[va_m]
        Xb_te, yb_te = X_base[te_m], y_base[te_m]

        Xr_tr, yr_tr = X_roll[tr_m], y_roll[tr_m]
        Xr_va, yr_va = X_roll[va_m], y_roll[va_m]
        Xr_te, yr_te = X_roll[te_m], y_roll[te_m]
    else:
        # Random split fallback (kept minimal; primary path is time split)
        from sklearn.model_selection import train_test_split
        Xb_tr, Xb_te, yb_tr, yb_te = train_test_split(X_base, y_base, test_size=0.2, random_state=42, stratify=y_base)
        # Use the test split also as "val" for simplicity here
        Xb_va, yb_va = Xb_te, yb_te

        Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(X_roll, y_roll, test_size=0.2, random_state=42, stratify=y_roll)
        Xr_va, yr_va = Xr_te, yr_te

    # ---------- Train & evaluate ----------
    print("Training baseline (logreg)…")
    base_model, base_metrics, base_fig = fit_eval_logreg(Xb_tr, yb_tr, Xb_va, yb_va, Xb_te, yb_te, title_suffix="(baseline)")

    print("Training with rolling features (logreg)…")
    roll_model, roll_metrics, roll_fig = fit_eval_logreg(Xr_tr, yr_tr, Xr_va, yr_va, Xr_te, yr_te, title_suffix="(+rolling)")

    # ---------- Save ----------
    import joblib
    joblib.dump({"model": base_model, "columns": X_base.columns.tolist()}, out_dir / "logreg_baseline.joblib")
    joblib.dump({"model": roll_model, "columns": X_roll.columns.tolist()}, out_dir / "logreg_rolling.joblib")

    with open(out_dir / "metrics_logreg_baseline.json", "w") as f:
        json.dump(base_metrics, f, indent=2)
    with open(out_dir / "metrics_logreg_rolling.json", "w") as f:
        json.dump(roll_metrics, f, indent=2)

    base_fig.savefig(out_dir / "cm_test_logreg_baseline.png", dpi=150)
    roll_fig.savefig(out_dir / "cm_test_logreg_rolling.png", dpi=150)

    print(f"Done. Artifacts saved in: {out_dir}")


if __name__ == "__main__":
    main()
