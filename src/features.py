import numpy as np
import pandas as pd

def add_grid_pitlane_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds:
      - pitlane_start: 1 if grid == 0 (pit lane), else 0
      - grid_clean: replaces grid==0 with (per-race max grid + 1) so lower numbers remain better
    """
    out = df.copy()
    out["pitlane_start"] = (out["grid"] == 0).astype(int)
    race_max_grid = out.groupby("raceId")["grid"].transform("max").fillna(20)
    out["grid_clean"] = np.where(out["grid"] == 0, race_max_grid + 1, out["grid"])
    return out

def build_baseline_design_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Baseline features:
      - grid_clean (numeric)
      - pitlane_start (binary)
      - constructorId (one-hot)
    Target:
      - podium: 1 if positionOrder <= 3 else 0
    """
    y = (df["positionOrder"] <= 3).astype(int)
    features = pd.DataFrame({
        "grid_clean": df["grid_clean"],
        "pitlane_start": df["pitlane_start"],
        "constructorId": df["constructorId"].astype("Int64"),
    })
    X = pd.get_dummies(features, columns=["constructorId"], drop_first=True)
    return X, y
