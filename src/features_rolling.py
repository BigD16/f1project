from __future__ import annotations
from typing import List, Sequence, Optional, Tuple
import pandas as pd
import numpy as np

def _shifted_rolling(
    s: pd.Series,
    window: int,
    min_periods: int,
    shift: int,
    agg: str,
) -> pd.Series:
    """Compute rolling aggregation on a Series, shifted by `shift` periods to avoid
    leakage of current row into rolling calculation.
    Parameters
    ----------
    s : pd.Series ,Input time series.
    window : int , Size of the rolling window.
    min_periods : int , Minimum number of observations in window required to have a value.
    shift : int , Number of periods to shift the rolling window backwards.
    agg : str , Aggregation function to apply ('mean', 'sum', etc.).
    Returns
    -------
    pd.Series
        Series of the same length as `s` with the rolling aggregation."""
    if shift < 1:
        raise ValueError("shift must be at least 1 to avoid leakage")
    roll = s.rolling(window=window, min_periods= min_periods)
    if not hasattr(roll, agg):
        raise ValueError(f"Aggregation '{agg}' is not supported on rolling objects")
    return getattr(roll, agg)()

def add_rolling_features(
    df: pd.DataFrame,
    groupby_cols: list[str],
    target_cols: str,
    windows: list[int],
    aggs: list[str],
     *,
    time_sort_cols: list[str] = ("date", "raceId", "driverId"),
    shift: int = 1,
    min_periods: int = 1,
    prefix: Optional[str]= None,
) -> pd.DataFrame:
    """
    Adds rolling features to the DataFrame.
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame sorted by time.
    groupby_cols : list[str]
        Columns to group by (e.g., driverId).
    target_col : str
        Target column to compute rolling statistics on (e.g., positionOrder).
    windows : list[int]
        List of window sizes for rolling calculations.
    aggs : list[str]
        List of aggregation functions to apply ('mean', 'sum', etc.).
    Returns
    -------
    pd.DataFrame
        DataFrame with added rolling feature columns.
    """
    if isinstance(target_cols, str):
        target_cols = [target_cols]

    out = df.sort_values(list(time_sort_cols)).copy()
    base = prefix or "_".join(groupby_cols)
    for target in target_cols:
        if target not in out.columns:
            raise ValueError(f"Target column '{target}' not found in DataFrame")
        for window in windows:
            for agg in aggs:
                feat = f"{base}_rolling_{target}_{agg}_w{window}"
                out[feat] = out.groupby(groupby_cols)[target].transform(
                    lambda s: _shifted_rolling(s, window=window, min_periods=min_periods, shift=shift, agg=agg)
                )

    return out

def add_domain_rolling_features(
        df: pd.DataFrame,
        *,
        driver_windows: Sequence[int] = (3, 5, 10),
        team_windows: Sequence[int] = (3, 5, 10),
        ) -> pd.DataFrame:
    
    """
    Adds driver & constructor (team) rolling/history features that are
    leakage-safe by construction (shifted).

    Requires df columns:
      - date, raceId, driverId, constructorId
      - positionOrder, points
      - year (for decade)
      - dob (optional, for driver_age)
      - grid_clean, pitlane_start (from your baseline preprocessing)
    """
    # Ensure chronological order
    out = df.sort_values(["date", "raceId", "driverId"]).copy()

    # Helpful binary target for "rates"
    out["is_podium"] = (out["positionOrder"] <= 3).astype(int)

    # ---- Driver history (cumulative up to previous race) ----
    out["drv_prev_starts"] = out.groupby("driverId").cumcount()
    out["drv_prev_podiums"] = (
        out.groupby("driverId")["is_podium"]
           .transform(lambda s: s.shift(1).cumsum())
    )
    out["drv_prev_points"] = (
        out.groupby("driverId")["points"]
           .transform(lambda s: s.shift(1).cumsum())
    )

    # Rolling driver form
    out = add_rolling_features(
        out,
        groupby_cols=["driverId"],
        target_cols=["positionOrder", "points", "is_podium"],
        windows=list(driver_windows),
        aggs=["mean"],
        prefix="drv",
        time_sort_cols=("date","raceId","driverId"),
        shift=1,
        min_periods=1,
    )

    # ---- Team history (cumulative up to previous race) ----
    out["team_prev_starts"] = out.groupby("constructorId").cumcount()
    out["team_prev_points"] = (
        out.groupby("constructorId")["points"]
           .transform(lambda s: s.shift(1).cumsum())
    )

    # Rolling team form
    out = add_rolling_features(
        out,
        groupby_cols=["constructorId"],
        target_cols=["positionOrder", "points", "is_podium"],
        windows=list(team_windows),
        aggs=["mean"],
        prefix="team",
        time_sort_cols=("date","raceId","driverId"),
        shift=1,
        min_periods=1,
    )

    # ---- Calendar / era ----
    if "date" in out.columns and pd.api.types.is_datetime64_any_dtype(out["date"]):
        out["month"] = out["date"].dt.month
    if "year" in out.columns:
        out["decade"] = (out["year"] // 10) * 10

    # ---- Driver age on race day (optional) ----
    if "dob" in out.columns:
        dob = pd.to_datetime(out["dob"], errors="coerce")
        out["driver_age"] = ((out["date"] - dob).dt.days / 365.25).astype(float)

    return out


# ---------- Assemble design matrix ----------

def build_tabular_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Build X, y for modeling from a DataFrame that already has:
      - baseline features: grid_clean, pitlane_start
      - domain rolling features (via add_domain_rolling_features)
      - ids/meta: constructorId, circuitId, year/date, etc.

    Returns
    -------
    X : pd.DataFrame
    y : pd.Series  (1 if podium, else 0)
    """
    # Target
    y = (df["positionOrder"] <= 3).astype(int)

    # Numeric features to include (known + patterns)
    numeric_whitelist = {
        "grid_clean", "pitlane_start",
        "drv_prev_starts", "drv_prev_podiums", "drv_prev_points",
        "team_prev_starts", "team_prev_points",
        "month", "driver_age"
    }

    # Auto-collect any generated rolling features (names contain "__roll_")
    roll_cols = [c for c in df.columns if "__roll_" in c]

    num_cols = [c for c in numeric_whitelist if c in df.columns] + roll_cols

    # Assemble numeric matrix, fill early-history NaNs with 0
    X_num = df[num_cols].copy() if num_cols else pd.DataFrame(index=df.index)
    X_num = X_num.fillna(0)

    # Compact categoricals (avoid exploding feature space)
    cat_cols = []
    if "constructorId" in df.columns:
        cat_cols.append("constructorId")
    if "circuitId" in df.columns:
        cat_cols.append("circuitId")
    if "decade" in df.columns:
        cat_cols.append("decade")

    X_cat = (
        pd.get_dummies(df[cat_cols].astype("category"), drop_first=True)
        if cat_cols else
        pd.DataFrame(index=df.index)
    )
    X = pd.concat([X_num, X_cat], axis=1)

    return X, y    
    