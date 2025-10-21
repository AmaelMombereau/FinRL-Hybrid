from __future__ import annotations
import ast, re
from typing import List, Optional
import numpy as np
import pandas as pd

def _to_vec(x):
    if isinstance(x, (list, tuple, np.ndarray)):
        return np.array(x, dtype=float)
    s = str(x)
    try:
        return np.array(ast.literal_eval(s), dtype=float)
    except Exception:
        nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
        return np.array([float(v) for v in nums], dtype=float) if nums else np.array([], dtype=float)

def ensure_actions_with_dates(
    df_actions_like: Optional[pd.DataFrame],
    ref_df: pd.DataFrame,
    env_tickers: List[str]
) -> Optional[pd.DataFrame]:
    """Return ['date','transactions'] ordered by env_tickers, values list[float]."""
    if df_actions_like is None:
        return None
    if not isinstance(df_actions_like, pd.DataFrame):
        df_actions_like = pd.DataFrame(df_actions_like)

    out = df_actions_like.copy()
    idx_names = list(out.index.names) if hasattr(out.index, "names") else [out.index.name]
    if idx_names and ("date" in idx_names):
        out = out.reset_index()
    out.index.name = None

    if out.columns.duplicated().any():
        out = out.loc[:, ~out.columns.duplicated()]

    lower = {c: c.lower() for c in out.columns}
    vec_col = None
    for c in out.columns:
        if lower[c] in ("transactions","transaction","actions","action","positions","weights"):
            vec_col = c; break

    if vec_col is None:
        if set(env_tickers).issubset(set(out.columns)):
            out["transactions"] = out.apply(lambda r: [float(r[t]) if pd.notna(r[t]) else 0.0 for t in env_tickers], axis=1)
        else:
            out["transactions"] = out.apply(lambda r: [float(x) for x in r.values if np.isscalar(x)], axis=1)
    elif vec_col != "transactions":
        out = out.rename(columns={vec_col: "transactions"})

    if "date" not in out.columns:
        ref_dates = (
            ref_df[["date"]].drop_duplicates().sort_values("date")["date"]
            .astype("datetime64[ns]").tolist()
        )
        n = min(len(ref_dates), len(out))
        out = out.iloc[:n].copy()
        out.insert(0, "date", ref_dates[:n])
    else:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")

    out["transactions"] = out["transactions"].apply(
        lambda v: np.array(v, dtype=float).tolist()
        if not isinstance(v, (list, np.ndarray))
        else np.array(v, dtype=float).tolist()
    )
    return out.sort_values("date").reset_index(drop=True)
