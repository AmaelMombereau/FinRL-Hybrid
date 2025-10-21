from __future__ import annotations
from typing import Optional, List
import numpy as np
import pandas as pd
from .config import Config

class FusionEngine:
    def __init__(self, cfg: Config, env_tickers: List[str]):
        self.cfg = cfg
        self.env_tickers = env_tickers

    def combine(self, df_sac: Optional[pd.DataFrame], df_kal: Optional[pd.DataFrame],
                regime: Optional[pd.Series]) -> pd.DataFrame:
        def _to_np_vec(x) -> np.ndarray:
            if x is None:
                return np.zeros(len(self.env_tickers), dtype=float)
            arr = np.array(x, dtype=float).ravel()
            L = len(self.env_tickers)
            if arr.size < L:
                arr = np.pad(arr, (0, L - arr.size))
            elif arr.size > L:
                arr = arr[:L]
            return arr

        df_sac = df_sac.copy() if df_sac is not None else None
        df_kal = df_kal.copy() if df_kal is not None else None
        if df_sac is not None:
            df_sac["date"] = pd.to_datetime(df_sac["date"])
        if df_kal is not None:
            df_kal["date"] = pd.to_datetime(df_kal["date"])

        if df_sac is None and df_kal is None:
            raise RuntimeError("Neither SAC nor Kalman provided.")
        if df_sac is None:  return df_kal.sort_values("date").reset_index(drop=True)
        if df_kal is None:  return df_sac.sort_values("date").reset_index(drop=True)

        sac_map = {pd.to_datetime(d): _to_np_vec(v) for d, v in zip(df_sac["date"], df_sac["transactions"])}
        kal_map = {pd.to_datetime(d): _to_np_vec(v) for d, v in zip(df_kal["date"], df_kal["transactions"])}
        all_dates = sorted(set(sac_map.keys()).union(kal_map.keys()))

        w_sac_base, w_kal_base = 0.8, 0.2
        out_dates, out_vecs = [], []
        for dt in all_dates:
            v_sac = sac_map.get(dt)
            v_kal = kal_map.get(dt)
            if v_sac is None and v_kal is None:
                continue
            if v_sac is None:
                v = v_kal
            elif v_kal is None:
                v = v_sac
            else:
                if regime is not None and dt in regime.index:
                    r = float(regime.loc[dt])  # [-1, 1]
                    w_sac = w_sac_base * (0.5 + 0.5*(r+1)/2) + 0.2
                    w_kal = w_kal_base * (0.5 + 0.5*(1 - (r+1)/2)) + 0.2
                else:
                    w_sac, w_kal = w_sac_base, w_kal_base
                v = w_sac * v_sac + w_kal * v_kal
            v = np.clip(v, -self.cfg.hmax_per_tic, self.cfg.hmax_per_tic)
            v[np.abs(v) < self.cfg.eps_trade] = 0.0
            out_dates.append(dt); out_vecs.append(v.tolist())

        return pd.DataFrame({"date": out_dates, "transactions": out_vecs}).sort_values("date").reset_index(drop=True)
