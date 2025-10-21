from __future__ import annotations
from typing import List, Tuple
import numpy as np
import pandas as pd
from .config import Config

class KalmanPairsStrategy:
    def __init__(self, cfg: Config, env_tickers: List[str], test_df: pd.DataFrame):
        self.cfg = cfg
        self.env_tickers = env_tickers
        self.test = test_df

    def pick_pair(self) -> Tuple[str, str]:
        available = set(self.test["tic"].unique())
        for a,b in self.cfg.kalman_pair_priority:
            if a in available and b in available:
                return a,b
        if len(self.env_tickers) >= 2:
            return self.env_tickers[0], self.env_tickers[1]
        raise RuntimeError("Not enough tickers to form a Kalman pair.")

    def _pair_actions(self, y_tic: str, x_tic: str) -> pd.DataFrame:
        px = (self.test[["date","tic","close"]]
              .drop_duplicates(["date","tic"])
              .pivot(index="date", columns="tic", values="close")
              .sort_index()
              .dropna(how="any", subset=[y_tic, x_tic]))
        if px.empty:
            return pd.DataFrame(columns=["date","transactions"])
        y = px[y_tic].astype(float); x = px[x_tic].astype(float)

        # 1D Kalman for time-varying beta
        Qv, Rv = 1e-6, 1e-3
        beta_est, P, beta = [], 1.0, (y/(x.replace(0,np.nan))).fillna(1.0).iloc[0]
        for yt, xt in zip(y.values, x.values):
            beta_pred = beta; P_pred = P + Qv
            if abs(xt) < 1e-12:
                beta, P = beta_pred, P_pred
            else:
                H = xt; innov = yt - H*beta_pred; S = H*P_pred*H + Rv
                if S == 0: S = 1e-12
                K = (P_pred * H) / S
                beta = beta_pred + K*innov
                P = (1 - K*H) * P_pred
            beta_est.append(beta)
        beta_series = pd.Series(beta_est, index=px.index).ewm(alpha=0.2, adjust=False).mean()

        # Mean-reversion target via spread z-score
        win, z_out = 60, 0.2
        spread = y - beta_series * x
        mu = spread.rolling(win, min_periods=max(10,win//3)).mean()
        sd = spread.rolling(win, min_periods=max(10,win//3)).std(ddof=0)
        z = ((spread - mu) / sd.replace(0,np.nan)).fillna(0.0)

        max_pos = self.cfg.hmax_per_tic
        size_k  = 0.5
        target_y = (-size_k * z).clip(-1.0, 1.0) * max_pos
        target_x = -(beta_series * target_y * (y / x)).replace([np.inf, -np.inf], 0.0).fillna(0.0)

        pos_y, pos_x = [], []
        for tz, ty, tx_ in zip(z.values, target_y.values, target_x.values):
            if abs(tz) < z_out: ty, tx_ = 0.0, 0.0
            pos_y.append(float(np.clip(ty, -max_pos, max_pos)))
            pos_x.append(float(np.clip(tx_, -max_pos, max_pos)))

        pos = pd.DataFrame({"date": px.index, f"{y_tic}_pos": pos_y, f"{x_tic}_pos": pos_x}).set_index("date")
        dpos = pos.diff().fillna(pos.iloc[[0]]).where(lambda d: d.abs() > self.cfg.eps_trade, 0.0)

        env_index = {tic: j for j, tic in enumerate(self.env_tickers)}
        vecs, dates = [], px.index.tolist()
        for dt in dates:
            v = np.zeros(len(self.env_tickers), dtype=float)
            if y_tic in env_index: v[env_index[y_tic]] = float(dpos.loc[dt, f"{y_tic}_pos"])
            if x_tic in env_index: v[env_index[x_tic]] = float(dpos.loc[dt, f"{x_tic}_pos"])
            vecs.append(v)
        return pd.DataFrame({"date": dates, "transactions": [v.tolist() for v in vecs]})

    def build_actions(self) -> pd.DataFrame:
        y_tic, x_tic = self.pick_pair()
        return self._pair_actions(y_tic, x_tic)

    def build_actions_multi(self) -> pd.DataFrame:
        env_set = set(self.env_tickers)
        pairs = [(a,b) for (a,b) in (self.cfg.kalman_pairs or []) if a in env_set and b in env_set]
        if not pairs:
            y,x = self.pick_pair()
            return self._pair_actions(y,x)

        frames = []
        for (y,x) in pairs:
            dfp = self._pair_actions(y,x)
            if not dfp.empty:
                dfp["date"] = pd.to_datetime(dfp["date"])
                frames.append(dfp)
        if not frames:
            return pd.DataFrame(columns=["date","transactions"])

        all_dates = sorted(pd.to_datetime(pd.concat([f["date"] for f in frames], ignore_index=True)).unique())
        L = len(self.env_tickers)
        def norm(v):
            a = np.array(v, float).ravel()
            if a.size < L: a = np.pad(a, (0, L-a.size))
            elif a.size > L: a = a[:L]
            return a

        out = []
        for dt in all_dates:
            acc = np.zeros(L, float)
            for f in frames:
                row = f.loc[f["date"]==dt, "transactions"]
                if len(row):
                    acc += norm(row.iloc[0])
            acc = np.clip(acc, -self.cfg.hmax_per_tic, self.cfg.hmax_per_tic)
            acc[np.abs(acc) < self.cfg.eps_trade] = 0.0
            out.append(acc.tolist())
        return pd.DataFrame({"date": all_dates, "transactions": out}).sort_values("date").reset_index(drop=True)
