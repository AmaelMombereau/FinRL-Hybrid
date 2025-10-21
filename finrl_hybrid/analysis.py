from __future__ import annotations
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from .config import Config
from .utils import _to_vec

class Analyzer:
    def __init__(self, cfg: Config, env_tickers: List[str], test_df: pd.DataFrame):
        self.cfg = cfg
        self.env_tickers = env_tickers
        self.test_df = test_df

    def _infer_tx_vecs(self, df_actions: pd.DataFrame) -> List[np.ndarray]:
        L = len(self.env_tickers)
        def norm_vec(v) -> np.ndarray:
            arr = np.array(v, dtype=float).ravel()
            if arr.size < L:
                arr = np.pad(arr, (0, L - arr.size))
            elif arr.size > L:
                arr = arr[:L]
            return arr
        lower = {c: c.lower() for c in df_actions.columns}
        cands = [c for c in df_actions.columns if lower[c] in ("transactions","transaction","actions","action","positions","weights")]
        action_col = cands[0] if cands else None
        if action_col is not None:
            tx_vecs = [norm_vec(_to_vec(x)) for x in df_actions[action_col].tolist()]
        else:
            per_tic_cols = [c for c in df_actions.columns if c in self.env_tickers]
            if per_tic_cols:
                rows = []
                for _, row in df_actions.iterrows():
                    v = [float(row[t]) if (t in row and pd.notna(row[t])) else 0.0 for t in self.env_tickers]
                    rows.append(norm_vec(v))
                tx_vecs = rows
            else:
                def looks_like_vec(series):
                    sample = series.astype(str).head(10).tolist()
                    hits = sum(1 for s in sample if re.search(r"[\[\(].+[, ].+[\]\)]", s))
                    return hits >= 2
                tx_vecs = None
                for c in [c for c in df_actions.columns if lower[c] not in ("date","datetime")]:
                    if looks_like_vec(df_actions[c]):
                        tx_vecs = [norm_vec(_to_vec(x)) for x in df_actions[c].tolist()]
                        break
                if tx_vecs is None:
                    raise RuntimeError("Could not identify action vectors.")
        # If vectors look like positions, convert to Δshares
        if len(tx_vecs) >= 2 and len(tx_vecs[0]) > 0:
            arr = np.vstack([v if len(v)==len(tx_vecs[0]) else np.zeros(len(tx_vecs[0])) for v in tx_vecs])
            dif = np.vstack([np.zeros_like(arr[0]), arr[1:] - arr[:-1]])
            if np.nansum(np.abs(dif)) * 5 < np.nansum(np.abs(arr)):
                print("[info] positions detected -> converting to Δshares (diff)")
                tx_vecs = [dif[i] for i in range(dif.shape[0])]
        tx_vecs = [norm_vec(v) for v in tx_vecs]
        return tx_vecs

    def pnl_by_ticker(self, df_actions: pd.DataFrame) -> Dict[str, Any]:
        df_actions = df_actions.sort_values("date").copy()
        tx_vecs = self._infer_tx_vecs(df_actions)
        dates_tx = pd.to_datetime(df_actions["date"]).sort_values().tolist()
        L = len(self.env_tickers)
        TX = np.vstack([
            (np.array(v, dtype=float).ravel()[:L] if len(v)>=L else np.pad(np.array(v, dtype=float).ravel(), (0, L-len(v))))
            for v in tx_vecs
        ])

        # Align prices (asof) on action dates
        px_all = (
            self.test_df[["date","tic","close"]]
            .drop_duplicates(["date","tic"])
            .assign(date=lambda d: pd.to_datetime(d["date"], errors="coerce"))
            .pivot(index="date", columns="tic", values="close")
            .sort_index()
        )
        left_df = pd.DataFrame({"date": pd.to_datetime(dates_tx, errors="coerce")}).dropna().sort_values("date")
        right_df = px_all.reset_index().assign(date=lambda d: pd.to_datetime(d["date"], errors="coerce")).dropna(subset=["date"]).sort_values("date")
        px_aligned = (pd.merge_asof(left_df, right_df, on="date", direction="backward", tolerance=pd.Timedelta("7D")).set_index("date"))
        px_aligned = px_aligned.reindex(columns=self.env_tickers).ffill().dropna(axis=0, how="any")

        common_T = min(len(px_aligned), len(TX))
        dates_tx = px_aligned.index[:common_T]
        PX = px_aligned.iloc[:common_T].values
        TX = TX[:common_T, :]

        POS = np.zeros_like(TX); POS[0] = TX[0]
        if common_T > 1:
            POS[1:] = np.cumsum(TX[1:], axis=0) + POS[0]
        dP = np.zeros_like(PX)
        if common_T > 1:
            dP[1:] = PX[1:] - PX[:-1]
        PnL_tick = np.zeros_like(PX)
        if common_T > 1:
            PnL_tick[1:] = POS[:-1] * dP[1:]

        buy_fee  = (TX > 0) * self.cfg.buy_cost_pct
        sell_fee = (TX < 0) * self.cfg.sell_cost_pct
        fees = np.abs(TX) * PX * (buy_fee + sell_fee)

        PnL_tick_net = PnL_tick - fees
        PnL_tick_cum = np.cumsum(PnL_tick_net, axis=0)

        per_ticker_pnl = {tic: float(PnL_tick_net[:, j].sum()) for j, tic in enumerate(self.env_tickers)}
        pnl_df = pd.DataFrame({"ticker": self.env_tickers, "pnl_net": [per_ticker_pnl[t] for t in self.env_tickers]}).sort_values("pnl_net", ascending=False)

        # trade stats
        trade_stats = {}
        for j, tic in enumerate(self.env_tickers):
            col = np.array([v[j] if len(v) > j else 0.0 for v in tx_vecs], dtype=float)
            buys  = int((col >  self.cfg.eps_trade).sum())
            sells = int((col < -self.cfg.eps_trade).sum())
            trade_stats[tic] = {"buys": buys, "sells": sells, "trades": buys + sells}

        return dict(
            dates_tx=dates_tx, TX=TX, PX=PX,
            pnl_df=pnl_df, pnl_cum=PnL_tick_cum,
            tx_vecs=tx_vecs, trade_stats=trade_stats,
            per_ticker_pnl=per_ticker_pnl
        )
