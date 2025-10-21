from __future__ import annotations
import os
from typing import List, Dict
import numpy as np
import pandas as pd

class Plotter:
    def __init__(self, out_dir: str):
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        import matplotlib
        if "JPY_PARENT_PID" not in os.environ:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        self.plt = plt

    def portfolio_vs_price(self, env_tickers: List[str], test_df: pd.DataFrame, df_account_value: pd.DataFrame):
        plt = self.plt
        acc = df_account_value.copy().sort_values("date").reset_index(drop=True)
        acc["date"] = pd.to_datetime(acc["date"]); acc["pnl"] = acc["account_value"] - acc["account_value"].iloc[0]

        for tic in env_tickers:
            px_one = (test_df.loc[test_df["tic"] == tic, ["date","close"]]
                      .drop_duplicates("date").sort_values("date").reset_index(drop=True))
            px_one["date"] = pd.to_datetime(px_one["date"])
            m = pd.merge(acc[["date","account_value","pnl"]], px_one[["date","close"]], on="date", how="inner").sort_values("date").reset_index(drop=True)
            if m.empty:
                m_asof = pd.merge_asof(acc.sort_values("date"), px_one.sort_values("date"),
                                       on="date", tolerance=pd.Timedelta("3D"), direction="backward")
                m = m_asof.dropna(subset=["close"]).copy()
            if m.empty:
                print(f"[warn] could not align account_value with {tic}"); continue

            fig = plt.figure(); plt.plot(px_one["date"], px_one["close"])
            plt.title(f"{tic} price (test)"); plt.xlabel("Date"); plt.ylabel("Close")
            plt.tight_layout(); fig.savefig(os.path.join(self.out_dir, f"{tic}_price_test.png"), dpi=150, bbox_inches="tight"); plt.close(fig)

            fig = plt.figure()
            plt.plot(m["date"], m["account_value"], label="Account value")
            plt.plot(m["date"], m["pnl"], label="PnL (Δ)")
            plt.title("Account value & PnL"); plt.xlabel("Date"); plt.ylabel("USD"); plt.legend(); plt.tight_layout()
            fig.savefig(os.path.join(self.out_dir, f"account_value_and_pnl_{tic}.png"), dpi=150, bbox_inches="tight"); plt.close(fig)

            base_acct, base_price = float(m["account_value"].iloc[0]), float(m["close"].iloc[0])
            m["acct_100"]  = m["account_value"] / (base_acct  if base_acct  != 0 else 1.0) * 100.0
            m["price_100"] = m["close"]         / (base_price if base_price != 0 else 1.0) * 100.0
            fig = plt.figure()
            plt.plot(m["date"], m["acct_100"],  label="Portfolio (100)")
            plt.plot(m["date"], m["price_100"], '--', label=f"{tic} (100)")
            plt.title(f"Portfolio vs {tic} — rebased 100"); plt.xlabel("Date"); plt.ylabel("Index (100 = start)")
            plt.legend(); plt.tight_layout()
            fig.savefig(os.path.join(self.out_dir, f"rebased100_portfolio_vs_{tic}.png"), dpi=150, bbox_inches="tight"); plt.close(fig)

    def trades_and_pnl(self, env_tickers: List[str], dates_tx: pd.Index, PX: np.ndarray, pnl_cum: np.ndarray,
                       tx_vecs, eps_trade: float, trade_stats: Dict[str, dict], per_ticker_pnl: Dict[str, float]):
        plt = self.plt
        for j, tic in enumerate(env_tickers):
            if len(dates_tx) == 0:
                print(f"[warn] no aligned dates for {tic}")
                continue
            px_series = pd.Series(PX[:, j], index=dates_tx)
            pnl_cum_series = pd.Series(pnl_cum[:, j], index=dates_tx)

            col = np.array([v[j] if len(v) > j else 0.0 for v in tx_vecs], dtype=float)
            T2 = min(len(col), len(dates_tx))
            col = col[:T2]; dates_v = pd.Index(dates_tx[:T2])
            buy_idx  = np.where(col >  eps_trade)[0]
            sell_idx = np.where(col < -eps_trade)[0]

            fig, ax1 = plt.subplots()
            ax1.plot(dates_v, px_series.iloc[:T2], label=f"Price {tic}")
            ax1.set_xlabel("Date"); ax1.set_ylabel("Price")
            if len(buy_idx):
                ax1.scatter(dates_v[buy_idx], px_series.iloc[:T2].iloc[buy_idx], marker="^", label="Buy")
            if len(sell_idx):
                ax1.scatter(dates_v[sell_idx], px_series.iloc[:T2].iloc[sell_idx], marker="v", label="Sell")
            ax1.legend(loc="upper left")

            ax2 = ax1.twinx()
            ax2.plot(dates_v, pnl_cum_series.iloc[:T2], linestyle="--", label="Cum PnL")
            ax2.set_ylabel("Cum PnL (USD)")
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax2.legend(lines1+lines2, labels1+labels2, loc="upper right")

            trades_n = trade_stats.get(tic, {}).get("trades", 0)
            pnl_net  = per_ticker_pnl.get(tic, float(pnl_cum_series.iloc[T2-1]))
            plt.title(f"{tic}: Price + Trades + Cum PnL\nTrades = {trades_n} | Net PnL = {pnl_net:,.2f} $")
            plt.tight_layout()
            out = os.path.join(self.out_dir, f"{tic}_price_trades_pnl.png")
            fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
            print(f"[FIG] {out} — Final {tic} PnL = {pnl_cum_series.iloc[T2-1]:,.2f} $")
