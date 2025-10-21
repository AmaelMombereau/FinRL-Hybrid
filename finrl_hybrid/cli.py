from __future__ import annotations
import os
from dataclasses import asdict
from typing import Optional
import pandas as pd
from stable_baselines3 import SAC
from finrl.agents.stablebaselines3.models import DRLAgent

from .config import Config
from .data import DataModule
from .envs import EnvFactory
from .kalman import KalmanPairsStrategy
from .sac_trainer import SACTrainer
from .regime import RecurrentPPORegime
from .fusion import FusionEngine
from .analysis import Analyzer
from .plotting import Plotter
from .utils import ensure_actions_with_dates

def main(cfg: Optional[Config] = None):
    cfg = cfg or Config()
    print("[Config]", asdict(cfg))
    os.makedirs(cfg.out_dir, exist_ok=True)

    # Data
    dm = DataModule(cfg); dm.download(); dm.build_features()

    # Envs
    envf = EnvFactory(cfg, dm.train, dm.test)
    print("[env] tickers =", envf.env_tickers)

    # Kalman (multi-pair)
    df_actions_kal = None
    if cfg.use_kalman:
        kal = KalmanPairsStrategy(cfg, envf.env_tickers, dm.test)
        df_actions_kal = kal.build_actions_multi()
        df_actions_kal = ensure_actions_with_dates(df_actions_kal, dm.test, envf.env_tickers)

    # SAC
    df_acc_sac = df_actions_sac = None
    if cfg.use_sac:
        sac = SACTrainer(cfg, envf.env_train, envf.env_test, cfg.out_dir)
        sac.train()
        df_acc_sac, df_actions_sac = sac.predict_actions(envf.env_train, envf.env_test, dm.test, envf.env_tickers)
        df_actions_sac = ensure_actions_with_dates(df_actions_sac, dm.test, envf.env_tickers)

    # Regime (optional)
    regime_series = None
    if cfg.use_ppo_lstm:
        rp = RecurrentPPORegime(cfg, envf.env_kwargs, dm.train, dm.test)
        regime_series = rp.train_and_score()

    # Fusion
    fusion = FusionEngine(cfg, envf.env_tickers)
    df_actions_combined = fusion.combine(df_actions_sac, df_actions_kal, regime_series)
    print("[COMBINED] actions:", df_actions_combined.shape)

    # Account value curve (for plots)
    agent_eval = DRLAgent(env=envf.env_test)
    model_eval = SAC("MlpPolicy", envf.env_train) if cfg.use_sac else None
    df_account_value, _ = agent_eval.DRL_prediction(model=(sac.model if cfg.use_sac else model_eval),
                                                    environment=envf.env_test)

    # Analysis
    analyzer = Analyzer(cfg, envf.env_tickers, dm.test)
    ana = analyzer.pnl_by_ticker(df_actions_combined)
    pnl_df: pd.DataFrame = ana["pnl_df"]
    pnl_df.to_csv(os.path.join(cfg.out_dir, "per_ticker_pnl.csv"), index=False)
    print("\n=== Net PnL by ticker (test) ===")
    print(pnl_df.to_string(index=False))

    # Final portfolio summary CSV
    acc_sorted = df_account_value.sort_values("date").copy()
    acc_sorted["date"] = pd.to_datetime(acc_sorted["date"])
    base_val, last_val = float(acc_sorted["account_value"].iloc[0]), float(acc_sorted["account_value"].iloc[-1])
    final_pnl = last_val - base_val
    final_ret = (last_val / (base_val if base_val != 0 else 1.0) - 1.0) * 100.0
    pd.DataFrame([{
        "start_date": acc_sorted["date"].iloc[0].date(),
        "end_date":   acc_sorted["date"].iloc[-1].date(),
        "start_value": base_val,
        "end_value":   last_val,
        "pnl":         final_pnl,
        "return_pct":  final_ret,
    }]).to_csv(os.path.join(cfg.out_dir, "summary_pnl.csv"), index=False)
    print(f"\n=== Final portfolio ===\nPnL = {final_pnl:,.2f} $ | Return = {final_ret:,.2f} %")

    # Plots
    plotter = Plotter(cfg.out_dir)
    plotter.portfolio_vs_price(envf.env_tickers, dm.test, df_account_value)
    plotter.trades_and_pnl(envf.env_tickers, ana["dates_tx"], ana["PX"], ana["pnl_cum"],
                           ana["tx_vecs"], cfg.eps_trade,
                           trade_stats=ana["trade_stats"], per_ticker_pnl=ana["per_ticker_pnl"])
    print(f"[FIGS] saved in: {cfg.out_dir}")

if __name__ == "__main__":
    main()
