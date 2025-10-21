from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional

@dataclass
class Config:
    # Dates
    start_date: str = "2014-01-04"
    end_date: str   = "2025-07-31"

    # User preference for test start:
    # - If None -> falls back to auto mode using latest first date + min_train_days
    # - If set  -> we will take max(user_test_start, latest_first + min_train_days) unless force_exact_test_start=True
    user_test_start: Optional[str] = "2019-01-01"
    force_exact_test_start: bool = False        # if True, use user_test_start exactly (no auto bump)
    min_train_days: int = 1                     # ensure at least this many training days for the last-emerging ticker

    # Universe
    tickers: List[str] = None

    # Trading env params
    initial_amount: float = 1_000_000.0
    hmax_per_tic: int = 100
    buy_cost_pct: float = 5e-4
    sell_cost_pct: float = 5e-4
    reward_scaling: float = 1e-4
    eps_trade: float = 0.1

    # Components
    use_kalman: bool = True
    use_sac: bool = True
    use_ppo_lstm: bool = True

    # Training budgets
    total_timesteps_sac: int = 50_000
    total_timesteps_ppo: int = 30_000

    # Features
    tech_indicator_list: List[str] = None

    # IO
    out_dir: str = "fig"

    # Kalman preferences
    kalman_pair_priority: List[Tuple[str,str]] = None
    kalman_pairs: Optional[List[Tuple[str, str]]] = None

    def __post_init__(self):
        if self.tickers is None:
            self.tickers = ["QQQ","NVDA","MSFT","TSLA","NFLX"]
        if self.tech_indicator_list is None:
            self.tech_indicator_list = [
                "macd","rsi_30","cci_30","dx_30",
                "boll_ub","boll_lb",
                "close_50_sma","close_100_sma","close_200_sma",
            ]
        if self.kalman_pair_priority is None:
            self.kalman_pair_priority = [("NVDA","QQQ"), ("NFLX","QQQ"), ("QQQ","SPY"), ("SPY","IWM"), ("QQQ","IWM")]
        if self.kalman_pairs is None:
            anchor = "QQQ"
            self.kalman_pairs = [(t, anchor) for t in self.tickers if t != anchor and anchor in self.tickers]
