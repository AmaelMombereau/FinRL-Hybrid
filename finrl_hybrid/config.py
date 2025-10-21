from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional

@dataclass
class Config:
    # Dates
    start_date: str = "2016-01-04"
    end_date: str   = "2025-07-31"

    # Universe
    tickers: List[str] = None

    # Trading env params
    initial_amount: float = 1_000_000.0
    hmax_per_tic: int = 200
    buy_cost_pct: float = 5e-4
    sell_cost_pct: float = 5e-4
    reward_scaling: float = 1e-3
    eps_trade: float = 0.01

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
    kalman_pair_priority: List[Tuple[str,str]] = None  # fallback (y,x)
    kalman_pairs: Optional[List[Tuple[str, str]]] = None  # multi-pair list

    def __post_init__(self):
        if self.tickers is None:
            self.tickers = ["QQQ","NVDA","GOOGL","AAPL","MSFT","TSLA","NFLX","RUN"]
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
