from __future__ import annotations
from typing import List
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from .config import Config
import pandas as pd

class EnvFactory:
    def __init__(self, cfg: Config, train: pd.DataFrame, test: pd.DataFrame):
        self.cfg = cfg
        train_set = set(train["tic"].unique())
        test_set  = set(test["tic"].unique())
        both = train_set.intersection(test_set)
        missing_both = [t for t in cfg.tickers if t not in both]
        if missing_both:
            raise AssertionError(
                "Some tickers are not present in BOTH train and test after dynamic split: "
                f"{missing_both}. Adjust dates if needed."
            )
        self.env_tickers: List[str] = [t for t in cfg.tickers if t in both]
        self.stock_dim = len(self.env_tickers)
        self.state_space = 1 + 2*self.stock_dim + len(cfg.tech_indicator_list)*self.stock_dim

        self.env_kwargs = dict(
            hmax=[cfg.hmax_per_tic] * self.stock_dim,
            initial_amount=cfg.initial_amount,
            buy_cost_pct=[cfg.buy_cost_pct] * self.stock_dim,
            sell_cost_pct=[cfg.sell_cost_pct] * self.stock_dim,
            state_space=self.state_space,
            stock_dim=self.stock_dim,
            tech_indicator_list=cfg.tech_indicator_list,
            action_space=self.stock_dim,
            reward_scaling=cfg.reward_scaling,
            num_stock_shares=[0] * self.stock_dim,
        )
        self.env_train = StockTradingEnv(df=train, **self.env_kwargs)
        self.env_test  = StockTradingEnv(df=test,  **self.env_kwargs)
