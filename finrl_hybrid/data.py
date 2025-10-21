from __future__ import annotations
from typing import Optional
import pandas as pd
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from .config import Config

class DataModule:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.raw: Optional[pd.DataFrame] = None
        self.train: Optional[pd.DataFrame] = None
        self.test: Optional[pd.DataFrame] = None

    def download(self):
        self.raw = YahooDownloader(
            start_date=self.cfg.start_date, end_date=self.cfg.end_date, ticker_list=self.cfg.tickers
        ).fetch_data()
        assert self.raw is not None and len(self.raw) > 0, "Empty download — check internet/proxy."
        requested = set(self.cfg.tickers)
        available = set(self.raw["tic"].unique())
        missing = sorted(requested - available)
        if missing:
            raise RuntimeError(f"Missing tickers: {missing}. Check spelling or adjust date range.")

    def build_features(self):
        fe = FeatureEngineer(use_technical_indicator=True, tech_indicator_list=self.cfg.tech_indicator_list)
        df = fe.preprocess_data(self.raw.copy())
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).copy()

        # Dynamic split: ensure all tickers exist in both train and test
        fdates = (
            df.sort_values(["tic","date"])
              .groupby("tic", as_index=False)["date"].min()
              .rename(columns={"date":"first_date"})
        )
        latest_first = fdates["first_date"].max()
        user_test_start = pd.Timestamp("2021-01-01")
        test_start = max(user_test_start, latest_first)
        if test_start <= pd.to_datetime(self.cfg.start_date):
            test_start = pd.to_datetime(self.cfg.start_date) + pd.Timedelta(days=1)

        self.train = data_split(df, self.cfg.start_date, (test_start - pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
        self.test  = data_split(df, test_start.strftime("%Y-%m-%d"), self.cfg.end_date)

        self.train["date"] = pd.to_datetime(self.train["date"], errors="coerce")
        self.test["date"]  = pd.to_datetime(self.test["date"],  errors="coerce")
