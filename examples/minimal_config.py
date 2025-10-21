from finrl_hybrid.cli import main
from finrl_hybrid.config import Config

if __name__ == "__main__":
    cfg = Config(
        start_date="2018-01-01",
        end_date="2025-07-31",
        kalman_pairs=[("NVDA","QQQ"), ("AAPL","QQQ"), ("MSFT","QQQ"), ("NFLX","QQQ"), ("TSLA","QQQ")]
    )
    main(cfg)
