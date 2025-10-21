from __future__ import annotations
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd
from .config import Config
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from stable_baselines3.common.callbacks import CallbackList, ProgressBarCallback
from .sac_trainer import TrainProgressCallback, FinRLPnLCallback

class RecurrentPPORegime:
    def __init__(self, cfg: Config, env_kwargs: Dict[str, Any], train_df: pd.DataFrame, test_df: pd.DataFrame):
        self.cfg = cfg
        self.env_kwargs = env_kwargs
        self.train_df = train_df
        self.test_df = test_df
        self.model = None

    def train_and_score(self) -> Optional[pd.Series]:
        if not self.cfg.use_ppo_lstm:
            return None
        try:
            from sb3_contrib import RecurrentPPO

            def make_env(df_src):
                return StockTradingEnv(df=df_src.copy(), **self.env_kwargs)

            env_train = make_env(self.train_df)
            env_test  = make_env(self.test_df)

            ppo_kwargs = dict(
                learning_rate=3e-4, n_steps=1024, gamma=0.99, gae_lambda=0.95,
                ent_coef=0.01, vf_coef=0.5,
                policy_kwargs=dict(lstm_hidden_size=128, n_lstm_layers=1, net_arch=[128,128], ortho_init=False),
                verbose=1, seed=123,
            )
            self.model = RecurrentPPO("MlpLstmPolicy", env_train, **ppo_kwargs)

            callbacks = CallbackList([
                ProgressBarCallback(),
                TrainProgressCallback(total_timesteps=self.cfg.total_timesteps_ppo, step_print=10_000),
                FinRLPnLCallback(env_test, eval_freq=20_000, out_csv=None, name="R-PPO_EVAL")
            ])
            self.model.learn(total_timesteps=self.cfg.total_timesteps_ppo, callback=callbacks)

            # Recurrent rollout to get regime proxy
            obs, info = env_test.reset()
            lstm_state = None
            episode_start = np.array([True])
            actions_seq, dates_seq = [], []
            fallback_dates = (
                self.test_df[["date"]].drop_duplicates().sort_values("date")["date"].astype("datetime64[ns]").tolist()
            )
            t_counter = 0
            while True:
                action, lstm_state = self.model.predict(
                    obs, state=lstm_state, episode_start=episode_start, deterministic=True
                )
                obs, reward, terminated, truncated, info = env_test.step(action)
                dt = info.get("date", None)
                if dt is None:
                    dt = fallback_dates[min(t_counter, len(fallback_dates)-1)]
                else:
                    try:
                        dt = pd.to_datetime(dt)
                    except Exception:
                        dt = fallback_dates[min(t_counter, len(fallback_dates)-1)]
                dates_seq.append(dt)
                actions_seq.append(np.array(action).ravel().tolist())
                t_counter += 1
                episode_start[:] = (terminated or truncated)
                if terminated or truncated:
                    break

            scores = [float(np.tanh(np.mean(v))) if len(v) else 0.0 for v in actions_seq]
            return pd.Series(scores, index=pd.to_datetime(pd.Series(dates_seq))).clip(-1.0, 1.0)
        except Exception as e:
            print(f"[RecurrentPPO] unavailable or error: {e}")
            return None
