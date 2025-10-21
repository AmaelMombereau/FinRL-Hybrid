from __future__ import annotations
from typing import Optional, Tuple, List
import os
import pandas as pd
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, ProgressBarCallback
from finrl.agents.stablebaselines3.models import DRLAgent
from .config import Config
from .utils import ensure_actions_with_dates

class TrainProgressCallback(BaseCallback):
    def __init__(self, total_timesteps: int, step_print: int = 10_000):
        super().__init__()
        self.total = int(total_timesteps)
        self.step_print = int(step_print)
    def _on_step(self) -> bool:
        if self.step_print > 0 and self.num_timesteps % self.step_print == 0:
            pct = 100.0 * self.num_timesteps / max(1, self.total)
            print(f"[{self.model.__class__.__name__}] {self.num_timesteps}/{self.total}  ({pct:.1f}%)")
        return True

class FinRLPnLCallback(BaseCallback):
    def __init__(self, env_test, eval_freq=20_000, out_csv=None, name="EVAL"):
        super().__init__()
        self.env_test = env_test
        self.eval_freq = int(eval_freq)
        self.out_csv = out_csv
        self.name = name
        self.history = []
        self.agent_eval = DRLAgent(env=self.env_test)
    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.num_timesteps % self.eval_freq == 0:
            try:
                df_acc, _ = self.agent_eval.DRL_prediction(model=self.model, environment=self.env_test)
                start = float(df_acc["account_value"].iloc[0]); end = float(df_acc["account_value"].iloc[-1])
                pnl = end - start
                ret = 100.0 * (end / (start if start != 0 else 1.0) - 1.0)
                print(f"[{self.name}@{self.num_timesteps}] PnL={pnl:,.0f}  |  Return={ret:.2f}%")
                self.history.append({"timesteps": self.num_timesteps, "pnl": pnl, "ret_pct": ret})
                if self.out_csv:
                    import pandas as pd
                    pd.DataFrame(self.history).to_csv(self.out_csv, index=False)
            except Exception as e:
                print(f"[{self.name}] Eval error @ {self.num_timesteps}: {e}")
        return True

class SACTrainer:
    def __init__(self, cfg: Config, env_train, env_test, out_dir: str):
        self.cfg = cfg
        self.env_train = env_train
        self.env_test  = env_test
        self.model: Optional[SAC] = None
        self.out_dir = out_dir

    def train(self):
        sac_kwargs = dict(
            learning_rate=3e-4, gamma=0.99, tau=0.005,
            buffer_size=200_000, batch_size=256,
            ent_coef="auto", train_freq=(4, "step"),
            gradient_steps=1, learning_starts=10_000,
            verbose=1
        )
        self.model = SAC("MlpPolicy", self.env_train, **sac_kwargs)

        callbacks = CallbackList([
            ProgressBarCallback(),
            TrainProgressCallback(total_timesteps=self.cfg.total_timesteps_sac, step_print=10_000),
            FinRLPnLCallback(self.env_test, eval_freq=20_000,
                             out_csv=os.path.join(self.out_dir, "pnl_track_sac.csv"),
                             name="SAC_EVAL")
        ])
        self.model.learn(total_timesteps=self.cfg.total_timesteps_sac, callback=callbacks)

    def predict_actions(
        self, env_train, env_test, test_df, env_tickers: List[str]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        agent = DRLAgent(env=env_train)
        df_acc, df_actions = agent.DRL_prediction(model=self.model, environment=env_test)
        df_actions = ensure_actions_with_dates(df_actions, test_df, env_tickers)
        return df_acc, df_actions
