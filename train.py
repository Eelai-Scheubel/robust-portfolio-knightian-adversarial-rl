"""
train.py: train the normal and the robust trader
================================================

  python train.py normal    # the normal trader: 3,000 training markets, never manipulated
  python train.py robust    # the robust trader: the SAME 3,000 markets, same starting network,
                            # but a manipulator is active in the 80% of them that have one

Both traders start from the same random network (same --seed) and play the same fixed list of
training markets in the same order. So the only difference between them is the manipulator.

The robust trader and the manipulator take turns: every --block episodes, the manipulator
learns for --block episodes of its own (against the frozen trader), then the trader learns for
--block episodes of the shared list (against the frozen manipulator). They never learn at the
same time. The manipulator's markets are separate, so the trader's list stays identical.

No checkpoint selection: both traders are saved at the end of training, treated the same way.

The two commands are independent: you can run them at the same time in two terminals.
Outputs: models/normal.zip, models/robust.zip, models/manipulator.zip, and a training log.
"""

import argparse
import csv
import os
import time

import numpy as np
import torch
from stable_baselines3 import SAC

from market import Config, Player, TraderEnv, ManipulatorEnv, market_list

torch.set_num_threads(1)          # small networks: one thread is usually fastest


def make_sac(env, seed):
    return SAC("MlpPolicy", env, learning_rate=3e-4, batch_size=256, buffer_size=200_000,
               learning_starts=1_000, gamma=0.99, policy_kwargs=dict(net_arch=[128, 128]),
               seed=seed, verbose=0)


class EpisodeLog:
    """Writes one line per finished training episode."""

    def __init__(self, path, header):
        self.f = open(path, "w", newline="")
        self.w = csv.writer(self.f)
        self.w.writerow(header)

    def add(self, row):
        self.w.writerow(row)
        self.f.flush()


class LoggedTraderEnv(TraderEnv):
    """TraderEnv that records each episode's market and the trader's final return."""

    def __init__(self, *args, log=None, **kw):
        super().__init__(*args, **kw)
        self.log, self.total = log, 0.0

    def reset(self, seed=None, options=None):
        self.total = 0.0
        return super().reset(seed=seed, options=options)

    def step(self, action):
        obs, r, term, trunc, info = super().step(action)
        self.total += r
        if trunc and self.log:
            m = self.market.m
            self.log.add([self.k, round(m["mu"], 7), m["present"], round(np.expm1(self.total / 100), 5)])
        return obs, r, term, trunc, info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("which", choices=["normal", "robust"])
    ap.add_argument("--episodes", type=int, default=3000, help="training markets per trader")
    ap.add_argument("--block", type=int, default=50, help="episodes per turn (robust only)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="models")
    a = ap.parse_args()

    cfg = Config()
    os.makedirs(a.out, exist_ok=True)
    markets = market_list(a.episodes, seed=a.seed, cfg=cfg)      # the shared training markets
    log = EpisodeLog(f"{a.out}/train_log_{a.which}.csv", ["episode", "mu", "manipulator", "return"])
    t0 = time.time()

    if a.which == "normal":
        trader = make_sac(LoggedTraderEnv(cfg, markets, log=log), a.seed)
        chunk = 100
        for done in range(0, a.episodes, chunk):
            n = min(chunk, a.episodes - done)
            trader.learn(total_timesteps=n * cfg.T, reset_num_timesteps=False)
            print(f"  normal: {done + n:>5}/{a.episodes} episodes  {time.time() - t0:7.0f}s", flush=True)
        trader.save(f"{a.out}/normal")
        return

    # robust: trader and manipulator take turns
    manip_player, trader_player = Player(), Player()             # live references to each other
    trader = make_sac(LoggedTraderEnv(cfg, markets, manipulator=manip_player, log=log), a.seed)
    manip = make_sac(ManipulatorEnv(cfg, seed=a.seed + 10_000, trader=trader_player), a.seed + 1)
    trader_player.model, manip_player.model = trader, manip
    mlog = EpisodeLog(f"{a.out}/train_log_manipulator.csv", ["block", "avg_reward"])

    done, block = 0, 0
    while done < a.episodes:
        n = min(a.block, a.episodes - done)
        manip.learn(total_timesteps=n * cfg.T, reset_num_timesteps=False)    # trader frozen
        trader.learn(total_timesteps=n * cfg.T, reset_num_timesteps=False)   # manipulator frozen
        done, block = done + n, block + 1
        mlog.add([block, float(np.mean([e["r"] for e in manip.ep_info_buffer]) if manip.ep_info_buffer else 0.0)])
        if block % 2 == 0 or done >= a.episodes:
            print(f"  robust: {done:>5}/{a.episodes} episodes  {time.time() - t0:7.0f}s", flush=True)
    trader.save(f"{a.out}/robust")
    manip.save(f"{a.out}/manipulator")


if __name__ == "__main__":
    main()
