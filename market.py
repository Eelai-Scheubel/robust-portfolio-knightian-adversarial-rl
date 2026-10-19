"""
market.py: a very simplified stock market with a trader and a manipulator
=========================================================================

One stock, 260 trading days per episode (one year).

PRICE = TRUE VALUE x DISTORTION
  true value V   grows by a constant trend mu every day (mu is hidden from the trader)
  distortion I   created by trading; half of it fades away every day
  price          P = V * exp(I)

Each day both players send an order at the same time. The net order Q (shares) moves the price:
  dI = eta * sign(Q) * sqrt(|Q| / daily_volume)          (square-root impact)
and both trade at the average price along that move, paying a 0.1% fee.

RANDOM PER EPISODE
  mu      uniform in +/-0.0943% a day  (= +/-2% a month, about +/-27% a year)
  eta     how easily the price moves, log-uniform 0.5%..5%
  F       free float, log-uniform 10M..1B shares; daily volume = 0.5% of F
  80% of episodes have a manipulator, with
  B       cash, log-uniform $100k..$1B          (limits how much it can buy)
  q0/F    shares owned, log-uniform 0.01%..15%   (limits how much it can sell; no short selling)

PLAYERS
  trader       action in [-1, 1] = trade that fraction of its wealth (+ buy, - sell).
               Sees: its last 20 daily returns, its average return over the last 20 days,
               60 days and since day 1, how invested it is, its last trade, days left, and the
               stock's eta and float. Nothing about the manipulator.
               Reward: 100 x log(wealth today / wealth yesterday). Shares are valued at the true
               value on the last day, so a fake last-minute dip is not a real loss.
  manipulator  action in [-1, 1] = target price push, log scale from 0.01% to 5% (|action|
               below 0.1 = do nothing). The market turns that into a number of shares; its cash
               and shares cap what it can actually do.
               Sees everything, including mu and the trader's position.
               Reward (worst case): the trader's loss compared with the SAME market without
               manipulation (a shadow copy runs alongside), so doing nothing scores exactly 0.
"""

from dataclasses import dataclass

import numpy as np
import gymnasium as gym
from gymnasium import spaces


@dataclass
class Config:
    T: int = 260                       # trading days per episode
    mu_max: float = 0.000943           # true daily trend in [-mu_max, +mu_max]
    P0: float = 100.0
    capital: float = 1e6               # trader starts with $1M cash
    fee: float = 0.001
    min_trade: float = 0.01            # trader trades below 1% of wealth are ignored
    rho: float = 0.5                   # share of the distortion that survives each day
    eta_range: tuple = (0.005, 0.05)
    float_range: tuple = (1e7, 1e9)
    turnover: float = 0.005            # daily volume = 0.5% of float
    p_manipulator: float = 0.8
    cash_range: tuple = (1e5, 1e9)
    own_range: tuple = (1e-4, 0.15)
    max_push: float = 0.05
    min_push: float = 1e-4
    dead_zone: float = 0.1
    hist_len: int = 20
    avg_windows: tuple = (20, 60)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def loguniform(rng, lo, hi):
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def lognorm(x, lo, hi):
    return float((np.log(x) - np.log(lo)) / (np.log(hi) - np.log(lo)))


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def ret_feature(r, cfg):
    """A return as an input: sign(r) * log(1 + |r| / mu_max). True trends land in [-0.7, 0.7];
    much bigger moves (fakes, own price impact) stay readable without swamping the inputs."""
    return np.sign(r) * np.log1p(np.abs(r) / cfg.mu_max)


def push_from_action(y, cfg):
    a = abs(float(y))
    if a < cfg.dead_zone:
        return 0.0
    frac = (1.0 - min(a, 1.0)) / (1.0 - cfg.dead_zone)
    return float(np.sign(y)) * cfg.max_push * (cfg.min_push / cfg.max_push) ** frac


def sample_market(rng, cfg, force_manipulator=False):
    present = force_manipulator or bool(rng.random() < cfg.p_manipulator)
    m = dict(mu=float(rng.uniform(-cfg.mu_max, cfg.mu_max)),
             eta=loguniform(rng, *cfg.eta_range),
             F=loguniform(rng, *cfg.float_range),
             present=present,
             B=loguniform(rng, *cfg.cash_range),
             own=loguniform(rng, *cfg.own_range))
    return m


def market_list(n, seed, cfg, force_manipulator=False):
    """A fixed, reproducible list of markets. The normal and the robust trader are trained on
    the same list, in the same order."""
    rng = np.random.default_rng(seed)
    return [sample_market(rng, cfg, force_manipulator) for _ in range(n)]


# ---------------------------------------------------------------------------
# The market
# ---------------------------------------------------------------------------
class Market:
    def __init__(self, cfg):
        self.cfg = cfg

    def reset(self, m, manipulator_on=True):
        c = self.cfg
        self.m = m
        self.mu, self.eta, self.F = m["mu"], m["eta"], m["F"]
        self.active = bool(m["present"] and manipulator_on)
        self.B = m["B"] if self.active else 0.0
        self.q0 = m["own"] * m["F"] if self.active else 0.0
        self.adv = c.turnover * self.F                          # daily volume (shares)
        self.t, self.V, self.I = 0, c.P0, 0.0
        self.hist = [c.P0]                                      # observed prices
        self.a_cash, self.a_sh, self.last_trade = c.capital, 0.0, 0.0
        self.d_cash, self.d_sh = self.B, self.q0
        return self

    @property
    def price(self):
        return self.V * np.exp(self.I)

    def wealth(self, price=None):
        return self.a_cash + self.a_sh * (self.price if price is None else price)

    def invested(self, price=None):
        p = self.price if price is None else price
        w = self.a_cash + self.a_sh * p
        return self.a_sh * p / w if w > 0 else 0.0

    def capacity(self):
        """Largest fake one-day move the manipulator could create (spend all cash / sell all
        shares). For analysis only: no reward or rule uses it."""
        if not self.active:
            return 0.0, 0.0
        pump = self.eta * np.sqrt(min(self.B / self.cfg.P0, self.F - self.q0) / self.adv)
        dump = self.eta * np.sqrt(self.q0 / self.adv)
        return float(pump), float(dump)

    # -- orders
    def _trader_order(self, x, P):
        c = self.cfg
        x = clamp(float(x), -1.0, 1.0)
        if abs(x) < c.min_trade:
            return 0.0
        w = self.wealth(P)
        if x > 0:
            dollars = min(x * w, self.a_cash / (1 + c.fee))
            return dollars / P if dollars >= c.min_trade * w else 0.0
        shares = min(-x * w / P, self.a_sh)
        if self.a_sh - shares < 1e-9 * self.a_sh + 1e-12:
            shares = self.a_sh
        return -shares if (shares * P >= c.min_trade * w or shares == self.a_sh) else 0.0

    def _manip_order(self, y, P):
        if not self.active:
            return 0.0
        push = push_from_action(clamp(float(y), -1.0, 1.0), self.cfg)
        if push == 0.0:
            return 0.0
        shares = self.adv * (abs(push) / self.eta) ** 2
        if push > 0:
            return min(shares, self.d_cash / (P * (1 + self.cfg.fee)), max(self.F - self.d_sh, 0.0))
        return -min(shares, self.d_sh)

    def _impact(self, Q):
        return self.eta * np.sign(Q) * np.sqrt(abs(Q) / self.adv) if Q != 0 else 0.0

    # -- one trading day
    def step(self, x, y):
        c = self.cfg
        P = self.price
        w_before = self.wealth(P)
        qa, qd = self._trader_order(x, P), self._manip_order(y, P)
        for _ in range(2):                                      # buyers must afford the price
            px = self.V * np.exp(self.I + 2 / 3 * self._impact(qa + qd))
            if qa > 0:
                qa = min(qa, self.a_cash / (px * (1 + c.fee)))
            if qd > 0:
                qd = min(qd, self.d_cash / (px * (1 + c.fee)))
        dI = self._impact(qa + qd)
        px = self.V * np.exp(self.I + 2 / 3 * dI)
        self.a_cash = max(self.a_cash - qa * px - c.fee * abs(qa) * px, 0.0)
        self.a_sh += qa
        self.d_cash = max(self.d_cash - qd * px - c.fee * abs(qd) * px, 0.0)
        self.d_sh += qd
        self.last_trade = qa * px / w_before

        self.I = c.rho * (self.I + dI)
        self.V *= 1.0 + self.mu
        self.t += 1
        self.hist.append(self.price)
        done = self.t >= c.T
        w_after = self.wealth(self.V if done else None)          # true value on the last day
        return dict(reward=100.0 * np.log(w_after / w_before), loss=w_before - w_after,
                    done=done, qa=qa, qd=qd, px=px)

    # -- what each player sees
    def _avg(self, window=None):
        n = len(self.hist) - 1
        if n == 0:
            return 0.0
        w = n if window is None else min(window, n)
        return float(ret_feature(np.log(self.hist[-1] / self.hist[-1 - w]) / w, self.cfg))

    def trader_obs(self):
        c = self.cfg
        h = np.asarray(self.hist[-(c.hist_len + 1):])
        rets = np.zeros(c.hist_len)
        if len(h) > 1:
            rets[:len(h) - 1] = ret_feature(h[1:] / h[:-1] - 1.0, c)[::-1]   # most recent first
        obs = [min(len(self.hist) - 1, c.hist_len) / c.hist_len, self.invested(),
               (c.T - self.t) / c.T, self.last_trade, self._avg()] \
            + [self._avg(w) for w in c.avg_windows] + rets.tolist() \
            + [lognorm(self.eta, *c.eta_range), lognorm(self.F, *c.float_range)]
        return np.array(obs, dtype=np.float32)

    def manip_obs(self):
        c = self.cfg
        last = float(ret_feature(self.hist[-1] / self.hist[-2] - 1.0, c)) if len(self.hist) > 1 else 0.0
        obs = [self.mu / c.mu_max, last, float(len(self.hist) > 1), self.invested(),
               (c.T - self.t) / c.T, self.I / c.max_push, np.tanh((self.d_sh - self.q0) / self.adv),
               self.d_cash / self.B if self.B > 0 else 0.0,
               lognorm(self.eta, *c.eta_range), lognorm(self.F, *c.float_range),
               lognorm(self.m["B"], *c.cash_range), lognorm(self.m["own"], *c.own_range),
               self.last_trade]
        return np.array(obs, dtype=np.float32)


def trader_obs_dim(cfg):
    return 5 + len(cfg.avg_windows) + cfg.hist_len + 2


MANIP_OBS_DIM = 13


# ---------------------------------------------------------------------------
# Players as functions of an observation
# ---------------------------------------------------------------------------
class Player:
    """Wraps a trained stable-baselines3 model (or nothing = passive). Acts on one
    observation or on a batch. `model` can be swapped while training."""

    def __init__(self, model=None):
        self.model = model

    def __call__(self, obs):
        if self.model is None:
            return 0.0
        return float(np.asarray(self.model.predict(obs, deterministic=True)[0]).reshape(-1)[0])

    def batch(self, obs_list):
        if self.model is None:
            return np.zeros(len(obs_list))
        a, _ = self.model.predict(np.stack(obs_list), deterministic=True)
        return np.asarray(a, dtype=float).reshape(len(obs_list), -1)[:, 0]


# ---------------------------------------------------------------------------
# Training environments
# ---------------------------------------------------------------------------
class TraderEnv(gym.Env):
    """The trader's view. Plays the given markets in order. The manipulator is a fixed Player
    (frozen while the trader learns); Player(None) = no manipulation."""

    def __init__(self, cfg, markets, manipulator=None):
        super().__init__()
        self.cfg, self.markets = cfg, markets
        self.manipulator = manipulator or Player(None)
        self.market = Market(cfg)
        self.k = 0                                            # next market in the list
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(trader_obs_dim(cfg),),
                                            dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.market.reset(self.markets[self.k % len(self.markets)])
        self.k += 1
        return self.market.trader_obs(), {}

    def step(self, action):
        y = self.manipulator(self.market.manip_obs())
        out = self.market.step(float(np.asarray(action).reshape(-1)[0]), y)
        # the episode ends because time runs out (a time limit, not a true end state)
        return self.market.trader_obs(), out["reward"], False, out["done"], {}


class ManipulatorEnv(gym.Env):
    """The manipulator's view, on its own markets (always with a manipulator). The trader is a
    fixed Player. A shadow copy of each market runs without manipulation; the reward is the
    trader's extra loss, in % of its starting capital (worst case: costs don't count)."""

    def __init__(self, cfg, seed, trader):
        super().__init__()
        self.cfg, self.trader = cfg, trader
        self.rng = np.random.default_rng(seed)
        self.market, self.shadow = Market(cfg), Market(cfg)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(MANIP_OBS_DIM,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        m = sample_market(self.rng, self.cfg, force_manipulator=True)
        self.market.reset(m)
        self.shadow.reset(m, manipulator_on=False)
        return self.market.manip_obs(), {}

    def step(self, action):
        out = self.market.step(self.trader(self.market.trader_obs()), float(np.asarray(action).reshape(-1)[0]))
        ref = self.shadow.step(self.trader(self.shadow.trader_obs()), 0.0)
        reward = 100.0 * (out["loss"] - ref["loss"]) / self.cfg.capital
        return self.market.manip_obs(), clamp(reward, -100.0, 100.0), False, out["done"], {}


# ---------------------------------------------------------------------------
# Playing many markets at once (testing)
# ---------------------------------------------------------------------------
def play(cfg, trader, manipulator, markets, manipulator_on, log_n=0):
    """Play every market in `markets` with frozen players, all side by side (one batched
    network call per day). Returns one summary per market, and a day-by-day log of both
    players for the first `log_n` markets."""
    T, n = cfg.T, len(markets)
    games = [Market(cfg).reset(m, manipulator_on) for m in markets]
    caps = [g.capacity() for g in games]
    inv, trade = np.zeros((n, T)), np.zeros((n, T))
    rows = []
    for t in range(T):
        xs = trader.batch([g.trader_obs() for g in games])
        ys = manipulator.batch([g.manip_obs() for g in games]) if manipulator_on else np.zeros(n)
        for i, g in enumerate(games):
            if i < log_n:
                pre = dict(market=i, day=t + 1, price=g.price, true_value=g.V, distortion=g.I,
                           trader_cash=g.a_cash, trader_shares=g.a_sh, trader_wealth=g.wealth(),
                           manip_cash=g.d_cash, manip_shares=g.d_sh)
            w = g.wealth()
            out = g.step(xs[i], ys[i])
            inv[i, t] = g.invested(g.V if out["done"] else None)
            trade[i, t] = out["qa"] * out["px"] / w
            if i < log_n:
                rows.append(dict(pre, trader_action=float(xs[i]), trader_trade=trade[i, t],
                                 invested_after=inv[i, t], manip_action=float(ys[i]),
                                 manip_push=push_from_action(ys[i], cfg) if g.active else 0.0,
                                 manip_shares_traded=out["qd"], exec_price=out["px"]))
    out = []
    for i, (g, m) in enumerate(zip(games, markets)):
        f = inv[i]
        commit = next((k + 1 for k in range(T) if (f[k:k + 5] > 0.5).all()), T)
        out.append(dict(market=i, mu=m["mu"], eta=m["eta"], F=m["F"],
                        manipulator=g.active, B=m["B"] if m["present"] else 0.0,
                        own=m["own"] if m["present"] else 0.0,
                        pump_capacity=caps[i][0], dump_capacity=caps[i][1],
                        trader_return=g.wealth(g.V) / cfg.capital - 1.0,
                        best_possible=(1 + m["mu"]) ** (T - 1) - 1 if m["mu"] > 0 else 0.0,
                        avg_invested=float(f.mean()), final_invested=float(f[-1]),
                        day1_bet=float(f[0]), days_to_commit=int(commit),
                        n_trades=int((np.abs(trade[i]) > 1e-6).sum())))
    return out, rows
