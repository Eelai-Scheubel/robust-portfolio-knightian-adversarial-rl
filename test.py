"""
test.py: compare the normal and the robust trader on the same 10,000 new markets
===============================================================================

Nothing learns any more. Both traders play the SAME 10,000 test markets (drawn from a seed
different from training), twice:
  clean     : manipulator switched off everywhere        -> how each trader invests normally
  attacked  : the trained manipulator active where present -> what robustness protects against

Outputs in results/:
  episodes.csv        one row per (scenario, trader, market): market parameters, return,
                      % invested, day-1 bet, day it commits, number of trades
  daily_log.csv.gz    day-by-day log of both players for the first 500 markets
  investment.png      % invested day by day; decision maps (true trend x day)
  outcomes.png        % invested vs true trend; distribution of returns under attack

Usage:  python test.py            (expects models/normal.zip, models/robust.zip, models/manipulator.zip)
"""

import argparse
import os
import time

import numpy as np
import pandas as pd
from stable_baselines3 import SAC

from market import Config, Player, market_list, play

COLORS = {"normal": "#2a78d6", "robust": "#eb6834"}
INK, INK2, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7", "#fcfcfb"


def run(cfg, traders, manipulator, markets, log_n, chunk=2000):
    eps, logs = [], []
    for scenario, on in (("clean", False), ("attacked", True)):
        for name, trader in traders.items():
            t0 = time.time()
            for k in range(0, len(markets), chunk):
                summ, rows = play(cfg, trader, manipulator, markets[k:k + chunk], on, max(log_n - k, 0))
                for s in summ:
                    s.update(scenario=scenario, trader=name, market=k + s["market"])
                for r in rows:
                    r.update(scenario=scenario, trader=name, market=k + r["market"])
                eps += summ
                logs += rows
            print(f"  {scenario:<9} {name:<7} {time.time() - t0:6.0f}s", flush=True)
    return pd.DataFrame(eps), pd.DataFrame(logs)


def report(ep, cfg):
    ep = ep.copy()
    ep["trend"] = np.where(ep.mu > 0, "rising", "falling")
    pct = lambda s: 100 * s
    pd.set_option("display.width", 200)

    print("\n=== CLEAN (no manipulator): how each trader invests ===")
    c = ep[ep.scenario == "clean"]
    t = c.groupby(["trend", "trader"]).agg(
        avg_invested=("avg_invested", lambda s: pct(s.mean())),
        final_invested=("final_invested", lambda s: pct(s.mean())),
        day1_bet=("day1_bet", lambda s: pct(s.mean())),
        days_to_commit=("days_to_commit", "median"),
        never_commits=("days_to_commit", lambda s: pct((s >= cfg.T).mean())),
        trades=("n_trades", "mean"),
        avg_return=("trader_return", lambda s: pct(s.mean())),
        best_possible=("best_possible", lambda s: pct(s.mean())))
    print(t.round(2).to_string())
    print("(all in %, except days and trades; best_possible = watch one day, then all-in if rising)")

    edges = np.array([-1, -.5, -.25, -.05, 0, .05, .25, .5, 1]) * 100 * cfg.mu_max
    c = c.assign(trend_bin=pd.cut(100 * c.mu, edges))
    print("\n% invested by true daily trend (%):")
    print(c.pivot_table(index="trend_bin", columns="trader", values="avg_invested", aggfunc="mean",
                        observed=True).mul(100).round(1).to_string())

    print("\n=== CLEAN vs ATTACKED: returns (%), including the worst outcomes ===")
    g = ep.groupby(["scenario", "trader"]).trader_return
    t = pd.DataFrame({"average": pct(g.mean()), "median": pct(g.median()),
                      "worst 5% (avg)": pct(g.apply(lambda s: s[s <= s.quantile(0.05)].mean())),
                      "worst 1%": pct(g.quantile(0.01)),
                      "worst": pct(g.min())})
    print(t.round(2).to_string())
    print("\n% invested (episode average), by scenario:")
    print(ep.pivot_table(index=["scenario", "trend"], columns="trader", values="avg_invested",
                         aggfunc="mean").mul(100).round(1).to_string())


def style(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.set_xlabel(xlabel, color=INK2)
    ax.set_ylabel(ylabel, color=INK2)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors="#898781")


def plots(ep, logs, cfg, out):
    import matplotlib.pyplot as plt
    names = [n for n in ("normal", "robust") if n in set(ep.trader)]
    clean = logs[logs.scenario == "clean"]
    edges = np.linspace(-100 * cfg.mu_max, 100 * cfg.mu_max, 17)

    # 1. % invested day by day + decision maps
    fig = plt.figure(figsize=(15, 9), facecolor=SURFACE, layout="constrained")
    gs = fig.add_gridspec(2, 2)
    for j, (label, sel) in enumerate((("rising stocks", clean.mu > 0), ("falling stocks", clean.mu <= 0))):
        ax = fig.add_subplot(gs[0, j])
        for n in names:
            s = clean[sel & (clean.trader == n)].groupby("day").invested_after.mean() * 100
            ax.plot(s.index, s.values, lw=2, color=COLORS[n], label=n)
        style(ax, f"% of wealth invested, day by day ({label}, no manipulator)", "day", "% invested")
        ax.set_ylim(-3, 103)
        ax.legend(frameon=False, fontsize=9)
    sub = clean.assign(trend_bin=pd.cut(100 * clean.mu, edges))
    im = None
    for j, n in enumerate(names):
        ax = fig.add_subplot(gs[1, j])
        M = sub[sub.trader == n].pivot_table(index="trend_bin", columns="day", values="invested_after",
                                             aggfunc="mean", observed=False) * 100
        im = ax.pcolormesh(np.arange(M.shape[1] + 1) + 0.5, edges, M.values, cmap="Blues",
                           vmin=0, vmax=100, shading="flat")
        style(ax, f"{n} trader: % invested by true trend and day", "day", "true daily trend (%)")
        ax.axhline(0, color=AXIS, lw=1)
        ax.grid(False)
    if im is not None:
        fig.colorbar(im, ax=fig.axes[2:], label="% of wealth invested")
    fig.savefig(f"{out}/investment.png", dpi=130, facecolor=SURFACE)

    # 2. % invested vs trend; return distribution under attack
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), facecolor=SURFACE, layout="constrained")
    centers = (edges[:-1] + edges[1:]) / 2
    c = ep[ep.scenario == "clean"]
    for n in names:
        s = c[c.trader == n]
        g = s.groupby(pd.cut(100 * s.mu, edges), observed=False).avg_invested.mean() * 100
        axes[0].plot(centers, g.values, "o-", lw=2, ms=6, color=COLORS[n], label=n,
                     markeredgecolor=SURFACE, markeredgewidth=1.5)
    style(axes[0], "Episode-average % invested vs true trend (no manipulator)", "true daily trend (%)",
          "% invested")
    att = ep[ep.scenario == "attacked"]
    lo = 100 * att.trader_return.quantile(0.001)
    bins = np.linspace(min(lo, -1), 100 * att.trader_return.max(), 80)
    for n in names:
        axes[1].hist(100 * att[att.trader == n].trader_return, bins=bins, color=COLORS[n], lw=2,
                     histtype="step", label=n, log=True)
    style(axes[1], "Returns with the manipulator active (log count)", "return over the year (%)",
          "markets")
    for ax in axes:
        ax.legend(frameon=False, fontsize=9)
    fig.savefig(f"{out}/outcomes.png", dpi=130, facecolor=SURFACE)
    print(f"saved {out}/investment.png and {out}/outcomes.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--markets", type=int, default=10_000)
    ap.add_argument("--log-markets", type=int, default=500)
    ap.add_argument("--seed", type=int, default=777, help="test markets (different from training)")
    ap.add_argument("--models", default="models")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    cfg = Config()
    os.makedirs(a.out, exist_ok=True)
    traders = {n: Player(SAC.load(f"{a.models}/{n}")) for n in ("normal", "robust")
               if os.path.exists(f"{a.models}/{n}.zip")}
    manipulator = Player(SAC.load(f"{a.models}/manipulator"))
    markets = market_list(a.markets, seed=a.seed, cfg=cfg)
    print(f"testing {list(traders)} on {a.markets} markets x {cfg.T} days")

    ep, logs = run(cfg, traders, manipulator, markets, a.log_markets)
    mcols = ["market", "mu", "eta", "F"]
    logs = logs.merge(ep[ep.scenario == "clean"][mcols].drop_duplicates("market"), on="market")
    ep.to_csv(f"{a.out}/episodes.csv", index=False)
    logs.to_csv(f"{a.out}/daily_log.csv.gz", index=False, compression="gzip")
    print(f"saved {a.out}/episodes.csv ({len(ep)} rows), {a.out}/daily_log.csv.gz ({len(logs)} rows)")
    report(ep, cfg)
    try:
        plots(ep, logs, cfg, a.out)
    except ImportError:
        print("(matplotlib not installed: no plots)")


if __name__ == "__main__":
    main()
