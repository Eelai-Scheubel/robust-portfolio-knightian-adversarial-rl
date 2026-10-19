# Normal vs robust trader in a very simplified market

**Question:** how does a trader trained against a worst-case manipulator invest, compared with
an identical trader trained without one?

## Files

| File | What it does |
|---|---|
| `market.py` | The market, both players, the training environments, and fast batched play for testing |
| `train.py` | Trains the normal trader, or the robust trader together with its manipulator |
| `test.py` | Plays both traders on the same 10,000 new markets, with and without manipulation |

## Run

```
pip install numpy pandas gymnasium stable-baselines3 matplotlib
python train.py normal       # ~1-2 h
python train.py robust       # ~3-4 h   (independent of the first: can run at the same time)
python test.py               # ~10 min
```

Options: `--episodes` (training markets per trader, default 3,000), `--block` (episodes per
turn, default 50), `--seed`; for `test.py`: `--markets` (default 10,000), `--log-markets`
(default 500).

## The market (one episode = 260 trading days)

- **Price = true value × distortion.** The true value moves by a constant daily trend μ,
  drawn between −0.0943% and +0.0943% a day (±2% a month, about ±27% a year). The trader never
  sees μ.
- **Trading moves the price:** a net order of Q shares adds η·√(Q / daily volume) to the
  distortion. Half the distortion fades every day. 0.1% fee per trade.
- **Random per episode:** η (how easily the price moves, 0.5–5%), float (10M–1B shares; daily
  volume = 0.5% of it). 80% of episodes have a manipulator, with random cash ($100k–$1B) and
  shares (0.01–15% of the float).

## The players

**Trader:** each day, trades a fraction of its wealth (−100% sell all … +100% buy all). It sees
its last 20 daily returns, its average return over the last 20 days, 60 days and since day 1,
how invested it is, its last trade, days left, and the stock's η and float. **Nothing about the
manipulator.** Reward: the daily growth of its wealth (log), so it learns only from its own
money. On the last day its shares are valued at the true value.

**Manipulator (worst case):** each day, chooses a price push (0.01%–5% up or down, or nothing).
The market converts it into shares; its cash limits buying, its shares limit selling (no
shorting). It sees everything, including μ and the trader's position. Reward: the trader's loss
compared with a shadow copy of the same market without manipulation. Its own costs don't count.

Both players use SAC (stable-baselines3), a small neural network (two layers of 128).

## Training: the controlled comparison

- The 3,000 training markets are generated once from `--seed` as a fixed list.
- **Normal trader:** plays the list in order, with no manipulation.
- **Robust trader:** plays **the same list, in the same order, from the same starting
  network**. In the 80% of markets that have a manipulator, the manipulator acts. The only
  difference between the two traders is the manipulator.
- **Turns:** every 50 episodes, the manipulator learns for 50 episodes on its own separate
  markets (trader frozen), then the trader learns for 50 episodes of the shared list
  (manipulator frozen). They never learn at the same time.
- No checkpoint selection: both traders are saved at the end.

## Test

Both frozen traders play the same 10,000 new markets (seed 777), twice: **clean** (no
manipulator anywhere) and **attacked** (the final manipulator active where present).

| Output (`results/`) | Content |
|---|---|
| `episodes.csv` | one row per scenario × trader × market: μ, η, float, manipulator cash and shares, capacity, return, % invested (average and final), day-1 bet, day it commits (stays >50% invested for 5 days), number of trades |
| `daily_log.csv.gz` | day by day for the first 500 markets: price, true value, distortion, both players' actions, trades, cash and shares |
| `investment.png` | % invested day by day (rising / falling stocks), and decision maps (true trend × day) |
| `outcomes.png` | % invested vs true trend; distribution of returns under attack |

The printed tables cover how each trader invests without manipulation, and its returns
(average, median, worst 5%, worst 1%, worst) with and without manipulation.
