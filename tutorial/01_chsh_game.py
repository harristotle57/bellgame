# %% [markdown]
# # Tutorial 01: The CHSH game
#
# **New in bellgame:** `referee_wins`, `play` (exact and with `rounds=`), `CLASSICAL_PLAYERS`,
# `bell_pair`, `optimal_strategy`, and the plots.
# **Physics:** classical players win at most 75% of the time (S <= 2); players
# sharing a Bell pair win 85.4% (S = 2√2). This gap is what E91, the goal of
# this tutorial, uses to prove a key is secret.
#
# A referee gives Alice a random bit `x` and Bob a random bit `y`. They can't talk.
# Alice answers a bit `a`, Bob answers a bit `b`. They **win** when
#
#     a XOR b  ==  x AND y
#
# so their answers should differ only when both questions are 1.

# %%
import matplotlib.pyplot as plt
import numpy as np

import bellgame as bg

print(bg.referee_wins(x=0, y=0, a=0, b=0))  # (x AND y = 0) => answers equal: win
print(bg.referee_wins(x=1, y=1, a=0, b=0))  # (x AND y = 1) => answers equal: lose

# %% [markdown]
# ## Classical players
#
# A classical player is a function from the question bit to an answer bit. There
# are only four strategies (`bg.CLASSICAL_PLAYERS`), so we can try every combo
# `bg.play((alice, bob))` gives the exact win rate.

# %%
print(f"{'Alice':12} {'Bob':12} win rate")
for alice_name, alice in bg.CLASSICAL_PLAYERS.items():
    for bob_name, bob in bg.CLASSICAL_PLAYERS.items():
        print(f"{alice_name:12} {bob_name:12} {bg.play((alice, bob))['win_rate']:.2f}")
print("classical bound:", bg.CLASSICAL_WIN_RATE, "(equivalently S <=", bg.CLASSICAL_S, ", since S=8*(win_rate - 1/2))")

# %% [markdown]
# No team beats **0.75**. Shared randomness doesn't help: a random strategy is a
# mix of these sixteen, so it can't beat the best one. Neither does memory
# (players of the form `player(bit, history)`, see `examples/classical_strategies.py`),
# as long as the referee's questions are fresh, private coin flips.
#
# ## Quantum players
#
# Now Alice and Bob share an entangled photon pair. Each has two **polarizer
# angles**, one per question; the photon passes (answer 0) or is blocked
# (answer 1). A strategy is a dict of angles in degrees.

# %%
strategy = bg.optimal_strategy()
print(strategy)
ax = bg.plot_angles(strategy)
ax.set_title(
    "Optimal CHSH strategy\nPolarizer angles (H = 0 deg, V = 90 deg)", color=bg.plots.TEXT, fontsize=11, loc="left"
)
plt.show()

result = bg.play(strategy, bg.bell_pair())
print("win rate:", round(result["win_rate"], 4), " S:", round(result["S"], 4))
print("Tsirelson bound:", round(bg.TSIRELSON_WIN_RATE, 4), " S =", round(bg.TSIRELSON_S, 4))
ax = bg.plot_correlations(result)
ax.set_title(
    f"Correlations E(x, y)    S = {result['S']:.3f}, win rate = {result['win_rate']:.1%}\n"
    "(classical limit 2 / 75%, quantum limit 2.83 / 85.4%)",
    color=bg.plots.TEXT,
    fontsize=11,
    loc="left",
)
plt.show()

# %% [markdown]
# **85.4%** beats every classical team, and no quantum strategy does better.
# S measures the same thing on another scale (win rate = 1/2 + S/8): S <= 2
# classically, up to 2√2 ≈ 2.83 with entanglement.
#
# ## A real experiment only estimates S
#
# Without `rounds=`, `play` returns the exact expected values. With
# `rounds=`, it samples that many rounds, like a lab would. How much does the
# measured S wobble?

# %%
sizes = [100, 1_000, 10_000]
colors = ["#e34948", "#2a78d6", "#52514e"]  # red, blue, dark grey
experiments = 1000
s_values = {n: [bg.play(strategy, bg.bell_pair(), rounds=n, seed=k)["S"] for k in range(experiments)] for n in sizes}

# Shared bins, so a tighter spread shows up as a taller peak
all_s = np.concatenate(list(s_values.values()))
bins = np.arange(np.floor(all_s.min() * 50) / 50, all_s.max() + 0.02, 0.02)
fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
for (n, s), color in zip(s_values.items(), colors):
    print(f"{n:6,} rounds: S = {np.mean(s):.3f} +- {np.std(s):.3f}")
    weights = np.full(experiments, 1 / experiments)
    ax.hist(s, bins=bins, weights=weights, color=color, alpha=0.2)
    ax.hist(s, bins=bins, weights=weights, histtype="step", color=color, linewidth=2, label=f"{n:,} rounds")
ax.axvline(bg.CLASSICAL_S, color=bg.plots.TEXT, linewidth=1, zorder=0, label="classical limit")
ax.axvline(bg.TSIRELSON_S, color=bg.plots.TEXT, linewidth=1, linestyle="-.", zorder=0, label="quantum limit")
ax.set_xlabel("measured S")
ax.set_ylabel("fraction of experiments")
top = ax.secondary_xaxis("top", functions=(lambda s: 0.5 + s / 8, lambda w: 8 * (w - 0.5)))
top.xaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(xmax=1, decimals=0))
top.set_xlabel("win rate (= 1/2 + S/8)")
ax.legend(frameon=False, loc="upper left")
ax.set_title(f"Measured S in {experiments:,} experiments at each size", color=bg.plots.TEXT, fontsize=11, loc="left")
plt.show()

# %% [markdown]
# The spread shrinks like 1/√rounds. To *claim* a violation you need S - 2 to
# be several error bars, and this is why E91 needs many test rounds.
#
# `bg.plot_results` shows one sampled experiment: the measured win rate for
# each question pair.

# %%
sampled = bg.play(strategy, bg.bell_pair(), rounds=2000, seed=3)
_, ax = plt.subplots(figsize=(6, 4), layout="constrained")  # room for the margin labels
bg.plot_results(sampled, ax=ax)
ax.set_title(
    f"Win rate {sampled['win_rate']:.1%} over {sampled['rounds']} rounds\n"
    f"S = {sampled['S']:.3f}  (1/2 + S/8 = {0.5 + sampled['S'] / 8:.1%})",
    color=bg.plots.TEXT,
    loc="left",
)
plt.show()

# %% [markdown]
# ## Your turn
#
# 1. Rotate Bob's angles: `{"alice": [0, 45], "bob": [22.5 + t, -22.5 + t]}`.
#    How fast does S fall with `t`? (Tutorial 03 is about exactly this.)
# 2. With 100 rounds, how often does a perfect Bell pair *appear* to give S < 2?
#    How often does the best classical team appear to give S > 2?
