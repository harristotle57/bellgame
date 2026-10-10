# %% [markdown]
# # Tutorial 01: The CHSH game
#
# **New in bellgame:** `referee_wins`, `play_classical`, `CLASSICAL_PLAYERS`,
# `bell_pair`, `optimal_strategy`, `play_chsh` (exact and with `rounds=`), and the plots.
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

print(bg.referee_wins(x=0, y=0, a=0, b=0))   # x AND y = 0, answers equal: win
print(bg.referee_wins(x=1, y=1, a=0, b=0))   # x AND y = 1, answers equal: lose

# %% [markdown]
# ## Classical players
#
# A classical player is a function from the question bit to an answer bit. There
# are only four (`bg.CLASSICAL_PLAYERS`), so we can try every team.
# `bg.play_classical` gives the exact win rate.

# %%
print(f"{'Alice':12} {'Bob':12} win rate")
for alice_name, alice in bg.CLASSICAL_PLAYERS.items():
    for bob_name, bob in bg.CLASSICAL_PLAYERS.items():
        print(f"{alice_name:12} {bob_name:12} {bg.play_classical(alice, bob)['win_rate']:.2f}")
print("classical bound:", bg.CLASSICAL_WIN_RATE, " S <=", bg.CLASSICAL_S)

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
bg.plot_angles(strategy)
plt.show()

result = bg.play_chsh(strategy, bg.bell_pair())
print("win rate:", round(result["win_rate"], 4), " S:", round(result["S"], 4))
print("Tsirelson bound:", round(bg.TSIRELSON_WIN_RATE, 4), " S =", round(bg.TSIRELSON_S, 4))
bg.plot_correlations(result)
plt.show()

# %% [markdown]
# **85.4%** beats every classical team, and no quantum strategy does better.
# S measures the same thing on another scale (win rate = 1/2 + S/8): S <= 2
# classically, up to 2√2 ≈ 2.83 with entanglement.
#
# ## A real experiment only estimates S
#
# Without `rounds=`, `play_chsh` returns the exact expected values. With
# `rounds=`, it samples that many rounds, like a lab would. How much does the
# measured S wobble?

# %%
sizes = [100, 1_000, 10_000]
fig, ax = plt.subplots(figsize=(7, 4))
for i, n in enumerate(sizes):
    s_values = [bg.play_chsh(strategy, bg.bell_pair(), rounds=n, seed=k)["S"] for k in range(200)]
    print(f"{n:6,} rounds: S = {np.mean(s_values):.3f} +- {np.std(s_values):.3f}")
    ax.hist(s_values, bins=30, alpha=0.6, color=bg.plots.SERIES[i], label=f"{n:,} rounds")
ax.axvline(bg.CLASSICAL_S, color="gray", linestyle="--")
ax.axvline(bg.TSIRELSON_S, color="gray", linestyle=":")
ax.set_xlabel("measured S")
ax.set_ylabel("number of experiments")
ax.legend(frameon=False)
plt.show()

# %% [markdown]
# The spread shrinks like 1/√rounds. To *claim* a violation you need S - 2 to
# be several error bars, and this is why E91 needs many test rounds.
#
# `bg.plot_results` shows one sampled experiment: the measured win rate for
# each question pair.

# %%
sampled = bg.play_chsh(strategy, bg.bell_pair(), rounds=2000, seed=3)
bg.plot_results(sampled)
plt.show()

# %% [markdown]
# ## Your turn
#
# 1. Rotate Bob's angles: `{"alice": [0, 45], "bob": [22.5 + t, -22.5 + t]}`.
#    How fast does S fall with `t`? (Tutorial 03 is about exactly this.)
# 2. With 100 rounds, how often does a perfect Bell pair *appear* to give S < 2?
#    How often does the best classical team appear to give S > 2?
