# %% [markdown]
# # Lab 01: The CHSH game, played classically
#
# **Python skill:** writing and calling functions.
# **Physics idea:** a game that no classical team can win more than 75% of the time.
#
# A referee gives Alice a random bit `x` and Bob a random bit `y`. They can't talk.
# Alice answers a bit `a`, Bob answers a bit `b`. They **win** when
#
#     a XOR b  ==  x AND y
#
# In words: their answers should be *different* only when both questions are 1.

# %%
import numpy as np

import bellgame as bg

# The referee's rule, as a function. Try a few cases by hand first!
print(bg.referee_wins(x=0, y=0, a=0, b=0))   # x AND y = 0, answers equal -> win
print(bg.referee_wins(x=1, y=1, a=0, b=0))   # x AND y = 1, answers equal -> lose


# %% [markdown]
# ## A strategy is a function
#
# A classical player is just a function from the question bit to an answer bit.

# %%
def alice(x):
    return 0


def bob(y):
    return 0


result = bg.play_classical(alice, bob)
print("win rate:", result["win_rate"])

# %% [markdown]
# ## Your turn
#
# 1. Change `alice` and `bob` above. Can you beat 75%?
# 2. There are only four functions from one bit to one bit. bellgame has them in
#    `bg.CLASSICAL_PLAYERS`. The loop below tries every pair.

def alice(x):
    return int(round(np.random.uniform(), 0))

def bob(y):
    return int(round(np.random.uniform(), 0))

# Players who flip coins have no single exact answer, so play an experiment.
result = bg.play_classical(alice, bob, rounds=10_000, seed=1)
print("win rate", result["win_rate"])

# %%
best = 0
header = ["Alice Strat", "Bob Strat", "Win Rate"]
print(f"{header[0]:12} {header[1]:12} {header[2]}")
for alice_name, alice_fn in bg.CLASSICAL_PLAYERS.items():
    for bob_name, bob_fn in bg.CLASSICAL_PLAYERS.items():
        rate = bg.play_classical(alice_fn, bob_fn)["win_rate"]
        print(f"{alice_name:12s} {bob_name:12s} {rate:.2f}")
        best = max(best, rate)
print("best classical win rate:", best)

# %% [markdown]
# No pair beats **0.75**. (Shared random numbers don't help either: a random
# strategy is just a mix of these sixteen, so it can't beat the best one.)
#
# Memory doesn't help either. A player can also be `player(bit, history)`, where
# `history` is their own past rounds as `(question, answer)` pairs; see
# `examples/classical_strategies.py`. While the referee's questions are fair coin
# flips, the past says nothing about this round's questions. The only way past
# 75% is a referee whose questions can be predicted.
# Keep this number in mind: quantum players will beat it in lab 03.
