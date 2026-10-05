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

# %%
best = 0
for alice_name, alice_fn in bg.CLASSICAL_PLAYERS.items():
    for bob_name, bob_fn in bg.CLASSICAL_PLAYERS.items():
        rate = bg.play_classical(alice_fn, bob_fn)["win_rate"]
        print(f"{alice_name:12s} {bob_name:12s} {rate:.2f}")
        best = max(best, rate)
print("best classical win rate:", best)

# %% [markdown]
# No pair beats **0.75**. (Shared random numbers don't help either: a random
# strategy is just a mix of these sixteen, so it can't beat the best one.)
# Keep this number in mind: quantum players will beat it in lab 03.
