# %% [markdown]
# # Lab 02: Playing many rounds
#
# **Python skill:** loops and lists.
# **Physics idea:** an experiment only *estimates* a probability; more rounds means a better estimate.
#
# In lab 01 bellgame computed the exact win rate. A real experiment plays a
# finite number of rounds, so the measured win rate wobbles.

# %%
import random

import matplotlib.pyplot as plt

import bellgame as bg


def alice(x):
    return 0


def bob(y):
    return 0


# Play 1000 rounds by hand, with a loop.
random.seed(1)
wins = 0
for _ in range(1000):
    x = random.randint(0, 1)
    y = random.randint(0, 1)
    if bg.referee_wins(x, y, alice(x), bob(y)):
        wins += 1
print("won", wins, "of 1000 rounds")

# %% [markdown]
# bellgame can do the same thing for you: pass `rounds=` (and a `seed=` so you
# get the same random numbers every time you run it).

# %%
result = bg.play_classical(alice, bob, rounds=1000, seed=1)
print(result["win_rate"], result["counts"])

# %% [markdown]
# ## How much does the estimate wobble?
#
# Run 200 experiments of each size and collect the win rates in a list.

# %%
sizes = [10, 100, 1000]
fig, ax = plt.subplots(figsize=(6, 4))
for size in sizes:
    rates = []
    for seed in range(200):
        rates.append(bg.play_classical(alice, bob, rounds=size, seed=seed)["win_rate"])
    ax.hist(rates, bins=30, alpha=0.6, label=f"{size} rounds")
ax.axvline(0.75, color="gray", linestyle="--")
ax.set_xlabel("measured win rate")
ax.set_ylabel("number of experiments")
ax.legend(frameon=False)
plt.show()

# %% [markdown]
# **Question:** with 100 rounds, how often does a classical team *appear* to
# beat 75%? Count it with a loop. Why does this matter when you want to prove
# you have quantum entanglement?
