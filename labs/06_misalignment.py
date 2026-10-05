# %% [markdown]
# # Lab 06: Misalignment and optimization
#
# **Python skill:** passing a function to another function (`scipy.optimize.minimize`).
# **Physics idea:** fiber twists polarization; you can undo it if you can find the right correction.
#
# A long fiber slowly rotates the polarization in a way nobody knows. Below
# we twist both photons by random angles and S collapses.

# %%
import matplotlib.pyplot as plt
from scipy.optimize import minimize

import bellgame as bg

twist = bg.random_misalignment(seed=4)
print("hidden twist (degrees):", twist)
twisted_pair = bg.misaligned_source(bg.bell_pair(), twist)
print("S with the twist:", round(bg.play_chsh(bg.optimal_strategy(), twisted_pair)["S"], 3))

# %% [markdown]
# Alice can put three wave plates in front of her polarizer: three correction
# angles. Which angles? We don't know the twist, we only see S.
# That's an **optimization** problem: find the angles that make S biggest.
#
# `bg.chsh_objective` makes a function `f(angles)` that returns **-S**
# (minus, because scipy *minimizes*). Hand it to scipy:

# %%
f = bg.chsh_objective(twisted_pair)
print("f at no correction:", f([0, 0, 0]))

result = minimize(f, x0=[0, 0, 0], method="Nelder-Mead")
print("best angles:", result.x)
print("S after correction:", -result.fun)
print("number of tries:", result.nfev)

# %% [markdown]
# **Your turn:** try `method="COBYQA"` and `method="Powell"`. Which needs the
# fewest tries? Does the starting point `x0` matter?

# %% [markdown]
# ## With shot noise
#
# A real lab only gets S from a finite number of rounds, so every value `f`
# returns is a little wrong. `rounds=2500` simulates that.
# `bg.find_alignment` wraps the whole search for you.

# %%
noisy = bg.find_alignment(twisted_pair, method="Nelder-Mead", rounds=2500, seed=1,
                          max_evaluations=150)
print("best S seen during the search:", round(noisy["best_seen_S"], 3))
print("true S at the angles it chose:", round(noisy["S"], 3))

# %% [markdown]
# The best S *seen* is usually higher than the true S: with noisy data, the
# biggest number you saw is partly luck. That's why you always re-check.
#
# ## Racing optimizers
#
# `bg.compare_optimizers` repeats this for several random twists and records
# the best-so-far S over time, like the original study.

# %%
race = bg.compare_optimizers(bg.bell_pair(), runs=4, rounds=2500, max_evaluations=150, seed=0)
ax = None
for method, data in race.items():
    ax = bg.plot_convergence(data["histories"], ax=ax, label=method)
    print(method, "validated S:", [round(s, 3) for s in data["validated_S"]])
plt.show()
