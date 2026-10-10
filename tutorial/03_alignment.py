# %% [markdown]
# # Tutorial 03: Fiber twist and alignment
#
# **New in bellgame:** `random_misalignment`, `misaligned_source`,
# `chsh_objective` (with `scipy.optimize.minimize`), `find_alignment`, and
# `run_e91(corrections=...)`.
# **Physics:** a long fiber rotates polarization in a way nobody knows. That
# ruins the key, but Alice can undo it with wave plates, if she can find the
# right settings using only the measured S.

# %%
from scipy.optimize import minimize

import bellgame as bg

twist = bg.random_misalignment(seed=4)
print("hidden twist (degrees):", twist)
twisted_pair = bg.misaligned_source(bg.bell_pair(), twist)
e = bg.run_e91(twisted_pair, rounds=None)
print(f"E91 on the twisted pair: QBER = {e['qber']:.3f}, S = {e['S']:.3f}")

# %% [markdown]
# A perfect pair, and still almost half the key bits are wrong.
#
# Alice puts three wave plates in front of her polarizer: three correction
# angles. Nobody knows the twist, so the only way to choose them is to try
# settings and watch S. That's an optimization problem.
#
# `bg.chsh_objective(source)` is an ordinary function `f(angles)` that returns
# **-S** (minus, because scipy minimizes), so any scipy optimizer can use it.

# %%
f = bg.chsh_objective(twisted_pair)
print("f at no correction:", round(f([0, 0, 0]), 3))
result = minimize(f, x0=[0, 0, 0], method="Nelder-Mead")
print("best angles:", result.x.round(2))
print("S after correction:", round(-result.fun, 3), "after", result.nfev, "tries")

# %% [markdown]
# ## With shot noise
#
# In a lab every S comes from a finite number of rounds, so every value the
# optimizer sees is a little wrong. `bg.find_alignment` runs the whole search,
# here with 2500 rounds per try and a budget of 150 tries.

# %%
found = bg.find_alignment(twisted_pair, method="Nelder-Mead", rounds=2500, seed=1, max_evaluations=150)
print("correction:", [round(a, 2) for a in found["correction"]])
print("best S seen during the search:", round(found["best_seen_S"], 3))
print("true S at the angles it chose:", round(found["S"], 3))

# %% [markdown]
# The best S *seen* is usually higher than the true S: among many noisy
# measurements the biggest one is partly luck. Always re-check.
#
# ## The key comes back
#
# Pass the correction to E91.

# %%
after = bg.run_e91(twisted_pair, rounds=None, corrections={"alice_correction": found["correction"]})
print(f"before alignment: QBER = {e['qber']:.3f}, S = {e['S']:.3f}")
print(f"after alignment:  QBER = {after['qber']:.3f}, S = {after['S']:.3f}")

# %% [markdown]
# ## Your turn
#
# 1. Try `method="COBYQA"` and `method="Powell"` in `find_alignment`. Which gets
#    the best true S within 150 tries? (`examples/optimizer_comparison.py` races
#    them properly, over many twists.)
# 2. Fewer rounds per try means a faster search but noisier S. Try `rounds=250`.
#    How much key is left after alignment?
