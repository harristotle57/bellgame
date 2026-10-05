# %% [markdown]
# # Lab 03: The quantum strategy
#
# **Python skill:** dictionaries.
# **Physics idea:** entangled photons let Alice and Bob win 85.4% of the time.
#
# Alice and Bob now share a pair of entangled photons (a Bell pair). Instead of
# a function, each player has two **polarizer angles**: one for question 0 and
# one for question 1. The photon passes (answer 0) or is blocked (answer 1).
#
# A strategy is a dict of angles in degrees:

# %%
import matplotlib.pyplot as plt

import bellgame as bg

strategy = {"alice": [0, 45], "bob": [22.5, -22.5]}
print(strategy["alice"])        # Alice's two angles
print(strategy["bob"][1])       # Bob's angle for question y = 1

bg.plot_angles(strategy)
plt.show()

# %% [markdown]
# ## Play it on a perfect Bell pair

# %%
pair = bg.bell_pair()            # a 4x4 numpy array (a density matrix)
result = bg.play_chsh(strategy, pair)
print("win rate:", round(result["win_rate"], 4))
print("S:", round(result["S"], 4))

# %% [markdown]
# **85.4%** beats every classical strategy. The CHSH number **S** says the same
# thing on a different scale: classical teams can't get S above 2, quantum ones
# reach 2√2 ≈ 2.83 (win rate = 1/2 + S/8).

# %%
bg.plot_correlations(result)
plt.show()

# %% [markdown]
# ## Your turn
#
# 1. Change Bob's angles. Can you find a better strategy? A worse one?
# 2. Real pairs aren't perfect. `bg.bell_pair(fidelity=0.9)` is a noisier pair.
#    Use a loop to find the fidelity where S drops to 2.

# %%
for f in [1.0, 0.95, 0.9, 0.85, 0.8, 0.75]:
    s = bg.play_chsh(strategy, bg.bell_pair(f))["S"]
    print(f"fidelity {f:.2f}  ->  S = {s:.3f}")

# %% [markdown]
# 3. Simulate an experiment with `rounds=2000, seed=...` and plot it with
#    `bg.plot_results(result)`.

# %%
sampled = bg.play_chsh(strategy, bg.bell_pair(0.95), rounds=2000, seed=3)
bg.plot_results(sampled)
plt.show()
