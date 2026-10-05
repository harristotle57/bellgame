# %% [markdown]
# # Lab 05: Parameter studies
#
# **Python skill:** sweeping a parameter with a loop and plotting with matplotlib.
# **Physics idea:** every imperfection costs you; and how you treat missed photons can fake a result.
#
# The recipe is always the same:
#
# 1. make a list of values to try
# 2. for each value, build a link and play
# 3. plot the results

# %%
import matplotlib.pyplot as plt

import bellgame as bg

strategy = bg.optimal_strategy()
distances = [0, 50, 100, 150, 200, 250]

results = []
for d in distances:
    results.append(bg.play_chsh(strategy, bg.link(distance_km=d, dark_count_rate_hz=10_000)))

ax = bg.plot_sweep(distances, results, metric="S", label="10 kHz dark counts", xlabel="distance (km)")
plt.show()

# %% [markdown]
# Distance alone only lowers the *rate* (fewer photons arrive). It's the dark
# counts, false clicks, that spoil S once real photons become rare.
#
# **Your turn:** add a second curve with `dark_count_rate_hz=100` on the same `ax`
# (pass `ax=ax`). Then plot `metric="coincidence_prob"` instead.

# %% [markdown]
# ## The detection loophole
#
# When neither detector clicks, the player still has to answer something.
# bellgame's `no_click` knob decides:
#
# * `"discard"`: throw the round away (this assumes the lost photons are a fair
#   sample: the **fair-sampling assumption**)
# * `"zero"` or `"random"`: keep every round and answer anyway
#
# Without discarding, you need very good detectors. We use an idealized
# `heralded=True` source that only counts pulses with a real pair.

# %%
efficiencies = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0]
fig, ax = plt.subplots(figsize=(6, 4))
for policy in ["discard", "zero", "random"]:
    rs = []
    for eta in efficiencies:
        link = bg.link(distance_km=0, mean_photon_number=1e-4, dark_count_rate_hz=0,
                       detector_efficiency=eta, heralded=True, no_click=policy)
        rs.append(bg.play_chsh(strategy, link))
    bg.plot_sweep(efficiencies, rs, metric="S", ax=ax, label=f"no_click = {policy}",
                  xlabel="detector efficiency")
plt.show()

# %% [markdown]
# With `"zero"`, S only crosses 2 above an efficiency of 2(√2 − 1) ≈ **82.8%**.
# With `"discard"` it looks perfect at any efficiency, which is exactly why
# an experiment that discards rounds has to *assume* fair sampling.
#
# **Your turn:** find the threshold for `"random"`. Can you explain why it's higher?
