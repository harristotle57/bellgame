# %% [markdown]
# # How far can one link go?
#
# S (left) and the coincidence rate (right) vs. fiber length, for a few
# detector qualities. Dark counts are what eventually kill the violation.

# %%
import matplotlib.pyplot as plt
import numpy as np

import bellgame as bg

distances = np.arange(0, 301, 25)
detectors = {"good (100 Hz dark)": 100, "noisy (10 kHz dark)": 10_000, "bad (100 kHz dark)": 100_000}
repetition_rate_hz = 1e8   # source pulses per second

fig, (ax_s, ax_rate) = plt.subplots(1, 2, figsize=(12, 4))
for i, (label, dark) in enumerate(detectors.items()):
    results = [bg.play_chsh(bg.optimal_strategy(), bg.link(distance_km=d, dark_count_rate_hz=dark))
               for d in distances]
    bg.plot_sweep(distances, results, metric="S", ax=ax_s, label=label, xlabel="distance (km)")
    ax_rate.semilogy(distances, [r["coincidence_prob"] * repetition_rate_hz for r in results],
                     color=bg.plots.SERIES[i], linewidth=2, label=label)
ax_rate.set_xlabel("distance (km)")
ax_rate.set_ylabel("coincidences per second")
ax_rate.spines[["top", "right"]].set_visible(False)
ax_rate.legend(frameon=False)
fig.tight_layout()
plt.show()
