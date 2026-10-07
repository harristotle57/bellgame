# %% [markdown]
# # Tutorial 04: Photons as Fock modes
#
# **Python skill:** numpy arrays (shape, indexing, sums).
# **Physics idea:** a real photon source doesn't make exactly one pair. Sometimes it makes none, sometimes two.
#
# We describe light by counting photons in each **mode**. A link has four modes:
# Alice-H, Alice-V, Bob-H, Bob-V (H and V are horizontal and vertical
# polarization). Each mode can hold 0, 1, 2, ... photons; we keep up to
# `truncation` of them.

# %%
import matplotlib.pyplot as plt
import numpy as np

import bellgame as bg
from bellgame import fock

rho = fock.spdc_state(mean_photon_number=0.1, truncation=2)
print("shape:", rho.shape)          # (3 levels)^4 modes = 81
print("trace:", np.trace(rho).real) # probabilities add up to 1

# %% [markdown]
# The diagonal of `rho` holds the probability of each photon-number pattern.
# Reshape it to one axis per mode: `p[nAH, nAV, nBH, nBV]`.

# %%
p = np.real(np.diag(rho)).reshape(3, 3, 3, 3)
print("no photons at all:      ", p[0, 0, 0, 0])
print("one pair, both H:       ", p[1, 0, 1, 0])
print("one pair, both V:       ", p[0, 1, 0, 1])
print("one photon H, other V:  ", p[1, 0, 0, 1], "(never: the pair is entangled)")
print("two pairs:              ", p[2, 0, 2, 0] + p[0, 2, 0, 2] + p[1, 1, 1, 1])

# %% [markdown]
# ## A link
#
# `bg.link(...)` bundles the source with fiber and detectors. Print one to see
# every knob, and `bg.DEFAULTS` to read what each one means.

# %%
my_link = bg.link(mean_photon_number=0.1)
bg.describe_link(my_link)
bg.plot_photon_numbers(my_link)
plt.show()

# %% [markdown]
# ## Your turn
#
# 1. Make `mean_photon_number` smaller. What happens to the chance of two pairs?
#    And to the chance of getting any pair at all?
# 2. Play the CHSH game over the link. Why is S lower for a brighter source?

# %%
for mu in [0.001, 0.01, 0.1, 0.3]:
    r = bg.play_chsh(bg.optimal_strategy(), bg.link(mean_photon_number=mu))
    print(f"mean_photon_number {mu:<6} S = {r['S']:.3f}   coincidences per pulse = {r['coincidence_prob']:.2e}")
