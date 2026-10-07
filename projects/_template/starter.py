# %% [markdown]
# # My bellgame project
#
# Name:
# Question I want to answer:

# %%
import matplotlib.pyplot as plt

import bellgame as bg

# Every knob you can turn, with its unit and meaning:
for name in bg.DEFAULTS:
    print(bg.describe(name))

# %% [markdown]
# ## A first experiment
#
# Change one parameter, keep everything else fixed, and plot the result.

# %%
values = [0.001, 0.01, 0.05, 0.1]        # the values to try
results = []
for value in values:
    my_link = bg.link(mean_photon_number=value)
    results.append(bg.play_chsh(bg.optimal_strategy(), my_link))

bg.plot_sweep(values, results, metric="S", xlabel="mean photon number")
plt.show()
