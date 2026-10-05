# %% [markdown]
# # Nelder-Mead vs. COBYQA at undoing misalignment
#
# The original study: random polarization twists on both photons; Alice
# searches for three correction angles, seeing only a noisy S from 2500 rounds.
# Lines show the best S found so far (mean over runs), bands are +/- one std.

# %%
import matplotlib.pyplot as plt

import bellgame as bg

race = bg.compare_optimizers(bg.bell_pair(), methods=("Nelder-Mead", "COBYQA"),
                             runs=8, rounds=2500, max_evaluations=200, seed=0)
ax = None
for method, data in race.items():
    ax = bg.plot_convergence(data["histories"], ax=ax, label=method)
    mean_s = sum(data["validated_S"]) / len(data["validated_S"])
    print(f"{method:12s} validated S (exact, at the chosen angles): mean {mean_s:.3f}")
ax.set_title("Best S so far while searching for the alignment", loc="left")
plt.show()
