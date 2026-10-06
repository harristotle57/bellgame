# %% [markdown]
# # Nelder-Mead vs. COBYQA at undoing misalignment
#
# The original study, now on pairs delivered across the two-star network:
# random polarization twists on both players' photons; Alice searches for three
# correction angles, seeing only a noisy S from 2500 rounds. Lines show the best
# S found so far (mean over runs), bands are +/- one std.

# %%
import matplotlib.pyplot as plt

import bellgame as bg

net = bg.two_star("HubA", ["A1"], "HubB", ["B2"], leaf_km=5, hub_km=20)
run = bg.run_network(net, "A1", "B2", sim_time_s=0.5, seed=1)
print(f"pairs from the network: fidelity {run['fidelity']:.3f}, "
      f"best possible S {bg.play_chsh(bg.optimal_strategy(), run)['S']:.3f}")

race = bg.compare_optimizers(run, methods=("Nelder-Mead", "COBYQA"),
                             runs=8, rounds=2500, max_evaluations=200, seed=0)
ax = None
for method, data in race.items():
    ax = bg.plot_convergence(data["histories"], ax=ax, label=method)
    mean_s = sum(data["validated_S"]) / len(data["validated_S"])
    print(f"{method:12s} validated S (exact, at the chosen angles): mean {mean_s:.3f}")
ax.set_title("Best S so far while searching for the alignment", loc="left")
plt.show()
