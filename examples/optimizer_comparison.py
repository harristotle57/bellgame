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
print(
    f"pairs from the network: fidelity {run['fidelity']:.3f}, "
    f"best possible S {bg.play(bg.optimal_strategy(), run)['S']:.3f}"
)

# Each method can get its own scipy.optimize.minimize arguments. The values below are bellgame's
# defaults, written out as a starting point to tune. Nelder-Mead's other default, a 45-degree
# initial_simplex around each random start, isn't listed; maxfev comes from max_evaluations.
method_kwargs = {
    "Nelder-Mead": {"options": {"xatol": 1e-3, "fatol": 1e-6}},  # stop when steps and S changes are this small
    "COBYQA": {"options": {"initial_tr_radius": 45.0}},  # first step size, in degrees
}
race = bg.compare_optimizers(
    run,
    methods=("Nelder-Mead", "COBYQA"),
    runs=8,
    rounds=2500,
    max_evaluations=200,
    seed=0,
    method_kwargs=method_kwargs,
    bounds_deg=None,  # the default: search every angle. To limit it: [(-90, 90), (-90, 90), (0, 45)]
)
ax = None
for method, data in race.items():
    ax = bg.plot_convergence(data["histories"], ax=ax, label=method)
    mean_s = sum(data["validated_S"]) / len(data["validated_S"])
    print(f"{method:12s} validated S (exact, at the chosen angles): mean {mean_s:.3f}")
ax.set_title("Best S so far while searching for the alignment", loc="left")
plt.show()
