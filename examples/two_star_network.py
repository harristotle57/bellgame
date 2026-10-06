# %% [markdown]
# # Two-star network
#
# Two hubs, each with three leaf nodes; the hubs are joined by one long link.
# A1 and B2 play the CHSH game through two swaps (at HubA and HubB).

# %%
import matplotlib.pyplot as plt
import numpy as np

import bellgame as bg

net = bg.two_star(hub_a="HubA", leaves_a=["A1", "A2", "A3"],
                  hub_b="HubB", leaves_b=["B1", "B2", "B3"],
                  leaf_km=5, hub_km=20)            # leaves only connect to their own hub

# The same network from a distance matrix:
names = ["HubA", "A1", "A2", "A3", "HubB", "B1", "B2", "B3"]
dist = np.zeros((8, 8))
dist[0, 1:4] = dist[1:4, 0] = 5                    # star A
dist[4, 5:8] = dist[5:8, 4] = 5                    # star B
dist[0, 4] = dist[4, 0] = 20                       # hub to hub
assert bg.from_matrix(names, dist) == net

# ...or with SeQUeNCe's own graph builders (node names become "0", "1", ...):
from sequence.utils.graphs import build_star  # noqa: E402

one_star = bg.from_networkx(build_star(3, length=5))
print("SeQUeNCe star:", bg.links(one_star))

# %%
strategy = bg.optimal_strategy()
run = bg.run_network(net, "A1", "B2", sim_time_s=0.5, seed=1)  # A1 -> HubA -> HubB -> B2: 3 links, 2 swaps
print("route:", run["path"], f" {run["rounds_hz"]:.0f} rounds/s,  pair fidelity {run['fidelity']:.3f}")
print("S:", round(bg.play_chsh(strategy, run)["S"], 3))
bg.plot_network(net, highlight=run["path"])

# %% [markdown]
# ## Sweep the hub-to-hub distance
#
# S and rate have different units, so they get separate charts.

# %%
hub_kms = [5, 10, 20, 40, 60]
results, rates = [], []
for km in hub_kms:
    n = bg.two_star("HubA", ["A1", "A2", "A3"], "HubB", ["B1", "B2", "B3"], leaf_km=5, hub_km=km)
    r = bg.run_network(n, "A1", "B2", sim_time_s=0.5, seed=1)
    results.append(bg.play_chsh(strategy, r))
    rates.append(r["rounds_hz"])

fig, (ax_s, ax_rate) = plt.subplots(1, 2, figsize=(12, 4))
bg.plot_sweep(hub_kms, results, metric="S", ax=ax_s, xlabel="hub-to-hub distance (km)")
ax_s.set_title("Bell violation A1 to B2", loc="left")
ax_rate.plot(hub_kms, rates, color=bg.plots.SERIES[0], linewidth=2, marker="o")
ax_rate.set_xlabel("hub-to-hub distance (km)")
ax_rate.set_ylabel("rounds with a pair per second")
ax_rate.set_title("Rate A1 to B2", loc="left")
ax_rate.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
plt.show()

# %% [markdown]
# ## Sweep the memories and the swaps
#
# Pairs wait in the hubs' memories until both links are ready to be swapped,
# and each swap uses a gate that may not be perfect.

# %%
coherence_ms = [5, 10, 20, 50, 100, 1000]
gate_fidelities = {"perfect swaps": 1.0, "99% gates": 0.99, "97% gates": 0.97}
ax = None
for label, gate in gate_fidelities.items():
    results = []
    for t in coherence_ms:
        n = bg.two_star("HubA", ["A1"], "HubB", ["B2"], leaf_km=5, hub_km=20)
        bg.set_node(n, coherence_time_ms=t, gate_fidelity=gate)
        results.append(bg.play_chsh(strategy, bg.run_network(n, "A1", "B2", sim_time_s=0.5, seed=1)))
    ax = bg.plot_sweep(coherence_ms, results, metric="S", ax=ax, label=label, xlabel="memory coherence time (ms)")
ax.set_xscale("log")
ax.set_title("S between A1 and B2", loc="left")
plt.show()
