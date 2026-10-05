# %% [markdown]
# # Two-star network
#
# Two hubs, each with three leaf nodes; the hubs are joined by one long link.
# A1 and B2 share entanglement through two swaps (at HubA and HubB).

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
path = bg.end_to_end(net, "A1", "B2", seed=1)    # A1 -> HubA -> HubB -> B2: 3 links, 2 swaps
print("route:", path["path"], " rate:", path["rate_hz"], "pairs/s")
print(bg.play_chsh(bg.optimal_strategy(), path)["S"])
bg.plot_network(net, highlight=path["path"])

# %% [markdown]
# ## Sweep the hub-to-hub distance
#
# S and rate have different units, so they get separate charts.

# %%
hub_kms = [5, 10, 20, 40, 60]
results, rates = [], []
for km in hub_kms:
    n = bg.two_star("HubA", ["A1", "A2", "A3"], "HubB", ["B1", "B2", "B3"], leaf_km=5, hub_km=km)
    p = bg.end_to_end(n, "A1", "B2", seed=1)
    results.append(bg.play_chsh(bg.optimal_strategy(), p))
    rates.append(p["rate_hz"])

fig, (ax_s, ax_rate) = plt.subplots(1, 2, figsize=(12, 4))
bg.plot_sweep(hub_kms, results, metric="S", ax=ax_s, xlabel="hub-to-hub distance (km)")
ax_s.set_title("Bell violation A1 to B2", loc="left")
ax_rate.plot(hub_kms, rates, color=bg.plots.SERIES[0], linewidth=2, marker="o")
ax_rate.set_xlabel("hub-to-hub distance (km)")
ax_rate.set_ylabel("end-to-end pairs per second")
ax_rate.set_title("Rate A1 to B2", loc="left")
ax_rate.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
plt.show()
