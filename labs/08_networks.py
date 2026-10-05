# %% [markdown]
# # Lab 08: Quantum networks
#
# **Python skill:** building nested dicts and numpy matrices.
# **Physics idea:** entanglement swapping lets distant nodes share pairs, at a cost.
#
# A network is one dict. You can build it several ways; they all give the same dict.
# We'll build the **two-star** network: two hubs, each with three leaf nodes,
# and one long link between the hubs.

# %%
import matplotlib.pyplot as plt
import numpy as np

import bellgame as bg

# Way 1: a distance matrix (km). 0 means "no fiber".
names = ["HubA", "A1", "A2", "A3", "HubB", "B1", "B2", "B3"]
dist = np.zeros((8, 8))
dist[0, 1:4] = 5          # HubA to its leaves
dist[4, 5:8] = 5          # HubB to its leaves
dist[0, 4] = 20           # hub to hub
dist = dist + dist.T      # fibers go both ways: make it symmetric
net = bg.from_matrix(names, dist)

# Way 2: the builder
same_net = bg.two_star("HubA", ["A1", "A2", "A3"], "HubB", ["B1", "B2", "B3"], leaf_km=5, hub_km=20)
print("same network?", net == same_net)

plt.imshow(dist)
plt.colorbar(label="km")
plt.xticks(range(8), names)
plt.yticks(range(8), names)
plt.show()

# %% [markdown]
# Every knob lives in the dict. `bg.parameters(net)` prints them all.
# Change them with `bg.set_link` and `bg.set_memory`.

# %%
bg.parameters(net)
bg.set_memory(net, coherence_time_ms=50)

# %% [markdown]
# ## Entanglement across the network
#
# A1 and B2 have no fiber between them. SeQUeNCe (the network simulator inside
# bellgame) finds the route A1 -> HubA -> HubB -> B2, makes a pair on each link,
# and **swaps** at the hubs to join them.
#
# **Assumption:** each link's photons are stored in quantum memories when they
# arrive. While a qubit waits in memory it slowly loses its phase
# (`coherence_time_ms`), and that is one of the costs you'll see.

# %%
path = bg.end_to_end(net, "A1", "B2", seed=1)
print("route:", path["path"])
print("pairs per second:", path["rate_hz"])
print("time qubits waited (ms):", {k: round(v, 2) for k, v in path["wait_ms"].items()})
print("end-to-end fidelity:", round(path["fidelity"], 3))
for link in path["links"]:
    print("   link", link["nodes"], "fidelity", round(link["fidelity"], 4))

result = bg.play_chsh(bg.optimal_strategy(), path)
print("S between A1 and B2:", round(result["S"], 3))
bg.plot_network(net, highlight=path["path"])
plt.show()

# %% [markdown]
# ## Your turn: a sweep over the hub-to-hub distance

# %%
hub_kms = [10, 20, 40]
rates, results = [], []
for km in hub_kms:
    n = bg.two_star("HubA", ["A1"], "HubB", ["B2"], leaf_km=5, hub_km=km)
    p = bg.end_to_end(n, "A1", "B2", seed=1)
    rates.append(p["rate_hz"])
    results.append(bg.play_chsh(bg.optimal_strategy(), p))
bg.plot_sweep(hub_kms, results, metric="S", xlabel="hub-to-hub distance (km)")
plt.show()
print("rates (pairs/s):", rates)

# %% [markdown]
# ## Under the hood: SeQUeNCe
#
# `bg.build(net)` hands the dict to SeQUeNCe and gives you its topology object.
# Everything SeQUeNCe knows is in there. Explore!

# %%
topo = bg.build(net)
for router in topo.get_nodes_by_type("QuantumRouter"):
    memories = router.get_components_by_type("MemoryArray")[0]
    print(router.name, "has", len(memories), "memories")
print("quantum channels:", [qc.name for qc in topo.get_qchannels()][:4], "...")
