# %% [markdown]
# # Tutorial 08: Playing over a quantum network
#
# **Python skill:** building nested dicts and numpy matrices.
# **Physics idea:** a network has to *deliver* a pair before every question, and
# everything it does on the way (swapping, waiting in memory) costs fidelity.
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
# Change them with `bg.set_link` (fiber, detectors, pair quality) and
# `bg.set_node` (quantum memories, swapping hardware).

# %%
bg.parameters(net)

# %% [markdown]
# ## A game across the network
#
# A1 and B2 have no fiber between them. `bg.run_network` hands the network to
# **SeQUeNCe**, a quantum network simulator, and runs it like an experiment:
#
# 1. SeQUeNCe finds the route A1 -> HubA -> HubB -> B2, keeps making pairs on
#    every link, and **swaps** at the hubs to join them into an A1-B2 pair.
#    Pairs wait in quantum memories, and slowly decay while they wait.
# 2. A referee asks a question 1000 times per second. If A1 and B2 share a pair
#    at that moment, they measure it. If not, that question has no pair.
#
# The result's `"state"` is the average pair the players measured, so you can
# play it like any other source.

# %%
run = bg.run_network(net, "A1", "B2", sim_time_s=0.5, seed=1)
print("route:", run["path"])
print(f"{run['rounds']} of {run['questions']} questions had a pair")
print(f"asked {run['questions_hz']:.0f} questions/s, network delivered {run['pairs_hz']:.0f} pairs/s,",
      f"so {run['rounds_hz']:.0f} rounds/s were played")
print("pairs waited on average (ms):", round(run["pair_age_ms"], 2))
print("fidelity of the pairs used:", round(run["fidelity"], 3))
for link in run["links"]:
    print(f"   link {link['nodes']}: fresh-pair fidelity {link['fidelity']:.4f}, makes {link['pairs_hz']:.0f} pairs/s")
for node in run["nodes"]:
    print(f"   node {node['node']}: {node['swaps_hz']:.0f} swaps/s, {node['expired_hz']:.0f} pairs/s expired")
print(f"the players threw away {run['discarded_hz']:.0f} pairs/s (pick='newest' keeps only the latest)")
# The slowest step in the chain (a link, or a node's swaps) limits pairs_hz.

strategy = bg.optimal_strategy()
print("S, exact:", round(bg.play_chsh(strategy, run)["S"], 3))
# bg.play_history plays each recorded question on the pair it really got
measured = bg.play_history(strategy, run, seed=1)
print("S, as measured in those rounds:", round(measured["S"], 3))
# With a few hundred rounds the measured S wobbles by about +/- 0.15. Try other seeds!
# measured["history"] has every round: when it was asked, the pair's age, x, y, a, b and whether it won.
bg.plot_network(net, highlight=run["path"])
plt.show()

# %% [markdown]
# ## How good is a fresh pair? Three link models
#
# Every link has a `link_model`:
#
# * `"fixed"`: every pair has fidelity `raw_fidelity`, no matter how long the fiber
# * `"analytic"` (the default): the same, but dark counts fake some heralds, and
#   that matters more when fewer real photons survive a long fiber
# * `"fock"`: a full photon-by-photon simulation of the source (see tutorial 04),
#   including the source sometimes making two pairs at once
#
# **Your turn:** predict which model gives the highest S here, then check.

# %%
for model in ["fixed", "analytic", "fock"]:
    bg.set_all_links(net, link_model=model)
    r = bg.run_network(net, "A1", "B2", sim_time_s=0.3, seed=1)
    print(f"{model:9s} S = {bg.play_chsh(strategy, r)['S']:.3f}")
bg.set_all_links(net, link_model="analytic")

# %% [markdown]
# ## Your turn: sweeps
#
# 1. The hub-to-hub distance. Longer fiber means fewer pairs (the rate falls)
#    and pairs waiting longer in memory.
# 2. The memory coherence time: how long a memory keeps its qubit.

# %%
hub_kms = [10, 20, 40]
rates, results = [], []
for km in hub_kms:
    n = bg.two_star("HubA", ["A1"], "HubB", ["B2"], leaf_km=5, hub_km=km)
    r = bg.run_network(n, "A1", "B2", sim_time_s=0.3, seed=1)
    rates.append(r["rounds_hz"])
    results.append(bg.play_chsh(strategy, r))
bg.plot_sweep(hub_kms, results, metric="S", xlabel="hub-to-hub distance (km)")
plt.show()
print("rounds per second:", rates)

# %%
coherence_ms = [5, 20, 100, 1000]
results = []
for t in coherence_ms:
    n = bg.two_star("HubA", ["A1"], "HubB", ["B2"], leaf_km=5, hub_km=20)
    bg.set_node(n, coherence_time_ms=t)
    results.append(bg.play_chsh(strategy, bg.run_network(n, "A1", "B2", sim_time_s=0.3, seed=1)))
ax = bg.plot_sweep(coherence_ms, results, metric="S", xlabel="memory coherence time (ms)")
ax.set_xscale("log")
plt.show()

# %% [markdown]
# ## No pair, no fair?
#
# Above, questions without a pair were thrown away (`no_pair="discard"`). In a
# strict test of the game, the players must answer *every* question. With
# `no_pair="random"` they guess when they have no pair.
#
# **Your turn:** make the hub link longer until the players lose their quantum
# advantage under the strict rule, even though the pairs they do get are fine.

# %%
n = bg.two_star("HubA", ["A1"], "HubB", ["B2"], leaf_km=5, hub_km=20)
for policy in ["discard", "random"]:
    r = bg.run_network(n, "A1", "B2", sim_time_s=0.3, seed=1, no_pair=policy)
    print(f"{policy:8s} S = {bg.play_chsh(strategy, r)['S']:.3f}")

# %% [markdown]
# ## Under the hood: SeQUeNCe
#
# `bg.to_sequence(net)` hands the dict to SeQUeNCe and gives you its topology object.
# Everything SeQUeNCe knows is in there. Explore!

# %%
topo = bg.to_sequence(net)
for router in topo.get_nodes_by_type("QuantumRouter"):
    memories = router.get_components_by_type("MemoryArray")[0]
    print(router.name, "has", len(memories), "memories")
print("quantum channels:", [qc.name for qc in topo.get_qchannels()][:4], "...")
