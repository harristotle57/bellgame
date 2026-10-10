# %% [markdown]
# # Tutorial 04: E91 over a quantum network
#
# **New in bellgame:** network dicts (`two_star`, `from_matrix`, `parameters`,
# `set_link`, `set_node`, `set_all_links`), `run_network` (SeQUeNCe),
# `play_history`, `plot_network`, and `to_sequence`.
# **Physics:** a network has to *deliver* a pair for every round, and everything
# it does on the way (making pairs on each link, swapping at the hubs, waiting in
# memory) costs fidelity, and so key.
#
# A network is one dict. We'll build the **two-star** network: two hubs with
# three leaves each, and one long link between the hubs. The builder and a
# distance matrix give the same dict.

# %%
import matplotlib.pyplot as plt
import numpy as np

import bellgame as bg

net = bg.two_star("HubA", ["A1", "A2", "A3"], "HubB", ["B1", "B2", "B3"], leaf_km=5, hub_km=20)

names = ["HubA", "A1", "A2", "A3", "HubB", "B1", "B2", "B3"]
dist = np.zeros((8, 8))          # km; 0 means no fiber
dist[0, 1:4] = 5
dist[4, 5:8] = 5
dist[0, 4] = 20
print("same network?", bg.from_matrix(names, dist + dist.T) == net)

# %% [markdown]
# Every knob lives in the dict. `bg.parameters(net)` prints them all, with what
# each one means; here for the smallest network, two nodes and one link. Change
# them with `bg.set_link` (fiber, detectors, pair quality) and `bg.set_node`
# (quantum memories, swapping hardware).
#
# (The link knobs `mean_photon_number`, `truncation` and `heralded`, and
# `link_model="fock"`, belong to the photon-source model, which this tutorial
# doesn't cover yet.)

# %%
bg.parameters(bg.two_player_network())

# %% [markdown]
# ## Running the network
#
# A1 and B2 have no fiber between them. `bg.run_network` hands the network to
# **SeQUeNCe**, a quantum network simulator, and runs it like an experiment:
#
# 1. SeQUeNCe routes A1 -> HubA -> HubB -> B2, keeps making pairs on every link,
#    and **swaps** at the hubs to join them into an A1-B2 pair. Pairs wait in
#    quantum memories, and decay while they wait.
# 2. A referee asks for a round 1000 times per second. If A1 and B2 share a pair
#    at that moment, they measure it; if not, that round has no pair.
#
# The result's `"state"` is the average pair the players measured, so `run_e91`
# and `play_chsh` take the run itself as a source.

# %%
run = bg.run_network(net, "A1", "B2", sim_time_s=0.5, seed=1)
print("route:", run["path"])
print(f"{run['rounds']} of {run['questions']} rounds had a pair "
      f"({run['rounds_hz']:.0f}/s of {run['questions_hz']:.0f}/s asked)")
print("pairs waited on average (ms):", round(run["pair_age_ms"], 2))
print("fidelity of the pairs used:", round(run["fidelity"], 3))

e = bg.run_e91(run, rounds=None)
print(f"E91: QBER = {e['qber']:.3f}, S = {e['S']:.3f}, "
      f"raw key {run['rounds_hz'] * e['key_fraction']:.0f} bits/s")

bg.plot_network(net, highlight=run["path"])
plt.show()

# %% [markdown]
# ## Where the rate goes
#
# The run also reports what every link and node did, from SeQUeNCe's own
# metrics. The slowest step in the chain limits `pairs_hz`.

# %%
for link in run["links"]:
    print(f"link {link['nodes']}: fresh-pair fidelity {link['fidelity']:.4f}, {link['pairs_hz']:.0f} pairs/s")
for node in run["nodes"]:
    print(f"node {node['node']}: {node['swaps_hz']:.0f} swaps/s, {node['expired_hz']:.0f} pairs/s expired")
print(f"delivered {run['pairs_hz']:.0f} pairs/s; players threw away {run['discarded_hz']:.0f}/s "
      "(pick='newest' keeps only the latest)")

# %% [markdown]
# ## Round by round
#
# `run["history"]` records every round: when it was asked, whether a pair was
# there, how old it was, and its Bell weights. `bg.play_history` plays each
# round on the pair it really got, so you see the wobble of a real experiment.

# %%
strategy = bg.optimal_strategy()
measured = bg.play_history(strategy, run, seed=1)
print("S, exact from the average pair:", round(bg.play_chsh(strategy, run)["S"], 3))
print("S, measured over those rounds: ", round(measured["S"], 3))

# %% [markdown]
# ## Rounds without a pair
#
# Above, rounds without a pair are thrown away (`no_pair="discard"`). For E91
# that only costs key rate. For a strict CHSH test, though, the players must
# answer *every* round: with `no_pair="random"` they guess when they have no
# pair, and a slow network loses its quantum advantage even if the pairs it does
# deliver are good.

# %%
for policy in ["discard", "random"]:
    r = bg.run_network(net, "A1", "B2", sim_time_s=0.3, seed=1, no_pair=policy)
    print(f"{policy:8s} S = {bg.play_chsh(strategy, r)['S']:.3f}")

# %% [markdown]
# ## Under the hood: SeQUeNCe
#
# `bg.to_sequence(net)` hands the dict to SeQUeNCe and returns its topology
# object. Everything SeQUeNCe knows is in there. To run it yourself, do so
# inside `with bg.bell_diagonal_mode():`.

# %%
topo = bg.to_sequence(net)
for router in topo.get_nodes_by_type("QuantumRouter"):
    memories = router.get_components_by_type("MemoryArray")[0]
    print(router.name, "has", len(memories), "memories")
print("quantum channels:", [qc.name for qc in topo.get_qchannels()][:4], "...")

# %% [markdown]
# ## Your turn
#
# 1. Make one leaf link noisier: `bg.set_link(net, "HubA", "A1", raw_fidelity=0.9)`.
#    How do QBER and S change? Then set `raw_errors=[0, 0, 1]` (only phase
#    flips) on every link with `bg.set_all_links`: compare with tutorial 02's
#    Bell-diagonal pairs.
# 2. Route A1 to A2 instead, through one hub only. How much better is the key?
