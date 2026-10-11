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

stages = ([(f"link {a}-{b}", link["pairs_hz"], "#52514e") for link in run["links"] for a, b in [link["nodes"]]]
          + [(f"swaps at {node['node']}", node["swaps_hz"], "#52514e") for node in run["nodes"] if node["swaps_hz"]]
          + [("pairs delivered to A1, B2", run["pairs_hz"], "#2a78d6"),
             ("questions asked", run["questions_hz"], "#2a78d6"),
             ("rounds with a pair", run["rounds_hz"], "#2a78d6"),
             ("raw key bits", run["rounds_hz"] * e["key_fraction"], "#e34948")])
fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
labels, rates, colors = zip(*stages)
bars = ax.barh(range(len(stages)), rates, color=colors)
ax.bar_label(bars, [f"{r:.0f}/s" for r in rates], padding=3, fontsize=8, color=bg.plots.TEXT)
ax.set_yticks(range(len(stages)), labels)
ax.invert_yaxis()
ax.set_xlabel("per second")
ax.set_title("From links to key", color=bg.plots.TEXT, fontsize=11, loc="left")
ax.legend(handles=[plt.matplotlib.patches.Patch(color=c, label=name)
                   for name, c in [("network", "#52514e"), ("players", "#2a78d6"), ("key", "#e34948")]],
          frameon=False, loc="lower right")
ax.spines[["top", "right"]].set_visible(False)
plt.show()

# %% [markdown]
# A puzzle: the network delivers about 1350 pairs/s and the referee asks only
# 1000 times a second, yet about one round in five finds no pair. Supply beats
# demand *on average*, but pairs arrive at random times. Some milliseconds bring
# two or three pairs, and the players keep only the newest; others bring none,
# and that round goes without. If pairs arrived completely at random (Poisson),
# the chance of none in one gap between questions would be e^(-pairs/questions):

# %%
print(f"rounds without a pair: {1 - run['rounds'] / run['questions']:.0%} measured, "
      f"{np.exp(-run['pairs_hz'] / run['questions_hz']):.0%} if pairs arrived completely at random")

# %% [markdown]
# Close: the arrivals are nearly random.
#
# ## Older pairs are worse
#
# `run["history"]` records every round: when it was asked, whether a pair was
# there, how old it was, and its four Bell weights. A pair's fidelity is its
# first Bell weight. With perfect memories the pairs would only pay for the links
# and the two swaps: for these Werner links the visibilities v = (4F - 1)/3
# multiply, and F = (3v + 1)/4. Everything below that is memory decay.

# %%
h = run["history"]
had = np.asarray(h["had_pair"], dtype=bool)
age = np.asarray(h["age_ms"])[had]
fid = np.array([w[0] for w in h["weights"]])[had]
v = np.prod([(4 * link["fidelity"] - 1) / 3 for link in run["links"]])
no_decay = (3 * v + 1) / 4
print(f"fidelity with perfect memories: {no_decay:.3f}; measured average: {run['fidelity']:.3f}")

edges = np.array([0, 1, 1.5, 2, 3, 5, np.ceil(age.max())])  # wider bins where pairs are rare
which = np.digitize(age, edges) - 1
filled = [i for i in range(len(edges) - 1) if np.any(which == i)]
centers = [age[which == i].mean() for i in filled]
means = [fid[which == i].mean() for i in filled]

fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
ax.plot(age, fid, "o", color="#2a78d6", markersize=4, alpha=0.35, label="one round's pair")
ax.plot(centers, means, color=bg.plots.TEXT, linewidth=2, marker="s", label="average, by age")
ax.axhline(no_decay, color=bg.plots.REFERENCE, linestyle="--", linewidth=1.2)
ax.annotate(f"perfect\nmemories\n{no_decay:.3f}", xy=(1, no_decay), xycoords=("axes fraction", "data"),
            xytext=(6, 0), textcoords="offset points", va="center", fontsize=8,
            color=bg.plots.TEXT_SECONDARY, annotation_clip=False)
ax.set_xlabel("pair age when measured (ms)")
ax.set_ylabel("fidelity")
ax.set_title("Older pairs are worse", color=bg.plots.TEXT, fontsize=11, loc="left")
ax.legend(frameon=False, loc="upper right")
ax.spines[["top", "right"]].set_visible(False)
plt.show()

# %% [markdown]
# Every pair sits below the perfect-memory line, and the older a pair is when
# the players measure it, the worse it tends to be. Each dot also depends on how
# long its halves waited at the hubs before the swaps, so the trend is noisy.
#
# ## Round by round
#
# `bg.play_history` plays each round on the pair it really got, so you see the
# wobble of a real experiment. Tutorial 02's error bar applies: in the CHSH game
# sigma_S² = sum of (1 - E²)/n over the four question pairs.

# %%
strategy = bg.optimal_strategy()
measured = bg.play_history(strategy, run, seed=1)
n_xy = np.array([[sum(measured["counts"][(x, y)].values()) for y in (0, 1)] for x in (0, 1)])
sigma = np.sqrt(np.sum((1 - np.array(measured["E"]) ** 2) / n_xy))
exact_s = bg.play_chsh(strategy, run)["S"]
print(f"S, exact from the average pair: {exact_s:.3f}")
print(f"S, measured over {measured['rounds']} rounds: {measured['S']:.3f} +- {sigma:.3f} "
      f"({(measured['S'] - exact_s) / sigma:+.1f} sigma from exact)")

# %% [markdown]
# ## Rounds without a pair
#
# Above, rounds without a pair are thrown away (`no_pair="discard"`). Is that
# allowed? It depends on *when* the players learn there is no pair.
#
# Here they learn it before they measure. A pair sits in a quantum memory, and
# the network announces (heralds) it; when the question comes, either a pair is
# waiting or it isn't, and that has nothing to do with which question was asked.
# Throwing away the empty rounds is then honest. This is called an event-ready
# Bell test.
#
# With a photon sent straight to a detector, the players only learn "no photon
# arrived" after choosing their polarizer. Now throwing those rounds away is
# dangerous: a cheating source (or Eve) could arrange for photons to go missing
# more often for some polarizer settings than others, and the rounds that are
# left can show S > 2 with no entanglement at all. This is the **detection
# loophole**. To close it, the players must give an answer in *every* round,
# even if they just guess.
#
# `no_pair="random"` is that strict accounting: in a round with no pair the
# players guess. It costs S, but it is the right strategy whenever a missing
# pair only shows up after the measurement, because then Eve gains nothing by
# making pairs disappear. bellgame counts a guessed round as a pair with no
# correlation at all, so exactly
#
#     S(random) = (fraction of rounds with a pair) × S(discard)
#
# It shows the price of having to answer every round: the network must deliver
# a pair for nearly every question, not just deliver good ones. Ask faster and
# watch:

# %%
asked_hz = [250, 500, 1000, 2000, 4000]
sweep = [bg.run_network(net, "A1", "B2", questions_hz=q, sim_time_s=0.2, seed=1) for q in asked_hz]
fraction = np.array([r["rounds"] / r["questions"] for r in sweep])
s_discard = np.array([bg.play_chsh(strategy, r)["S"] for r in sweep])
s_random = fraction * s_discard
for q, frac, sd, sr in zip(asked_hz, fraction, s_discard, s_random):
    print(f"{q:5d} questions/s: {frac:4.0%} with a pair, S discard = {sd:.3f}, random = {sr:.3f}")

check = bg.run_network(net, "A1", "B2", questions_hz=1000, sim_time_s=0.2, seed=1, no_pair="random")
print(f"\nno_pair='random' at 1000/s gives S = {bg.play_chsh(strategy, check)['S']:.3f}, the same")

fig, (ax_frac, ax_s) = plt.subplots(2, 1, figsize=(7, 5.5), sharex=True, layout="constrained",
                                    height_ratios=[1, 2])
ax_frac.plot(asked_hz, 100 * fraction, color="#52514e", linewidth=2, marker="o")
ax_frac.set_ylabel("rounds with\na pair (%)")
ax_frac.set_ylim(0, 105)
ax_frac.set_title("Ask faster than the network delivers, and guessing costs S", color=bg.plots.TEXT,
                  fontsize=11, loc="left")
ax_s.plot(asked_hz, s_discard, color="#2a78d6", linewidth=2, marker="s", label='no_pair="discard"')
ax_s.plot(asked_hz, s_random, color="#e34948", linewidth=2, marker="o", label='no_pair="random"')
bg.reference_lines(ax_s, "S")
ax_s.set_xscale("log")
ax_s.set_xticks(asked_hz, [str(q) for q in asked_hz])
ax_s.minorticks_off()
ax_s.set_xlabel("questions per second")
ax_s.set_ylabel("S")
ax_s.legend(frameon=False, loc="lower left")
for ax in (ax_frac, ax_s):
    ax.spines[["top", "right"]].set_visible(False)
plt.show()

# %% [markdown]
# Discarding keeps S where the pairs put it, whatever the rate. Guessing drags S
# below 2 once the fraction with a pair falls under 2/S(discard), about 86% here:
# one round in seven without a pair is enough. That happens already at the
# default 1000 questions a second.
#
# For this memory network, with its heralded pairs, discarding is fine, and
# tutorial 05 uses it. The strict accounting matters for photonic links, where
# a missing photon only shows up after the measurement. There `bg.link`'s
# `no_click` setting plays the same role, and adds `"zero"`: always answer 0
# when nothing clicks, which loses less S than guessing. With a perfect pair it
# keeps S above 2 only if the detectors catch more than 2(√2 - 1) ≈ 83% of the
# photons.
#
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
