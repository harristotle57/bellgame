# %% [markdown]
# # How far can Alice and Bob be?
#
# **Part 1: over a network.** Alice and Bob are quantum routers joined by one
# fiber link. SeQUeNCe makes pairs between them, stores them in memories, and a
# referee asks 1000 questions per second. We sweep the fiber length for a few
# memory coherence times and plot S (left) and how many questions had a pair (right).

# %%
import matplotlib.pyplot as plt

import bellgame as bg

strategy = bg.optimal_strategy()
distances = [5, 10, 20, 40, 60, 80]
memories = {"10 ms memory": 10, "100 ms memory": 100, "perfect memory": float("inf")}

fig, (ax_s, ax_rate) = plt.subplots(1, 2, figsize=(12, 4))
for i, (label, coherence_ms) in enumerate(memories.items()):
    kms, results, rates = [], [], []
    for km in distances:
        net = bg.two_player_network(km)
        bg.set_node(net, coherence_time_ms=coherence_ms)
        run = bg.run_network(net, "Alice", "Bob", sim_time_s=1.0, seed=1, pick="oldest")
        rates.append(run["rounds_hz"])
        if run["rounds"] >= 20:  # too few rounds: no meaningful S
            kms.append(km)
            results.append(bg.play_chsh(strategy, run))
    bg.plot_sweep(kms, results, metric="S", ax=ax_s, label=label, xlabel="distance (km)")
    ax_rate.semilogy(distances, rates, color=bg.plots.SERIES[i], linewidth=2, marker="o", label=label)
ax_rate.set_xlabel("distance (km)")
ax_rate.set_ylabel("rounds with a pair per second")
ax_rate.spines[["top", "right"]].set_visible(False)
ax_rate.legend(frameon=False)
fig.tight_layout()
plt.show()

# %% [markdown]
# A surprise: with a short memory, S gets *better* as the fiber gets longer, at first.
# Pairs are used first in, first out (`pick="oldest"`). On a short link pairs
# are made faster than questions arrive, so they queue up in memory and decay
# while they wait. On a longer link fewer pairs arrive (every attempt needs a
# photon to survive the fiber, and a message to travel back), so each one is
# used soon after it is made. Beyond about 60 km there are hardly any rounds left.
#
# **Your turn:** run it again with `pick="newest"` (players use the freshest pair
# and throw the rest away). What happens to the 10 ms curve, and why?
#
# **Part 2: photons alone, no memories.** Here a source in the middle shoots
# photon pairs straight at Alice's and Bob's detectors (the Fock model, see tutorial
# 04). Without memories nothing waits, so we can go much further, until dark
# counts, false clicks, swamp the few real photons.

# %%
distances = list(range(0, 301, 25))
detectors = {"good (100 Hz dark)": 100, "noisy (10 kHz dark)": 10_000, "bad (100 kHz dark)": 100_000}
repetition_rate_hz = 1e8   # source pulses per second

fig, (ax_s, ax_rate) = plt.subplots(1, 2, figsize=(12, 4))
for i, (label, dark) in enumerate(detectors.items()):
    results = [bg.play_chsh(strategy, bg.link(distance_km=d, dark_count_rate_hz=dark)) for d in distances]
    bg.plot_sweep(distances, results, metric="S", ax=ax_s, label=label, xlabel="distance (km)")
    ax_rate.semilogy(distances, [r["coincidence_prob"] * repetition_rate_hz for r in results],
                     color=bg.plots.SERIES[i], linewidth=2, label=label)
ax_rate.set_xlabel("distance (km)")
ax_rate.set_ylabel("coincidences per second")
ax_rate.spines[["top", "right"]].set_visible(False)
ax_rate.legend(frameon=False)
fig.tight_layout()
plt.show()
