# %% [markdown]
# # Tutorial 05: How much secret key can the network make?
#
# **New in bellgame:** nothing new; this part puts the others together. Sweeps
# over `run_network` and `set_node`, with `run_e91` on every run.
# **Physics:** S > 2 proves the pairs were entangled, but it isn't enough for a
# key. From S and the QBER, Alice and Bob can work out how many of their key bits
# are actually secret, and that decides which networks can do QKD at all.
#
# ## From S to secret bits
#
# From the measured S, Alice and Bob can bound how much *any* eavesdropper, with
# any attack, knows about each key bit (Acin et al., PRL 98, 230501 (2007)):
#
#     chi(S) = h( (1 + sqrt((S/2)^2 - 1)) / 2 )      bits per key bit
#
# where h is the binary entropy. They remove that much by hashing the key
# (privacy amplification), and pay h(QBER) more to correct its errors. What's left:
#
#     r = 1 - h(QBER) - chi(S)      secret bits per key bit
#
# The secret key rate is then `rounds_hz * key_fraction * r`.

# %%
import matplotlib.pyplot as plt
import numpy as np

import bellgame as bg


def h(p):
    """Binary entropy in bits."""
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-p * np.log2(p) - (1 - p) * np.log2(1 - p))


def secret_fraction(e91):
    """Secret bits per key bit, r = 1 - h(QBER) - chi(S), from a ``run_e91`` result (0 if negative)."""
    s = min(e91["S"], bg.TSIRELSON_S)
    if s <= 2:
        return 0.0
    chi = h((1 + np.sqrt((s / 2) ** 2 - 1)) / 2)
    return max(0.0, 1 - h(e91["qber"]) - chi)


# Werner pairs: how good must a pair be?
for f in [1.0, 0.98, 0.95, 0.92, 0.9, 0.88, 0.85]:
    e = bg.run_e91(bg.bell_pair(f), rounds=None)
    print(f"F = {f:.2f}: S = {e['S']:.3f}, QBER = {e['qber']:.3f}, r = {secret_fraction(e):.3f}")

# %% [markdown]
# The bar is much higher than S > 2: with Werner noise the secret key runs out
# at F ≈ 0.89, S ≈ 2.42. Tutorial 04's 20 km two-star network gave S ≈ 2.3. So
# which networks clear the bar?

# %%


def key_rates(net, **run_options):
    """Run A1 -> B2 on ``net`` and return (raw key bits/s, secret key bits/s, E91 result)."""
    run = bg.run_network(net, "A1", "B2", sim_time_s=0.3, seed=1, **run_options)
    e = bg.run_e91(run, rounds=None)
    raw = run["rounds_hz"] * e["key_fraction"]
    return raw, raw * secret_fraction(e), e


def network(hub_km=20, coherence_time_ms=100):
    net = bg.two_star("HubA", ["A1"], "HubB", ["B2"], leaf_km=5, hub_km=hub_km)
    return bg.set_node(net, coherence_time_ms=coherence_time_ms)


def plot_rates(ax, xs, rates, xlabel):
    raw, secret = zip(*[(r[0], r[1]) for r in rates])
    ax.plot(xs, raw, color=bg.plots.SERIES[0], linewidth=2, marker="o", label="raw key")
    ax.plot(xs, secret, color=bg.plots.SERIES[2], linewidth=2, marker="o", label="secret key")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("key bits per second")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)


# %% [markdown]
# ## Sweep 1: the hub-to-hub distance
#
# A longer hub link makes fewer pairs, and the pairs that arrive have waited
# longer in memory. On short links the raw key is capped by the referee, not the
# network: 1000 rounds/s times a key fraction of 2/9 is 222 bits/s.

# %%
hub_kms = [5, 10, 20, 40]
by_distance = [key_rates(network(hub_km=km)) for km in hub_kms]
for km, (raw, secret, e) in zip(hub_kms, by_distance):
    print(f"{km:3d} km: S = {e['S']:.3f}, QBER = {e['qber']:.3f}, raw {raw:5.0f} bits/s, secret {secret:5.1f} bits/s")

# %% [markdown]
# ## Sweep 2: memory coherence time
#
# How long a memory keeps its qubit, at a 20 km hub link.

# %%
coherence_ms = [20, 100, 300, 1000]
by_coherence = [key_rates(network(coherence_time_ms=t)) for t in coherence_ms]
for t, (raw, secret, e) in zip(coherence_ms, by_coherence):
    print(f"{t:5d} ms: S = {e['S']:.3f}, QBER = {e['qber']:.3f}, raw {raw:5.0f} bits/s, secret {secret:5.1f} bits/s")

fig, (ax_d, ax_t) = plt.subplots(1, 2, figsize=(12, 4))
plot_rates(ax_d, hub_kms, by_distance, "hub-to-hub distance (km)")
plot_rates(ax_t, coherence_ms, by_coherence, "memory coherence time (ms)")
ax_t.set_xscale("log")
fig.tight_layout()
plt.show()

# %% [markdown]
# ## Sweep 3: which pair to use
#
# When several pairs are stored, the players can use the newest one and throw
# the rest away (`pick="newest"`, the default), or use them first in, first out
# (`pick="oldest"`). First in, first out wastes no pairs, which helps when the
# network can't keep up with the referee. At 10 km it can, so it only makes the
# pairs older.

# %%
for pick in ["newest", "oldest"]:
    raw, secret, e = key_rates(network(hub_km=10), pick=pick)
    print(f"{pick:7s} S = {e['S']:.3f}, QBER = {e['qber']:.3f}, raw {raw:5.0f} bits/s, secret {secret:5.1f} bits/s")

# %% [markdown]
# A network can have plenty of raw key and still no secret key. Here the memories
# set the limit: a pair that waits a few milliseconds in a 100 ms memory has
# already lost too much.
#
# Two caveats. These are exact expected values; a real run measures S from a
# finite number of rounds and must use S minus a few error bars (tutorial 01).
# And chi(S) covers every possible attack, so it is pessimistic about an Eve
# like tutorial 02's.
#
# ## Your turn
#
# 1. Find the shortest coherence time that still gives secret key at 40 km.
# 2. Is it better to improve the memories or the links? Try
#    `bg.set_all_links(net, raw_fidelity=0.995)` against a longer coherence time.
# 3. Turn one of these questions into a project: `cp -r projects/_template projects/<name>`.
