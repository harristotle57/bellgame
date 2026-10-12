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
    """Binary entropy in bits (works on numbers and arrays)."""
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def chi(s):
    """chi(S): the most any eavesdropper can know per key bit, from S (1 bit when S <= 2)."""
    s = np.minimum(s, bg.TSIRELSON_S)
    return np.where(s > 2, h((1 + np.sqrt(np.maximum((s / 2) ** 2 - 1, 0))) / 2), 1.0)


def secret_fraction(e91):
    """Secret bits per key bit, r = 1 - h(QBER) - chi(S), from a ``run_e91`` result (0 if negative)."""
    return max(0.0, float(1 - h(e91["qber"]) - chi(e91["S"])))


# Werner pairs: how good must a pair be?
for f in [1.0, 0.95, 0.9, 0.88]:
    e = bg.run_e91(bg.bell_pair(f), rounds=None)
    print(f"F = {f:.2f}: S = {e['S']:.3f}, QBER = {e['qber']:.3f}, r = {secret_fraction(e):.3f}")

fidelities = np.linspace(0.78, 1, 89)
werner = [bg.run_e91(bg.bell_pair(f), rounds=None) for f in fidelities]
cost_ec = h(np.array([e["qber"] for e in werner]))
cost_eve = chi(np.array([e["S"] for e in werner]))
spent = cost_ec + cost_eve
threshold = np.interp(1, spent[::-1], fidelities[::-1])  # where the costs add up to exactly one bit

fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
ax.fill_between(
    fidelities, 0, cost_ec, color="#52514e", alpha=0.5, linewidth=0, label="h(QBER): spent correcting errors"
)
ax.fill_between(
    fidelities, cost_ec, spent, color="#e34948", alpha=0.5, linewidth=0, label="chi(S): what Eve could know"
)
ax.fill_between(fidelities, spent, 1, where=spent < 1, color="#2a78d6", alpha=0.6, linewidth=0, label="r: secret")
ax.axhline(1, color=bg.plots.TEXT, linewidth=1, label="the whole key bit")
ax.axvline(threshold, color=bg.plots.REFERENCE, linestyle="--", linewidth=1.2)
ax.annotate(
    f"secret key\nruns out\nF ≈ {threshold:.3f}",
    xy=(threshold, 1.6),
    xytext=(-6, 0),
    textcoords="offset points",
    ha="right",
    va="top",
    fontsize=8,
    color=bg.plots.TEXT_SECONDARY,
)
ax.set_ylim(0, 1.6)
ax.set_xlabel("pair fidelity F (Werner pairs)")
ax.set_ylabel("bits, per key bit")
ax.set_title("Where each key bit goes", color=bg.plots.TEXT, fontsize=11, loc="left")
top = ax.secondary_xaxis(
    "top", functions=(lambda f: 2 * np.sqrt(2) * (4 * f - 1) / 3, lambda s: (3 * s / (2 * np.sqrt(2)) + 1) / 4)
)
top.set_xlabel("S")
ax.legend(frameon=False, loc="upper right")
ax.spines[["right"]].set_visible(False)
plt.show()

# %% [markdown]
# Error correction costs h(QBER), which grows slowly. Eve's share chi(S) grows
# much faster as S falls, and the two together eat the whole bit at F ≈ 0.89,
# S ≈ 2.42. Below that, Alice and Bob would have to throw away more than they
# have. The bar is much higher than S > 2. Tutorial 04's 20 km two-star network gave S ≈ 2.3. So
# which networks clear the bar?

# %%


def key_rates(net, **run_options):
    """Run A1 -> B2 on ``net`` and return (raw key bits/s, secret key bits/s, E91 result, the run)."""
    run = bg.run_network(net, "A1", "B2", sim_time_s=0.2, seed=1, **run_options)
    e = bg.run_e91(run, rounds=None)
    raw = run["rounds_hz"] * e["key_fraction"]
    return raw, raw * secret_fraction(e), e, run


def network(hub_km=20, coherence_time_ms=100):
    net = bg.two_star("HubA", ["A1"], "HubB", ["B2"], leaf_km=5, hub_km=hub_km)
    return bg.set_node(net, coherence_time_ms=coherence_time_ms)


def plot_sweep(ax_rate, ax_r, xs, results, xlabel):
    """Raw and secret key rates on top, the secret fraction r underneath."""
    raw = [r[0] for r in results]
    secret = [r[1] for r in results]
    ax_rate.plot(xs, raw, color="#2a78d6", linewidth=2, marker="s", label="raw key")
    ax_rate.plot(xs, secret, color="#e34948", linewidth=2, marker="o", label="secret key")
    ax_rate.set_ylabel("key bits per second")
    ax_rate.legend(frameon=False)
    ax_r.plot(xs, [secret_fraction(r[2]) for r in results], color="#e34948", linewidth=2, marker="o")
    ax_r.set_ylabel("secret bits per key bit, r")
    ax_r.set_xlabel(xlabel)
    for ax in (ax_rate, ax_r):
        ax.spines[["top", "right"]].set_visible(False)


# %% [markdown]
# ## Sweep 1: the hub-to-hub distance
#
# A longer hub link makes fewer pairs, and the pairs that arrive have waited
# longer in memory. On short links the raw key is capped by the referee, not the
# network: 1000 rounds/s times a key fraction of 2/9 is 222 bits/s.

# %%
hub_kms = [5, 10, 20, 40]
by_distance = [key_rates(network(hub_km=km)) for km in hub_kms]
for km, (raw, secret, e, _) in zip(hub_kms, by_distance):
    print(f"{km:3d} km: S = {e['S']:.3f}, QBER = {e['qber']:.3f}, raw {raw:5.0f} bits/s, secret {secret:5.1f} bits/s")

# %% [markdown]
# ## Sweep 2: memory coherence time
#
# How long a memory keeps its qubit, at a 20 km hub link.

# %%
coherence_ms = [20, 100, 300, 1000]
by_coherence = [key_rates(network(coherence_time_ms=t)) for t in coherence_ms]
for t, (raw, secret, e, _) in zip(coherence_ms, by_coherence):
    print(f"{t:5d} ms: S = {e['S']:.3f}, QBER = {e['qber']:.3f}, raw {raw:5.0f} bits/s, secret {secret:5.1f} bits/s")

fig, axes = plt.subplots(2, 2, figsize=(11, 6), sharex="col", sharey="row", layout="constrained", height_ratios=[3, 2])
plot_sweep(axes[0, 0], axes[1, 0], hub_kms, by_distance, "hub-to-hub distance (km)")
plot_sweep(axes[0, 1], axes[1, 1], coherence_ms, by_coherence, "memory coherence time (ms)")
axes[1, 1].set_xscale("log")
axes[1, 1].set_xticks(coherence_ms, [str(t) for t in coherence_ms])
axes[1, 1].minorticks_off()
axes[0, 0].set_title("Longer hub link (100 ms memories)", color=bg.plots.TEXT, fontsize=11, loc="left")
axes[0, 1].set_title("Better memories (20 km hub link)", color=bg.plots.TEXT, fontsize=11, loc="left")
plt.show()

# %% [markdown]
# The raw key barely notices the memories (right), but r does: below about
# 300 ms, the pairs lose too much while they wait, and r hits zero. On the left,
# the raw key is capped by the referee until 10 km; past that the pairs come
# slower *and* worse, and r is gone by 20 km.
#
# ## One number for distance and memory
#
# Distance and coherence time pull in opposite directions. Are they two sides of
# one thing? With perfect memories every one of these networks delivers the same
# fidelity, set by the links and the swaps (tutorial 04's formula). Distance only
# hurts through the memories: a longer hub link makes pairs more slowly (fiber
# loss means more failed attempts, and each attempt has to hear back from the
# middle of the link), so the qubits on the short leaf links wait longer for it.
#
# That suggests one number: how many hub-link pairs fit in one coherence time,
#
#     x = coherence time × (pairs per second on the hub link)
#
# If that's right, the fidelity lost to the memories should depend on x alone.


# %%
def perfect_memory_fidelity(run):
    """Fidelity with perfect memories: the Werner links' visibilities multiply through the swaps."""
    v = np.prod([(4 * link["fidelity"] - 1) / 3 for link in run["links"]])
    return (3 * v + 1) / 4


points = []
for label, settings, results in [("km", hub_kms, by_distance), ("ms", coherence_ms, by_coherence)]:
    for setting, (_, _, e, run) in zip(settings, results):
        hub = next(link for link in run["links"] if set(link["nodes"]) == {"HubA", "HubB"})
        coherence_s = (100 if label == "km" else setting) / 1000
        hub_km = setting if label == "km" else 20
        points.append(
            {
                "name": f"{hub_km} km, {coherence_s * 1000:.0f} ms",
                "x": coherence_s * hub["pairs_hz"],
                "loss": perfect_memory_fidelity(run) - run["fidelity"],
                "secret": secret_fraction(e) > 0,
                "sweep": label,
            }
        )
points = list({p["name"]: p for p in points}.values())  # 20 km, 100 ms is in both sweeps
for p in sorted(points, key=lambda p: p["x"]):
    print(
        f"{p['name']:>15}: x = {p['x']:6.0f}, fidelity lost to memory = {p['loss']:.3f}, "
        f"secret key: {'yes' if p['secret'] else 'no'}"
    )
c = np.median([p["loss"] * p["x"] for p in points if p["x"] > 200])

fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
for sweep, color, marker, label in [
    ("km", "#2a78d6", "s", "hub distance sweep (100 ms memories)"),
    ("ms", "#e34948", "o", "coherence sweep (20 km hub link)"),
]:
    for has_key in (True, False):
        chosen = [p for p in points if p["sweep"] == sweep and p["secret"] == has_key]
        ax.plot(
            [p["x"] for p in chosen],
            [p["loss"] for p in chosen],
            marker,
            color=color,
            markersize=9,
            markerfacecolor=color if has_key else "none",
            markeredgewidth=2,
            label=f"{label}, {'secret key' if has_key else 'no secret key'}",
        )
xs = np.logspace(np.log10(15), np.log10(2000))
ax.plot(xs, c / xs, color=bg.plots.REFERENCE, linestyle="--", label=f"loss ≈ {c:.0f} / x")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("x = coherence time × hub-link pairs per second")
ax.set_ylabel("fidelity lost to the memories")
ax.set_title("Distance and memory collapse onto one curve", color=bg.plots.TEXT, fontsize=11, loc="left")
ax.legend(frameon=False, fontsize=8, loc="lower left")
ax.spines[["top", "right"]].set_visible(False)
plt.show()

# %% [markdown]
# Points from the two sweeps land on the same curve. Near x ≈ 400, 5 km with
# 100 ms memories and 20 km with 300 ms memories both lose about 0.03. At x ≈ 25,
# 40 km with 100 ms and 20 km with 20 ms lose 0.21 and 0.27: close, but at such
# small x the loss levels off below c/x and the match is looser. Once x is a few
# hundred, the loss is close to c/x, so it is the *product* that matters:
# memories twice as good buy a hub link that makes pairs half as fast.
#
# Filled markers still give secret key, hollow ones don't. The switch sits
# between x ≈ 130 and x ≈ 330: roughly, a memory must hold its qubit for a couple
# of hundred hub-link pairs. Note that this has little to do with the light
# travel time: 20 km of fiber takes 0.1 ms, a thousand times shorter than the
# memories that fail here. What costs time is waiting through the failed
# attempts, and their number grows exponentially with distance.
#
# The constant c belongs to this network (its memory count, attempt rate and
# swapping), so measure it again for yours.

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
    raw, secret, e, _ = key_rates(network(hub_km=10), pick=pick)
    print(f"{pick:7s} S = {e['S']:.3f}, QBER = {e['qber']:.3f}, raw {raw:5.0f} bits/s, secret {secret:5.1f} bits/s")

# %% [markdown]
# A network can have plenty of raw key and still no secret key. Here the memories
# set the limit: a pair that waits a few milliseconds in a 100 ms memory has
# already lost too much.
#
# ## How long must you run?
#
# Everything so far used exact expected values. A real run measures S and the
# QBER from a finite number of pairs, so it has to assume the worst plausible
# values: S three error bars lower, the QBER three error bars higher. From
# tutorial 02, with N pairs E91 gives each CHSH question pair N/9 rounds and the
# key 2N/9 bits:
#
#     sigma_S² = sum of (1 - E²) / (N/9)        sigma_QBER² = QBER (1 - QBER) / (2N/9)
#
# Take the 10 km network from sweep 1, which makes about 1000 pairs a second:

# %%
raw_10km, _, e_10km, _ = by_distance[hub_kms.index(10)]
pairs_hz = raw_10km / e_10km["key_fraction"]
seconds = np.logspace(0, np.log10(3600), 60)
n = pairs_hz * seconds
sigma_s = np.sqrt(np.sum(1 - np.array(e_10km["E"]) ** 2) / (n / 9))
sigma_q = np.sqrt(e_10km["qber"] * (1 - e_10km["qber"]) / (2 * n / 9))
r_safe = np.maximum(0, 1 - h(e_10km["qber"] + 3 * sigma_q) - chi(e_10km["S"] - 3 * sigma_s))
r_exact = secret_fraction(e_10km)
first = seconds[np.argmax(r_safe > 0)]
for t in [10, 60, 300, 3600]:
    i = np.argmin(np.abs(seconds - t))
    print(f"after {t:5d} s: S = {e_10km['S']:.3f} - 3 x {sigma_s[i]:.4f}, r = {r_safe[i]:.3f}")
print(f"secret key first appears after about {first:.0f} s; r with exact values = {r_exact:.3f}")
for t in [60, 300, 3600]:
    i = np.argmin(np.abs(seconds - t))
    print(f"a {t:4d} s block: {raw_10km * t:8,.0f} raw key bits -> {raw_10km * t * r_safe[i]:7,.0f} secret bits")

fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
ax.plot(seconds, r_safe, color="#e34948", linewidth=2, label="r, using S and QBER 3 sigma worse")
ax.axhline(r_exact, color="#2a78d6", linestyle="--", linewidth=1.5, label="r with exact values")
ax.axvline(first, color=bg.plots.REFERENCE, linestyle=":", linewidth=1.2)
ax.set_xscale("log")
ax.set_xticks([1, 10, 60, 300, 3600], ["1 s", "10 s", "1 min", "5 min", "1 h"])
ax.minorticks_off()
ax.set_xlabel("block length: how long Alice and Bob collect before extracting key")
ax.set_ylabel("secret bits per key bit\nthey can guarantee")
ax.set_title(
    "Longer blocks: tighter error bars, more provably secret key", color=bg.plots.TEXT, fontsize=11, loc="left"
)
ax.legend(frameon=False, loc="lower right")
ax.spines[["top", "right"]].set_visible(False)
plt.show()

# %% [markdown]
# The network doesn't get better the longer it runs: the true r is the dashed
# line all along. What grows is how much Alice and Bob can *prove*. They collect
# key in blocks, estimate S and the QBER from that block's own test rounds, and
# only then shrink the block's key by privacy amplification. Every bit in a block
# gets the same r.
#
# A block shorter than about half a minute can't promise any secret key: its
# error bars are too wide. Longer blocks waste less, but you wait longer before
# you have any key. The error bars shrink like 1/√N, so r creeps towards its
# exact value slowly. A full finite-key analysis has more terms, but this is the
# heart of it.
#
# One more caveat: chi(S) covers every possible attack, so it is pessimistic
# about an Eve like tutorial 02's (`examples/eavesdropper.py` compares the two).
#
# ## Your turn
#
# 1. Find the shortest coherence time that still gives secret key at 40 km.
# 2. Is it better to improve the memories or the links? Try
#    `bg.set_all_links(net, raw_fidelity=0.995)` against a longer coherence time.
# 3. Turn one of these questions into a project: `cp -r projects/_template projects/<name>`.
