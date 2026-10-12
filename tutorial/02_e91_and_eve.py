# %% [markdown]
# # Tutorial 02: E91, and Eve
#
# **New in bellgame:** `run_e91` and `E91_SETTINGS`; noisy pairs with
# `bell_pair(fidelity)`, `werner` and `bell_diagonal`; Eve's attack built from
# `strategies.projectors`.
# **Physics:** the CHSH test from tutorial 01 doubles as a security check. Noise
# and an eavesdropper both cost key bits, but only the CHSH number S tells you how
# much an eavesdropper could know.
#
# In E91 (Ekert, 1991) Alice and Bob each measure every pair at one of **three**
# polarizer angles, picked at random. Afterwards they announce the angles (not
# the results):
#
# * rounds where they used the **same** angle give equal bits: the **key**
# * four of the other angle pairs are the CHSH game's question pairs: they give **S**
#
# **All angles are polarizer angles.** Ekert wrote his for spin measurements:
# Alice 0, 45, 90 and Bob 45, 90, 135 degrees. A polarizer angle is half the spin
# (Bloch-sphere) angle: the two outcomes H and V are 90 degrees apart on a
# polarizer but opposite (180 degrees) on the sphere. Halved, Ekert's choice is
# exactly `bg.E91_SETTINGS`. bellgame uses polarizer angles throughout, since
# that is what you set on the bench.

# %%
import matplotlib.pyplot as plt
import numpy as np

import bellgame as bg
from bellgame.strategies import projectors

print("Alice's angles:", bg.E91_SETTINGS["alice"])
print("Bob's angles:  ", bg.E91_SETTINGS["bob"])
print("key from (Alice, Bob) basis pairs:", bg.E91_SETTINGS["key_pairs"])

r = bg.run_e91(bg.bell_pair(), rounds=5000, seed=7)
print("key length:", r["key_length"], "bits from 5000 pairs")
print("Alice:", "".join(map(str, r["key_alice"][:40])))
print("Bob:  ", "".join(map(str, r["key_bob"][:40])))
print("QBER (fraction of key bits that differ):", r["qber"])
print("S:", round(r["S"], 3), "-> Bell violation (S > 2)" if r["S"] > 2 else "-> no Bell violation")

# %% [markdown]
# S came out *above* the quantum limit 2√2 ≈ 2.83. Nothing is broken: S is
# measured from a finite number of rounds, so it has an error bar.
#
# ## How big is the error bar on S?
#
# Each correlation E is the average of ±1 outcomes, so one round has variance
# 1 - E², and the average of n rounds has variance (1 - E²)/n. S adds four
# independent averages, one per question pair:
#
#     sigma_S² = sum over (x, y) of (1 - E_xy²) / n_xy
#
# For a perfect pair every E² is 1/2. The CHSH game spreads N rounds evenly
# over 4 question pairs, so sigma_S = 2√2/√N: that is the 0.28, 0.089, 0.028
# from tutorial 01. E91 uses only 4 of its 9 basis pairs for S (and 2 for key),
# so n_xy = N/9 and sigma_S = 3√2/√N, about 0.06 for 5000 pairs.


# %%
def s_error(r):
    """One standard deviation of a sampled S, from its correlations and round counts."""
    e = np.array(r["E"])
    n = np.array([[sum(r["chsh_counts"][(x, y)].values()) for y in (0, 1)] for x in (0, 1)])
    return np.sqrt(np.sum((1 - e**2) / n))


print(f"this run: S = {r['S']:.3f} +- {s_error(r):.3f}, {(r['S'] - 2) / s_error(r):.0f} standard deviations above 2")

sizes = [500, 5000, 50000]
colors = ["#e34948", "#2a78d6", "#52514e"]  # red, blue, dark grey, as in tutorial 01
fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
for i, (n, color) in enumerate(zip(sizes, colors)):
    runs = [bg.run_e91(bg.bell_pair(), rounds=n, seed=k) for k in range(200)]
    s_values = np.array([run["S"] for run in runs])
    errors = np.array([s_error(run) for run in runs])
    covered = np.mean(np.abs(s_values - bg.TSIRELSON_S) <= errors)
    print(
        f"{n:6,} pairs: spread of S over 200 runs = {np.std(s_values):.3f}, "
        f"predicted 3√2/√N = {3 * np.sqrt(2 / n):.3f}, error bar covers 2√2 in {covered:.0%} of runs"
    )
    shown = slice(0, 20)  # 20 runs per size keeps the plot readable
    ax.errorbar(
        i + np.linspace(-0.35, 0.35, 20),
        s_values[shown],
        yerr=errors[shown],
        fmt="o",
        color=color,
        markersize=4,
        capsize=2,
    )
bg.reference_lines(ax, "S")
ax.set_xticks(range(len(sizes)), [f"{n:,} pairs" for n in sizes])
ax.set_ylabel("measured S, +- one sigma")
ax.set_title("20 runs of E91 on a perfect pair at each size", color=bg.plots.TEXT, fontsize=11, loc="left")
ax.spines[["top", "right"]].set_visible(False)
plt.show()

# %% [markdown]
# The prediction matches. It is loosest at 500 pairs, where each question pair
# gets only about 55 rounds and those counts wobble too. About two in three
# error bars reach the true S = 2√2, as one-sigma error bars should. To claim
# S > 2 you want S - 2 to be several sigma, so the closer S is to 2, the more
# pairs you must spend on the test.
# `rounds=None` gives the exact values with no error bar, plus `key_fraction` (2/9).
#
# ## Noise
#
# `bg.bell_pair(F)` is a Werner pair: fidelity F with the Bell state, the rest
# spread evenly over the other three Bell states (`bg.werner(v)` is the same
# family, by visibility). Noise raises the QBER and lowers S together.

# %%
fidelities = [1.0, 0.95, 0.9, 0.85, 0.8, 0.75]
noisy = [bg.run_e91(bg.bell_pair(f), rounds=None) for f in fidelities]
for f, e in zip(fidelities, noisy):
    print(f"F = {f:.2f}: QBER = {e['qber']:.3f}, S = {e['S']:.3f}")

# %% [markdown]
# S drops below 2 near F = 0.78, where the QBER is about 15%.
#
# Not all noise looks alike. `bg.bell_diagonal([Phi+, Phi-, Psi+, Psi-])` puts the
# missing fidelity on one Bell error at a time, the way a network's pairs come out
# (tutorial 04). All three have the same fidelity:

# %%
for name, weights in [
    ("Phi- (phase flip)", [0.9, 0.1, 0, 0]),
    ("Psi+ (bit flip)", [0.9, 0, 0.1, 0]),
    ("Psi- (both)", [0.9, 0, 0, 0.1]),
]:
    e = bg.run_e91(bg.bell_diagonal(weights), rounds=None)
    print(f"{name:18s} QBER = {e['qber']:.3f}, S = {e['S']:.3f}")

# %% [markdown]
# A phase flip and a bit flip give the **same S** but different QBERs. S weighs
# the H/V and diagonal directions equally. The key is read at 22.5 and 45
# degrees, which lean diagonal: a phase flip scrambles the diagonal key bits, and
# a bit flip leaves them alone.
#
# ## Eve on the line
#
# Eve sits on the fiber to Bob. On a fraction `f` of the pairs she measures Bob's
# photon at an angle of her choice, writes down the result, and sends Bob a fresh
# photon polarized the way she found it (*intercept-resend*). The pair Alice and
# Bob then share is
#
#     rho_f = (1 - f) rho  +  f * sum_k (1 x P_k) rho (1 x P_k)
#
# where P_0, P_1 project onto Eve's angle and the angle 90 degrees from it.
# `bellgame.strategies.projectors(angle_deg)` gives the projectors bellgame uses for a polarizer.

# %%
I2 = np.eye(2)


def tapped(f, eve_deg, pair=None):
    """The pair Alice and Bob share when Eve intercepts a fraction ``f`` at ``eve_deg`` (default: a perfect pair)."""
    pair = bg.bell_pair() if pair is None else pair
    measured = sum(np.kron(I2, p) @ pair @ np.kron(I2, p) for p in projectors(eve_deg))
    return (1 - f) * pair + f * measured


print("Eve taps 30% of the pairs")
for eve_deg in [0, 22.5, 33.75, 45]:
    e = bg.run_e91(tapped(0.3, eve_deg), rounds=None)
    print(f"  at {eve_deg:5.2f} deg: QBER = {e['qber']:.4f}, S = {e['S']:.3f}")

# %% [markdown]
# Eve's best angle, 33.75 degrees (halfway between the two key angles), causes the
# fewest key errors. But **S is 2.404 whatever angle she picks**: a pair she has
# measured is no longer entangled, and the CHSH test counts broken pairs, not how
# they were broken.
#
# ## QBER alone can't tell Eve from noise
#
# Alice and Bob only see two numbers: the QBER and S. Suppose they measure some
# QBER. Is it Eve, or just a noisy source? For each amount of tapping, take Eve's
# QBER and build a noisy pair with no Eve (a Werner pair) that gives the same
# QBER. Then compare the S each one shows.

# %%
fs = np.linspace(0, 1, 21)
eve = [bg.run_e91(tapped(f, 33.75), rounds=None) for f in fs]
eve_qber = np.array([e["qber"] for e in eve])
# A Werner pair with QBER q has fidelity 1 - 3q/2
noise = [bg.run_e91(bg.bell_pair(1 - 1.5 * q), rounds=None) for q in eve_qber]

pairs = 5000  # one lab run, for the error bands


def expected_s_error(e, pairs):
    """sigma_S for a run of ``pairs`` pairs, from exact correlations (each question pair gets pairs/9 rounds)."""
    return np.sqrt(np.sum((1 - np.array(e["E"]) ** 2) / (pairs / 9)))


fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
for results, color, marker, label in [
    (eve, "#e34948", "o", "Eve taps a fraction f of pairs"),
    (noise, "#2a78d6", "s", "no Eve, just noise (Werner pair)"),
]:
    s = np.array([e["S"] for e in results])
    sigma = np.array([expected_s_error(e, pairs) for e in results])
    ax.fill_between(100 * eve_qber, s - sigma, s + sigma, color=color, alpha=0.2, linewidth=0)
    ax.plot(100 * eve_qber, s, color=color, linewidth=2, marker=marker, label=f"{label} (exact)")
ax.fill_between(
    [], [], [], color=bg.plots.REFERENCE, alpha=0.3, linewidth=0, label=f"band: ±1 sigma for one run of {pairs:,} pairs"
)
bg.reference_lines(ax, "S")
ax.axhline(np.sqrt(2), color=bg.plots.REFERENCE, linestyle="-.", linewidth=1.2, zorder=1)
ax.annotate(
    "Eve on every pair\nS = √2 (Ekert)",
    xy=(1, np.sqrt(2)),
    xycoords=("axes fraction", "data"),
    xytext=(6, 0),
    textcoords="offset points",
    va="center",
    fontsize=8,
    color=bg.plots.TEXT_SECONDARY,
    annotation_clip=False,
)
ax.set_xlabel("QBER Alice and Bob measure (%)")
ax.set_ylabel("S")
# Eve's QBER grows in proportion to f, so f can label the top axis
per_f = eve_qber[-1] / fs[-1]
top = ax.secondary_xaxis("top", functions=(lambda q: q / 100 / per_f, lambda f: 100 * per_f * f))
top.set_xlabel("fraction of pairs Eve taps, f (Eve's curve)")
ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 0.08))
ax.set_title("S tells Eve from noise; the QBER alone can't", color=bg.plots.TEXT, fontsize=11, loc="left")
ax.spines[["right"]].set_visible(False)
plt.show()

# %% [markdown]
# Tapping 30% of the pairs costs only a 2.2% QBER, which looks like ordinary
# noise. But noise with that QBER would give S ≈ 2.70, and Eve's pairs give 2.40.
# Eve's S falls in a straight line, S = 2√2 (1 - f/2), and lands on √2 when she
# taps every pair. That is the number Ekert found for intercept-resend in 1991.
# Even then her QBER is only 7.3%, while noise with that QBER would still give
# S ≈ 2.41.
#
# The bands show what one run of 5,000 pairs could measure. They overlap when Eve
# taps only a few percent of the pairs: then a single run can't tell her from
# noise. The QBER has an error bar too, about ±0.4 to ±0.8 percentage points at
# this size, which the plot leaves out. Change `pairs` to see how the bands
# shrink.
#
# Ekert's test was qualitative: is S still near 2√2? S > 2 alone is not enough
# for a secret key. S is what bounds how much *any* eavesdropper can know, and
# tutorial 05 turns it into a number of secret key bits.
#
# `examples/eavesdropper.py` goes further: what Eve actually learns, why she
# can't detect which angle Alice chose, and what happens if she can predict it.
#
# ## Your turn
#
# 1. Give Eve a noisy pair to tap: `tapped(f, 33.75, pair=bg.bell_pair(0.95))`.
#    Can Alice and Bob still tell her apart from the noise?
# 2. Run Eve's attack with `rounds=5000` instead of `rounds=None`. How small can
#    `f` be before S can't distinguish her from honest noise?
# 3. `bg.E91_OLD_SETTINGS` are the angles from the original project code. Run
#    E91 on a perfect pair with `settings=bg.E91_OLD_SETTINGS`. Why is the QBER
#    14.6%?
