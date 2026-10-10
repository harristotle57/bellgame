# %% [markdown]
# # Tutorial 02: E91, and Eve
#
# **New in bellgame:** `run_e91` and `E91_SETTINGS`; noisy pairs with
# `bell_pair(fidelity)`, `werner` and `bell_diagonal`; Eve's attack built from
# `measurement_projectors`.
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

# %%
import matplotlib.pyplot as plt
import numpy as np

import bellgame as bg
from bellgame.strategies import measurement_projectors

print("Alice's angles:", bg.E91_SETTINGS["alice"])
print("Bob's angles:  ", bg.E91_SETTINGS["bob"])
print("key from (Alice, Bob) basis pairs:", bg.E91_SETTINGS["key_pairs"])

r = bg.run_e91(bg.bell_pair(), rounds=5000, seed=7)
print("key length:", r["key_length"], "bits from 5000 pairs")
print("Alice:", "".join(map(str, r["key_alice"][:40])))
print("Bob:  ", "".join(map(str, r["key_bob"][:40])))
print("QBER (fraction of key bits that differ):", r["qber"])
print("S:", round(r["S"], 3), "-> secure" if r["secure"] else "-> NOT secure")

# %% [markdown]
# Only 2 of the 9 basis pairs make key, and 4 make CHSH rounds, so S here comes
# from about 2200 rounds: the same wobble as in tutorial 01.
# `rounds=None` gives the exact values, plus `key_fraction` (2/9).
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
# (tutorial 04). Same fidelity, but a different QBER and S for each:

# %%
for name, weights in [("Phi- (phase flip)", [0.9, 0.1, 0, 0]),
                      ("Psi+ (bit flip)", [0.9, 0, 0.1, 0]),
                      ("Psi- (both)", [0.9, 0, 0, 0.1])]:
    e = bg.run_e91(bg.bell_diagonal(weights), rounds=None)
    print(f"{name:18s} QBER = {e['qber']:.3f}, S = {e['S']:.3f}")

# %% [markdown]
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
# `measurement_projectors` gives the projectors bellgame uses for a polarizer.

# %%
I2 = np.eye(2)


def projectors(angle_deg):
    """[P0, P1] for a polarizer at ``angle_deg``."""
    return measurement_projectors({"alice": [angle_deg] * 2, "bob": [angle_deg] * 2}, "bob", 0)


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
# Compare Eve with an honest, noisy network that gives the *same* QBER.

# %%
fs = np.linspace(0, 0.6, 13)
eve = [bg.run_e91(tapped(f, 33.75), rounds=None) for f in fs]
# A Werner pair with QBER q has fidelity 1 - 3q/2
honest = [bg.run_e91(bg.bell_pair(1 - 1.5 * e["qber"]), rounds=None) for e in eve]

fig, (ax_q, ax_s) = plt.subplots(1, 2, figsize=(12, 4))
ax_q.plot(fs, [100 * e["qber"] for e in eve], color=bg.plots.SERIES[0], linewidth=2, marker="o")
ax_q.set_xlabel("fraction of pairs Eve taps, f")
ax_q.set_ylabel("QBER (%)")
ax_q.set_title("The key errors Eve causes", loc="left")
ax_s.plot(fs, [e["S"] for e in eve], color=bg.plots.SERIES[0], linewidth=2, marker="o", label="Eve")
ax_s.plot(fs, [e["S"] for e in honest], color=bg.plots.SERIES[1], linewidth=2, marker="o",
          label="honest noise, same QBER")
bg.reference_lines(ax_s, "S")
ax_s.set_xlabel("fraction of pairs Eve taps, f")
ax_s.set_ylabel("S")
ax_s.set_title("What the CHSH test sees", loc="left")
ax_s.legend(frameon=False)
for ax in (ax_q, ax_s):
    ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
plt.show()

# %% [markdown]
# Tapping 30% of the pairs costs only a 2.2% QBER, which looks like ordinary
# noise. But noise with that QBER would give S ≈ 2.70, and Eve's pairs give 2.40.
# S is what bounds how much *any* eavesdropper can know; tutorial 05 turns it into
# a number of secret key bits.
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
