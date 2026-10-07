# %% [markdown]
# # Eve on the line: eavesdropping on E91
#
# Alice and Bob share Bell pairs and run E91 (`bg.run_e91`): key bits from the
# rounds where their angles match, a CHSH test from the rounds where they don't.
# Eve sits on the fiber to Bob. On a fraction `f` of the pairs she measures Bob's
# photon at a polarizer angle of her choice, keeps the result, and sends Bob a
# fresh photon polarized the way she found it (*intercept-resend*).
#
# Measuring and resending the same state is a projective measurement whose result
# Eve writes down, so the pair Alice and Bob end up with is
#
#     rho_f = (1 - f) |Phi+><Phi+|  +  f * sum_k (1 x P_k) |Phi+><Phi+| (1 x P_k)
#
# where P_0, P_1 project onto Eve's angle and the angle 90 degrees from it.

# %%
import matplotlib.pyplot as plt
import numpy as np

import bellgame as bg
from bellgame.strategies import measurement_projectors

STRATEGY = bg.optimal_strategy()
PAIR = bg.bell_pair()
I2 = np.eye(2)


def projectors(angle_deg):
    """[P0, P1] for a polarizer at ``angle_deg``, in bellgame's convention."""
    return measurement_projectors({"alice": [angle_deg] * 2, "bob": [angle_deg] * 2}, "bob", 0)


def tapped(f, eve_deg):
    """The pair Alice and Bob share when Eve intercepts a fraction ``f`` at ``eve_deg``."""
    measured = sum(np.kron(I2, p) @ PAIR @ np.kron(I2, p) for p in projectors(eve_deg))
    return (1 - f) * PAIR + f * measured


def h(p):
    """Binary entropy in bits."""
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-p * np.log2(p) - (1 - p) * np.log2(1 - p))


def eve_information(f, eve_deg):
    """Eve's information about a key bit, I(A:E) in bits: from her own outcome on the rounds she tapped.

    On a tapped key round where Alice measured at ``alice_deg``, Eve's outcome equals
    Alice's bit with probability ``guess``; the untapped rounds tell her nothing.
    """
    info = []
    for alice_deg in [bg.E91_SETTINGS["alice"][i] for i, _ in bg.E91_SETTINGS["key_pairs"]]:
        guess = sum(np.real(np.trace(np.kron(pa, pe) @ PAIR))
                    for pa, pe in zip(projectors(alice_deg), projectors(eve_deg)))
        info.append(1 - h(guess))
    return f * float(np.mean(info))


# %% [markdown]
# ## 1. Which angle should Eve measure at?
#
# Eve taps 30% of the pairs and tries several angles. Watch the three columns:
# what she learns about the key, the error rate she causes in the key (QBER),
# and the CHSH number S.

# %%
f = 0.3
print(f"Eve taps {f:.0%} of the pairs")
print(" Eve's angle   I(A:E) bits/key bit   QBER     S")
for eve_deg in [0, 11.25, 22.5, 33.75, 45, 67.5]:
    rho = tapped(f, eve_deg)
    e91 = bg.run_e91(rho, rounds=None)
    print(f"   {eve_deg:6.2f}         {eve_information(f, eve_deg):.3f}            "
          f"{e91['qber']:.4f}   {e91['S']:.3f}")

# %% [markdown]
# Her angle trades knowledge for key errors: 33.75 degrees, halfway between E91's
# two key angles (22.5 and 45), learns the most and disturbs the key the least.
# But **S is the same whatever angle she picks**: 2.404 = (1 - f) 2 sqrt(2) + f sqrt(2).
# A pair she has measured is no longer entangled: both photons are now polarized
# along her angle (or both at 90 degrees to it), and for such a pair the CHSH
# angles give S = sqrt(2) whatever that angle is. The CHSH test doesn't care how
# she measured, only how many pairs she broke.

# %% [markdown]
# ## 2. Can Eve find out which basis is being used?
#
# Not from the photon. Alice picks her angle at random, on her side, and nobody
# announces the angles until every photon has been measured. Bob's photon is the
# same whatever Alice chose: averaged over Alice's outcome, its state is I/2
# for every angle, so there is nothing for Eve to detect.

# %%
for alice_deg in bg.E91_SETTINGS["alice"]:
    after_alice = sum(np.kron(p, I2) @ PAIR @ np.kron(p, I2) for p in projectors(alice_deg))
    bob_photon = np.trace(after_alice.reshape(2, 2, 2, 2), axis1=0, axis2=2)  # trace out Alice
    print(f"Alice at {alice_deg:4.1f} deg: Bob's photon =", (np.round(bob_photon.real, 6) + 0.0).tolist())

# %% [markdown]
# What *would* break it is predicting the choices. If Eve knows both players'
# settings in advance (a bad random number generator, or a leak from the
# equipment), she can measure Bob's photon at Alice's angle, learn Alice's bit,
# and send Bob a photon that makes him answer a XOR (x AND y). The pair is no
# longer entangled, yet S = 4, more than any quantum pair gives. It is the
# predictable referee from `classical_strategies.py`, part 4. Fresh, private
# randomness for the settings is an assumption of every Bell test.

# %%
table = np.zeros((2, 2, 2, 2))  # p(a, b | x, y)
for a in (0, 1):
    for x in (0, 1):
        for y in (0, 1):
            table[a, a ^ (x & y), x, y] = 0.5
print("Eve who knows the settings: S =", bg.chsh_value(table))

# %% [markdown]
# ## 3. Tapping a few pairs, and finite experiments
#
# A small `f` changes S only a little, so with few test rounds Eve hides in the
# statistical noise (left: S against `f`, with the spread of S over repeated
# experiments of each size; with 100,000 rounds the band is thinner than the
# line). That is fine: E91 doesn't need to *catch* Eve. From
# the measured S, Alice and Bob bound how much any Eve, with any attack, can know
# about the key (Acin et al., PRL 98, 230501 (2007)):
#
#     chi(S) = h( (1 + sqrt((S/2)^2 - 1)) / 2 )      bits per key bit
#
# and then hash the key down by that much (privacy amplification). What's left,
# after also paying h(QBER) to correct the errors, is
#
#     r = 1 - h(QBER) - chi(S)      secret bits per key bit.
#
# Right: chi(S) against what this Eve really knows (at her best angle), and r.
# The key survives a small tap; it just gets shorter. A real protocol uses S
# minus a few error bars, which is why it needs many test rounds.

# %%
best_deg = 33.75
fs = np.linspace(0, 0.6, 13)
test_rounds = [1_000, 10_000, 100_000]
fig, (ax_s, ax_key) = plt.subplots(1, 2, figsize=(12, 4))

exact_s = [bg.play_chsh(STRATEGY, tapped(f, best_deg))["S"] for f in fs]
for i, n in enumerate(test_rounds):
    spread = [np.std([bg.play_chsh(STRATEGY, tapped(f, best_deg), rounds=n, seed=k)["S"] for k in range(40)])
              for f in fs]
    ax_s.fill_between(fs, np.subtract(exact_s, spread), np.add(exact_s, spread),
                      color=bg.plots.SERIES[i], alpha=0.25, linewidth=0, label=f"+-1 sd, {n:,} rounds")
ax_s.plot(fs, exact_s, color=bg.plots.TEXT, linewidth=2, label="exact")
bg.reference_lines(ax_s, "S")
ax_s.set_xlabel("fraction of pairs Eve taps, f")
ax_s.set_ylabel("S")
ax_s.legend(frameon=False, loc="lower left")

chi, actual, rate = [], [], []
for f, s in zip(fs, exact_s):
    qber = bg.run_e91(tapped(f, best_deg), rounds=None)["qber"]
    chi.append(min(1.0, h((1 + np.sqrt(max((s / 2) ** 2 - 1, 0))) / 2)) if s > 2 else 1.0)
    actual.append(eve_information(f, best_deg))
    rate.append(max(0.0, 1 - h(qber) - chi[-1]))
for i, (values, label) in enumerate([(chi, "bound on any Eve, chi(S)"),
                                     (actual, "what this Eve knows, I(A:E)"),
                                     (rate, "secret key left, r")]):
    ax_key.plot(fs, values, color=bg.plots.SERIES[i], linewidth=2, marker="o", label=label)
ax_key.set_xlabel("fraction of pairs Eve taps, f")
ax_key.set_ylabel("bits per key bit")
ax_key.set_ylim(0, 1.05)
for ax in (ax_s, ax_key):
    ax.spines[["top", "right"]].set_visible(False)
ax_key.legend(frameon=False)
fig.tight_layout()
plt.show()

# %% [markdown]
# **Your turn:** the bound chi(S) is far above what this Eve knows, because it
# covers every attack, including ones that keep a quantum memory and wait for
# the announced angles. Replace `PAIR` with `bg.bell_pair(0.97)` (an honest but
# noisy network): how much key is left when Eve isn't there at all?
