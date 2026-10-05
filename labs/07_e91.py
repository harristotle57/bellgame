# %% [markdown]
# # Lab 07: E91, a secret key from entanglement
#
# **Python skill:** working with lists of bits (comparing, counting, slicing).
# **Physics idea:** the same Bell test that wins the CHSH game also proves a key is secret.
#
# In E91 (Ekert, 1991) Alice and Bob each pick one of **three** polarizer
# angles at random for every photon pair:

# %%
import bellgame as bg

print("Alice's angles:", bg.E91_SETTINGS["alice"])
print("Bob's angles:  ", bg.E91_SETTINGS["bob"])

# %% [markdown]
# * Rounds where they happened to use the **same** angle give equal bits: that's the key.
# * Some rounds with **different** angles are exactly the CHSH game: they give S.
#
# If S > 2, the photons were really entangled, and entangled photons can't
# also be correlated with an eavesdropper. If S <= 2, someone (or something)
# interfered: throw the key away.

# %%
r = bg.run_e91(bg.bell_pair(), rounds=5000, seed=7)
print("key length:", r["key_length"], "bits from 5000 pairs")
print("first 20 key bits, Alice:", r["key_alice"][:20])
print("first 20 key bits, Bob:  ", r["key_bob"][:20])
print("QBER (fraction of key bits that differ):", r["qber"])
print("S:", round(r["S"], 3), "-> secure" if r["secure"] else "-> NOT secure")

# %% [markdown]
# Only about 4/9 of the rounds are CHSH rounds, so S here comes from a few
# hundred samples and can even land a bit *above* 2.83 by chance. Try
# `rounds=50000` and watch it settle.
#
# ## Your turn
#
# 1. Count the disagreements yourself with a loop over `zip(r["key_alice"], r["key_bob"])`.
# 2. Use a noisy pair, `bg.bell_pair(0.85)`. How do QBER and S change together?
# 3. Run E91 over a photonic link, `bg.link(distance_km=50)`.

# %%
for f in [1.0, 0.95, 0.9, 0.85, 0.8]:
    exact = bg.run_e91(bg.bell_pair(f), rounds=None)
    print(f"fidelity {f:.2f}: QBER = {exact['qber']:.3f}, S = {exact['S']:.3f}")

# %% [markdown]
# ## Misalignment ruins the key, alignment saves it

# %%
twisted = bg.misaligned_source(bg.bell_pair(), bg.random_misalignment(seed=2))
before = bg.run_e91(twisted, rounds=None)
found = bg.find_alignment(twisted, seed=2)
after = bg.run_e91(twisted, rounds=None, corrections={"alice_correction": found["correction"]})
print(f"before alignment: QBER = {before['qber']:.3f}, S = {before['S']:.3f}")
print(f"after alignment:  QBER = {after['qber']:.3f}, S = {after['S']:.3f}")

# %% [markdown]
# ## A note on the angles
#
# The original project code used Alice {0, 45, 90} and Bob {22.5, 67.5, 112.5}.
# With polarizers those sets share no angle, so even a perfect pair gives a
# 14.6% QBER. Check it:

# %%
old = bg.run_e91(bg.bell_pair(), rounds=None, settings=bg.E91_OLD_SETTINGS)
print(f"old angles: QBER = {old['qber']:.3f}, S = {old['S']:.3f}")
