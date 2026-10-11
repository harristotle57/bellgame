# %% [markdown]
# # Tutorial 03: Fiber twist and alignment
#
# **New in bellgame:** `random_misalignment`, `misaligned_source`,
# `polarization_rotation`, `chsh_objective` (with `scipy.optimize.minimize`),
# `find_alignment` (any scipy method, and the `path` it took), and
# `run_e91(corrections=...)`.
# **Physics:** a long fiber rotates polarization in a way nobody knows. That
# ruins the key, but Alice can undo it with wave plates, if she can find the
# right settings using only the measured S. Eve, tapping the same fiber, has to
# solve the same problem.

# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

import bellgame as bg

twist = bg.random_misalignment(seed=4)
print("hidden twist (degrees):", {k: np.round(v, 1).tolist() for k, v in twist.items()})
twisted_pair = bg.misaligned_source(bg.bell_pair(), twist)
e = bg.run_e91(twisted_pair, rounds=None)
print(f"E91 on the twisted pair: QBER = {e['qber']:.3f}, S = {e['S']:.3f}")

# %% [markdown]
# A perfect pair, and still almost half the key bits are wrong.
#
# Alice puts wave plates in front of her polarizer, set by three correction
# angles. In bellgame these are rotations about X, then Y, then Z of the
# polarization (`bg.polarization_rotation`). On the bench, a quarter-, half-,
# quarter-wave plate stack can make any such rotation, but its plate angles are
# not these three numbers.
#
# Both photons were twisted, yet only Alice corrects. For a Bell pair that is
# enough: a rotation V on Bob's photon has the same effect as a rotation Vᵀ on
# Alice's, so Alice can undo both twists at once. This only works for a
# maximally entangled pair.
#
# Nobody knows the twist, so the only way to choose the angles is to try
# settings and watch S. That's an optimization problem.
# `bg.chsh_objective(source)` is an ordinary function `f(angles)` that returns
# **-S** (minus, because scipy minimizes), so any scipy optimizer can use it.
# A `callback` records where the optimizer is after each step.

# %%
f = bg.chsh_objective(twisted_pair)
path = [np.zeros(3)]
result = minimize(f, x0=[0, 0, 0], method="Nelder-Mead", callback=lambda angles: path.append(angles.copy()))
path = np.array(path)
s_path = np.array([-f(angles) for angles in path])
print("best angles:", result.x.round(2), " S:", round(-result.fun, 3), "after", result.nfev, "tries")

snapshots = np.linspace(0, len(path) - 1, 6).astype(int)
print(f"\n{'step':>4}  {'a':>6} {'b':>6} {'c':>6}   S")
for i in snapshots:
    print(f"{i:4d}  {path[i][0]:6.1f} {path[i][1]:6.1f} {path[i][2]:6.1f}   {s_path[i]:.3f}")

angle_styles = [("a (about X)", "#e34948", "o"), ("b (about Y)", "#2a78d6", "s"), ("c (about Z)", "#52514e", "^")]
fig, (ax_s, ax_a) = plt.subplots(2, 1, figsize=(7, 6), sharex=True, layout="constrained")
ax_s.plot(s_path, color=bg.plots.TEXT, linewidth=2)
ax_s.plot(snapshots, s_path[snapshots], "o", color=bg.plots.TEXT)
bg.reference_lines(ax_s, "S")
ax_s.set_ylabel("S")
ax_s.set_title("Exact search: Nelder-Mead, one step at a time", color=bg.plots.TEXT, fontsize=11, loc="left")
for k, (label, color, marker) in enumerate(angle_styles):
    ax_a.plot(path[:, k], color=color, linewidth=2, label=label)
    ax_a.plot(snapshots, path[snapshots, k], marker, color=color)
ax_a.set_xlabel("optimizer step")
ax_a.set_ylabel("correction angle (deg)")
ax_a.legend(frameon=False, loc="lower left")
for ax in (ax_s, ax_a):
    ax.spines[["top", "right"]].set_visible(False)
plt.show()

# %% [markdown]
# The angles wander while the optimizer feels out the landscape, then lock in.
# Once S reaches 2√2 the remaining steps only polish the last decimals.
#
# ## With shot noise
#
# In a lab every S comes from a finite number of rounds, so every value the
# optimizer sees is a little wrong. From tutorial 02, a CHSH-game run of N
# rounds near the optimum has sigma_S = 2√2/√N, about 0.057 for N = 2500.
#
# `bg.find_alignment` runs the whole search, here with 2500 rounds per try and a
# budget of 150 tries. `method` can be any `scipy.optimize.minimize` method, and
# extra keyword arguments go on to `minimize`. It returns the S it saw at each
# try (`history`) and the angles it tried (`path`), so we can also compute the
# true S at every try.

# %%
rounds = 2500
found = bg.find_alignment(twisted_pair, method="Nelder-Mead", rounds=rounds, seed=1, max_evaluations=150)
seen = np.array([s for _, s in found["history"]])
tried = np.array(found["path"])
true = np.array([bg.play_chsh(bg.with_correction(bg.optimal_strategy(), angles), twisted_pair)["S"]
                 for angles in tried])
sigma = 2 * np.sqrt(2) / np.sqrt(rounds)
best = int(np.argmax(seen))
print("correction:", [round(a, 2) for a in found["correction"]])
print(f"best S seen during the search: {found['best_seen_S']:.3f}")
print(f"true S at the angles it chose: {found['S']:.3f}")

tries = np.arange(len(seen))
fig, (ax_s, ax_a) = plt.subplots(2, 1, figsize=(7, 6), sharex=True, layout="constrained")
ax_s.fill_between(tries, bg.TSIRELSON_S - sigma, bg.TSIRELSON_S + sigma, color=bg.plots.REFERENCE, alpha=0.25,
                  linewidth=0, label=f"2√2 ± 1 sigma ({rounds:,} rounds)")
ax_s.plot(tries, seen, "o", color="#2a78d6", markersize=3, label="S measured at each try")
ax_s.plot(tries, true, color=bg.plots.TEXT, linewidth=1.5, label="true S at each try")
ax_s.plot(best, seen[best], "*", color="#e34948", markersize=14, label="best seen: the one it keeps")
bg.reference_lines(ax_s, "S")
ax_s.set_ylim(1.5, 3.2)
ax_s.set_ylabel("S")
ax_s.legend(frameon=False, loc="lower right", fontsize=8)
ax_s.set_title(f"Noisy search: {rounds:,} rounds per try", color=bg.plots.TEXT, fontsize=11, loc="left")
for k, (label, color, marker) in enumerate(angle_styles):
    ax_a.plot(tries, tried[:, k], marker, color=color, markersize=3, label=label)
ax_a.axvline(best, color="#e34948", linewidth=1, linestyle=":")
ax_a.set_xlabel("try")
ax_a.set_ylabel("correction angle (deg)")
ax_a.legend(frameon=False, loc="upper right", fontsize=8)
for ax in (ax_s, ax_a):
    ax.spines[["top", "right"]].set_visible(False)
plt.show()

# %% [markdown]
# Each measured S is fair on average: ⟨S_seen⟩ = S_true. But the search keeps
# the *biggest* one it saw, and the biggest of many noisy values is biased up.
# The largest of k independent draws sits about sigma √(2 ln k) above their mean:

# %%
near_top = true > true.max() - sigma
k = int(near_top.sum())
print(f"tries near the top: k = {k}")
print(f"average of (seen - true) there: {np.mean(seen[near_top] - true[near_top]):+.1e}  (fair on average)")
print(f"expected biggest: {true.max():.3f} + sigma √(2 ln k) = {true.max() + sigma * np.sqrt(2 * np.log(k)):.3f}")
print(f"actual biggest:   {seen.max():.3f}")

# %% [markdown]
# So "best S seen" overstates the alignment by a few sigma. Always re-measure S
# at the chosen angles before you trust it.
#
# ## The key comes back
#
# Pass the correction to E91.

# %%
corrected = {"alice_correction": found["correction"]}
after = bg.run_e91(twisted_pair, rounds=None, corrections=corrected)
print(f"before alignment: QBER = {e['qber']:.3f}, S = {e['S']:.3f}")
print(f"after alignment:  QBER = {after['qber']:.3f}, S = {after['S']:.3f}")
print(f"left over: about {np.degrees(np.arcsin(np.sqrt(after['qber']))):.1f} degrees, if it's all a tilt in the plane")

fig, (ax_before, ax_after) = plt.subplots(1, 2, figsize=(9, 3.6), layout="constrained")
bg.plot_correlations(bg.play_chsh(bg.optimal_strategy(), twisted_pair), ax=ax_before)
bg.plot_correlations(bg.play_chsh(found["strategy"], twisted_pair), ax=ax_after)
ax_before.set_title(f"Before alignment: S = {e['S']:.3f}", color=bg.plots.TEXT, fontsize=11, loc="left")
ax_after.set_title(f"After alignment: S = {after['S']:.3f}", color=bg.plots.TEXT, fontsize=11, loc="left")
plt.show()

# %% [markdown]
# The correlations are back near tutorial 01's ±0.71, but tilted: the noisy
# search left Alice a few degrees off. S barely notices, since a tilt δ costs
# a factor cos(2δ), while the QBER is sin²δ ≈ 0.2%.
#
# Alignment isn't free. This search spent 150 × 2500 = 375,000 pairs before a
# single key bit, and a real fiber's twist drifts as the temperature changes, so
# Alice has to realign every so often. Tutorial 05 counts that cost.
#
# ## Eve has to align too
#
# Eve taps Bob's fiber partway along. The fiber before her twists Bob's photon by
# a rotation nobody knows; the rest of the fiber, plus Alice's correction, undo
# it for Alice and Bob, but not for Eve. Her polarizer at 33.75 degrees (her best
# angle from tutorial 02) is pointing somewhere random in Alice and Bob's frame.
#
# She fixes it the way Alice does: three wave plates in front of her polarizer.
# She has to send her photon back out through the same plates in reverse, so the
# rest of the fiber twists it the way it would have twisted the original.
#
# What does she tune against? In E91 Alice and Bob compare the CHSH rounds and a
# sample of key bits over a public channel, so Eve can read the QBER and S as
# easily as they can. She minimizes the QBER, since that's what she hopes to hide.

# %%
I2 = np.eye(2)
eve_twist = bg.random_misalignment(seed=11)["bob"]  # the fiber between the source and Eve


def projectors(angle_deg):
    """[P0, P1] for a polarizer at ``angle_deg``: P0 = |θ⟩⟨θ| with |θ⟩ = (cos θ, sin θ)."""
    v = np.array([np.cos(np.radians(angle_deg)), np.sin(np.radians(angle_deg))])
    return [np.outer(v, v), I2 - np.outer(v, v)]


def eve_projectors(eve_correction, eve_deg=33.75):
    """Eve's polarizer projectors, as seen in Alice and Bob's frame.

    The fiber twist and her wave plates ``eve_correction`` come before her polarizer
    at ``eve_deg``. She sends the photon back out through her plates in reverse.
    """
    seen_by_eve = bg.polarization_rotation(eve_correction) @ bg.polarization_rotation(eve_twist)
    return [seen_by_eve.conj().T @ q @ seen_by_eve for q in projectors(eve_deg)]


def tapped_mid_fiber(f, eve_correction, eve_deg=33.75, pair=None):
    """The pair Alice and Bob share when Eve taps a fraction ``f`` behind the fiber twist.

    ``pair`` is what Alice and Bob would share without her (default: a perfect, aligned pair).
    """
    pair = bg.bell_pair() if pair is None else pair
    measured = sum(np.kron(I2, p) @ pair @ np.kron(I2, p) for p in eve_projectors(eve_correction, eve_deg))
    return (1 - f) * pair + f * measured


def eve_error(eve_correction, eve_deg=33.75, pair=None):
    """QBER between Alice and Eve: how often Eve's bit differs from Alice's key bit, on the key rounds Eve taps."""
    pair = bg.bell_pair() if pair is None else pair
    eve = eve_projectors(eve_correction, eve_deg)
    key_angles = [bg.E91_SETTINGS["alice"][i] for i, _ in bg.E91_SETTINGS["key_pairs"]]
    wrong = sum(np.trace(np.kron(projectors(angle)[bit], eve[1 - bit]) @ pair).real
                for angle in key_angles for bit in (0, 1))
    return wrong / len(key_angles)  # Alice uses each key angle equally often


f_eve = 0.3
eve_path = [np.zeros(3)]
eve_result = minimize(lambda c: bg.run_e91(tapped_mid_fiber(f_eve, c), rounds=None)["qber"], x0=[0, 0, 0],
                      method="Nelder-Mead", callback=lambda c: eve_path.append(c.copy()))
eve_runs = [bg.run_e91(tapped_mid_fiber(f_eve, c), rounds=None) for c in eve_path]
eve_qber = np.array([r["qber"] for r in eve_runs])
eve_s = np.array([r["S"] for r in eve_runs])
eve_wrong = np.array([eve_error(c) for c in eve_path])
for label, i in [("no wave plates", 0), ("tuned", -1)]:
    print(f"Eve, {label + ':':15s} QBER = {eve_qber[i]:.3f}, S = {eve_s[i]:.3f}, "
          f"Alice-Eve QBER on tapped rounds {eve_wrong[i]:.1%}")

fig, (ax_q, ax_e, ax_s) = plt.subplots(3, 1, figsize=(7, 8), sharex=True, layout="constrained")
ax_q.plot(100 * eve_qber, color="#e34948", linewidth=2)
ax_q.set_ylabel("QBER Alice and\nBob measure (%)")
ax_q.set_title(f"Each step is one setting of Eve's wave plates (she taps {f_eve:.0%} of pairs)",
               color=bg.plots.TEXT, fontsize=11, loc="left")
ax_e.plot(100 * eve_wrong, color="#52514e", linewidth=2, label="on the key rounds she taps")
# On the key rounds she doesn't tap, she can only guess: wrong half the time
ax_e.plot(100 * (f_eve * eve_wrong + (1 - f_eve) * 0.5), color="#52514e", linewidth=2, linestyle="--",
          label="over the whole key")
ax_e.axhline(50, color=bg.plots.REFERENCE, linestyle="--", linewidth=1.2)
ax_e.annotate("guessing\n50%", xy=(1, 50), xycoords=("axes fraction", "data"), xytext=(6, 0),
              textcoords="offset points", va="center", fontsize=8, color=bg.plots.TEXT_SECONDARY,
              annotation_clip=False)
ax_e.set_ylim(0, 55)
ax_e.set_ylabel("QBER between\nAlice and Eve (%)")
ax_e.legend(frameon=False, loc="center right")
ax_s.plot(eve_s, color="#e34948", linewidth=2, label="S Alice and Bob measure")
# A Werner pair with QBER q has S = 2√2 (1 - 2q)
ax_s.plot(2 * np.sqrt(2) * (1 - 2 * eve_qber), color="#2a78d6", linewidth=2, linestyle="--",
          label="S if that QBER were just noise")
bg.reference_lines(ax_s, "S")
ax_s.set_xlabel("Eve's optimizer step")
ax_s.set_ylabel("S Alice and\nBob measure")
ax_s.legend(frameon=False, loc="lower right")
for ax in (ax_q, ax_e, ax_s):
    ax.spines[["top", "right"]].set_visible(False)
plt.show()

# %% [markdown]
# All three panels are plotted against Eve's tuning. The top and bottom are what
# Alice and Bob measure. The middle panel is Eve's own score: the QBER between
# Alice and Eve, the fraction of key bits where Eve's bit differs from Alice's.
# On the key rounds she taps that's p; on the rest she can only guess, so over
# the whole key it is f p + (1 - f)/2.
#
# The two QBERs are linked. On a tapped round Bob gets Eve's resent photon and
# disagrees with *her* with the same chance p, so Alice and Bob disagree when
# exactly one of those mismatches happens: QBER = f 2p(1 - p). For tuned Eve,
# 0.3 × 2 × 0.038 × 0.962 = 2.2%.
#
# Before she aligns, the QBER she causes (9.8%) is so high that noise would
# explain an even *lower* S: she just looks like a bad link, too bad to make key
# from. Aligning cuts the QBER by more than four times, to the 2.2% of tutorial
# 02, low enough to pass for noise. The same alignment makes her bits better:
# wrong only 3.8% of the time, sin²(11.25°), since her polarizer sits 11.25
# degrees from each of Alice's key angles. Hiding and learning go together.
#
# S hardly moves: it tops out at 2.404 for 30% tapping, whatever she does. As
# the QBER falls, noise with that QBER would give a *higher* S, so the gap that
# gives her away gets wider, not narrower. QBER can be gamed; S can't, because
# any pair she measures is no longer entangled.
#
# ## Your experiment
#
# The cell below is yours. Some ideas:
#
# * Let Eve tune against a sampled QBER (`rounds=5000` in her objective), as she
#   would in real life. How close does she get?
# * Try other `eve_deg`, or a smaller `f_eve`. Is there any setting where S
#   stops giving her away within one run's error bar (tutorial 02)?
# * Give Alice and Bob a noisy pair (`pair=bg.bell_pair(0.97)`) and let Eve hide
#   in the noise.
# * Let Eve maximize S instead of minimizing the QBER. Can she push it above 2.404?

# %%
my_f = 0.3                          # fraction of pairs Eve taps
my_angle = 33.75                    # Eve's polarizer, in her own frame
my_correction = eve_result.x        # her wave plates: try [0, 0, 0] or your own
my_pair = None                      # None is a perfect pair; try bg.bell_pair(0.97)

mine = bg.run_e91(tapped_mid_fiber(my_f, my_correction, eve_deg=my_angle, pair=my_pair), rounds=None)
print(f"QBER = {mine['qber']:.4f}, S = {mine['S']:.3f}, "
      f"Alice-Eve QBER on tapped rounds {eve_error(my_correction, eve_deg=my_angle, pair=my_pair):.1%}")

# %% [markdown]
# ## Your turn
#
# 1. Try `method="COBYQA"` and `method="Powell"` in `find_alignment`, or any
#    other `scipy.optimize.minimize` method. Which gets the best true S within
#    150 tries? Options go straight to scipy, e.g. `options={"xtol": 1}` for
#    Powell. (`examples/optimizer_comparison.py` races them properly, over many
#    twists.)
# 2. Fewer rounds per try means a faster search but noisier S. Try `rounds=250`.
#    How far does the best seen S overshoot, and how much key is left after
#    alignment?
