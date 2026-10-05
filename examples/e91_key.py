# %% [markdown]
# # E91 over a photonic link
#
# Key length, QBER and S as the link gets longer.

# %%
import matplotlib.pyplot as plt

import bellgame as bg

distances = [0, 50, 100, 150, 200, 250, 300]
qbers, results = [], []
for d in distances:
    r = bg.run_e91(bg.link(distance_km=d, dark_count_rate_hz=10_000), rounds=None)
    qbers.append(r["qber"])
    results.append(r)

fig, (ax_s, ax_q) = plt.subplots(1, 2, figsize=(12, 4))
bg.plot_sweep(distances, results, metric="S", ax=ax_s, xlabel="distance (km)")
ax_s.set_title("CHSH check (key is secure while S > 2)", loc="left")
ax_q.plot(distances, [100 * q for q in qbers], color=bg.plots.SERIES[1], linewidth=2, marker="o")
ax_q.set_xlabel("distance (km)")
ax_q.set_ylabel("QBER (%)")
ax_q.set_title("Errors in the key", loc="left")
ax_q.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
plt.show()

# One sampled run at 50 km:
r = bg.run_e91(bg.link(distance_km=50), rounds=20_000, seed=1)
print(f"50 km: {r['key_length']} key bits from 20000 coincidences, QBER {r['qber']:.3%}, S {r['S']:.3f}")
