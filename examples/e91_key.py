# %% [markdown]
# # E91 over a network link
#
# Alice and Bob run E91 on the pairs a SeQUeNCe network delivers to them.
# As the link gets longer we track the CHSH check (is the key safe?), the
# errors in the key (QBER), and how many key bits per second they get.

# %%
import matplotlib.pyplot as plt

import bellgame as bg

distances = [5, 20, 40, 60]
results, qbers, key_rates = [], [], []
for d in distances:
    net = bg.two_player_network(d)
    run = bg.run_network(net, "Alice", "Bob", sim_time_s=1.0, seed=1)
    exact = bg.run_e91(run, rounds=None)
    results.append(exact)
    qbers.append(exact["qber"])
    key_rates.append(run["rounds_hz"] * exact["key_fraction"])

fig, (ax_s, ax_q, ax_k) = plt.subplots(1, 3, figsize=(15, 4))
bg.plot_sweep(distances, results, metric="S", ax=ax_s, xlabel="distance (km)")
ax_s.set_title("CHSH check (key is secure while S > 2)", loc="left")
ax_q.plot(distances, [100 * q for q in qbers], color=bg.plots.SERIES[1], linewidth=2, marker="o")
ax_q.set_xlabel("distance (km)")
ax_q.set_ylabel("QBER (%)")
ax_q.set_title("Errors in the key", loc="left")
ax_k.semilogy(distances, key_rates, color=bg.plots.SERIES[2], linewidth=2, marker="o")
ax_k.set_xlabel("distance (km)")
ax_k.set_ylabel("raw key bits per second")
ax_k.set_title("Key rate", loc="left")
for ax in (ax_q, ax_k):
    ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
plt.show()

# %%
# One sampled run at 20 km: as many E91 rounds as the network delivered.
run = bg.run_network(bg.two_player_network(20), "Alice", "Bob", sim_time_s=2.0, seed=1)
r = bg.run_e91(run, rounds=run["rounds"], seed=1)
print(f"20 km: {run['rounds']} rounds, {r['key_length']} key bits, QBER {r['qber']:.2%}, S {r['S']:.3f}")
