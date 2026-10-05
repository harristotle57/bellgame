"""Plots. Every function takes an optional ``ax=`` and returns the ``ax`` it drew on.

Charts of win rate or S show the two numbers that matter as gray reference lines:
the **classical limit** (75% / S = 2) and the **quantum (Tsirelson) limit**
(85.4% / S = 2.83).

Example:
    >>> import bellgame as bg
    >>> import matplotlib
    >>> matplotlib.use("Agg")
    >>> ax = bg.plot_correlations(bg.play_chsh(bg.optimal_strategy(), bg.bell_pair()))
"""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm

from .game import CLASSICAL_S, CLASSICAL_WIN_RATE, TSIRELSON_S, TSIRELSON_WIN_RATE
from .link import photon_numbers
from .network import KM, node_names

# Colors, by job (validated categorical order: blue, orange, aqua, ...)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
REFERENCE = "#8a8984"
SURFACE = "#fcfcfb"
DIVERGING = LinearSegmentedColormap.from_list("bellgame_diverging", ["#e34948", "#f0efec", "#2a78d6"])
SEQUENTIAL = LinearSegmentedColormap.from_list("bellgame_sequential", ["#cde2fb", "#3987e5", "#0d366b"])


def _new_ax(ax, figsize=(6, 4)):
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    return ax


def _style(ax, grid_axis="y"):
    """Quiet axes: no top/right frame, light grid behind the data."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(TEXT_SECONDARY)
    ax.tick_params(colors=TEXT_SECONDARY)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)


def reference_lines(ax, metric="S"):
    """Draw the classical and quantum limits for ``metric`` ('S' or 'win_rate') as labeled lines."""
    if metric == "S":
        lines = [(CLASSICAL_S, "classical limit S = 2"), (TSIRELSON_S, "quantum limit S = 2.83")]
    elif metric == "win_rate":
        lines = [(CLASSICAL_WIN_RATE, "classical limit 75%"), (TSIRELSON_WIN_RATE, "quantum limit 85.4%")]
    else:
        return ax
    for value, label in lines:
        style = "--" if "classical" in label else ":"
        ax.axhline(value, color=REFERENCE, linestyle=style, linewidth=1.2, zorder=1)
        # labels sit in the right margin, so they never collide with the data
        ax.annotate(label.replace(" limit ", " limit\n"), xy=(1, value),
                    xycoords=("axes fraction", "data"), xytext=(6, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=8, color=TEXT_SECONDARY, annotation_clip=False)
    return ax


def plot_results(result, ax=None):
    """Win and loss share for each question pair (x, y), from a ``play_chsh`` / ``play_classical`` result.

    Sampled results (``rounds=N``) are drawn from their counts; exact results from the table.
    """
    ax = _new_ax(ax)
    labels, wins, losses = [], [], []
    for x in (0, 1):
        for y in (0, 1):
            if "counts" in result:
                c = result["counts"][(x, y)]
                total = max(1, sum(c.values()))
                win = sum(k for ab, k in c.items() if (int(ab[0]) ^ int(ab[1])) == (x & y)) / total
            else:
                t = result["table"]
                win = sum(t[a, b, x, y] for a in (0, 1) for b in (0, 1) if (a ^ b) == (x & y))
            labels.append(f"x={x}, y={y}")
            wins.append(win)
            losses.append(1 - win)
    pos = np.arange(4)
    gap = 0.008  # thin surface gap between stacked segments
    ax.bar(pos, wins, width=0.6, color=SERIES[0], label="win", zorder=2)
    ax.bar(pos, np.array(losses) - gap, bottom=np.array(wins) + gap, width=0.6, color=SERIES[1],
           label="loss", zorder=2)
    for p, w in zip(pos, wins):
        ax.annotate(f"{w:.0%}", xy=(p, w), xytext=(0, -12), textcoords="offset points",
                    ha="center", fontsize=8, color="white")
    ax.set_xticks(pos, labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("share of rounds")
    title = f"Win rate {result['win_rate']:.1%}"
    if "rounds" in result:
        title += f" over {result['rounds']} rounds"
    ax.set_title(title, color=TEXT, loc="left")
    reference_lines(ax, "win_rate")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), frameon=False, ncols=2)
    _style(ax)
    return ax


def plot_correlations(result, ax=None):
    """2x2 heatmap of the correlations E(x, y) (a result dict or a probability table)."""
    from .game import chsh_value, correlations

    ax = _new_ax(ax, figsize=(4.2, 3.6))
    table = result["table"] if isinstance(result, dict) else result
    e = correlations(table)
    image = ax.imshow(e, cmap=DIVERGING, vmin=-1, vmax=1)
    for x in (0, 1):
        for y in (0, 1):
            ax.text(y, x, f"{e[x, y]:+.2f}", ha="center", va="center", color=TEXT, fontsize=11)
    ax.set_xticks([0, 1], ["y = 0", "y = 1"])
    ax.set_yticks([0, 1], ["x = 0", "x = 1"])
    s = chsh_value(table)
    ax.set_title(f"Correlations E(x, y)    S = {s:.3f}\n(classical limit 2, quantum limit 2.83)",
                 color=TEXT, fontsize=10, loc="left")
    plt.colorbar(image, ax=ax, label="E = p(same) - p(different)", shrink=0.85)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return ax


def plot_sweep(xs, results, metric="S", ax=None, label=None, xlabel=None):
    """Plot one number from a list of results against the values you swept.

    Call it again on the same ``ax`` with another ``label`` to compare sweeps.

    Example:
        >>> import bellgame as bg
        >>> xs = [0, 10, 20]
        >>> rs = [bg.play_chsh(bg.optimal_strategy(), bg.link(distance_km=d)) for d in xs]
        >>> ax = bg.plot_sweep(xs, rs, metric="S", xlabel="distance (km)")
    """
    ax = _new_ax(ax)
    ys = [r[metric] for r in results]
    previous = sum(1 for line in ax.get_lines() if line.get_gid() == "sweep")
    color = SERIES[previous % len(SERIES)]
    line, = ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=5, label=label,
                    zorder=3)
    line.set_gid("sweep")
    if previous == 0:
        reference_lines(ax, metric)
        _style(ax, grid_axis="both")
    ax.set_ylabel(metric)
    if xlabel:
        ax.set_xlabel(xlabel)
    if label is not None:
        ax.legend(frameon=False)
    return ax


def plot_angles(strategy, ax=None):
    """Draw each player's measurement directions (polarizer angles) on a half circle."""
    ax = _new_ax(ax, figsize=(4.5, 4.5))
    t = np.linspace(0, np.pi, 200)
    ax.plot(np.cos(t), np.sin(t), color=GRID, linewidth=1)
    ax.plot(np.cos(t), -np.sin(t), color=GRID, linewidth=1)
    for player, color, style in (("alice", SERIES[0], "-"), ("bob", SERIES[1], "--")):
        for bit, angle in enumerate(strategy[player]):
            a = np.radians(angle)
            ax.plot([-np.cos(a), np.cos(a)], [-np.sin(a), np.sin(a)], color=color, linewidth=2,
                    linestyle=style, label=f"{player.title()}" if bit == 0 else None)
            ax.annotate(f"{player[0].upper()}{bit}: {angle:g} deg", xy=(1.2 * np.cos(a), 1.2 * np.sin(a)),
                        ha="center", va="center", fontsize=8, color=TEXT)
    ax.set_aspect("equal")
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.axis("off")
    ax.legend(loc="lower right", frameon=False)
    ax.set_title("Polarizer angles (H = 0 deg, V = 90 deg)", color=TEXT, fontsize=10, loc="left")
    return ax


def plot_convergence(histories, ax=None, label=None):
    """Best S found so far vs. time: mean (line) and +/- one standard deviation (band).

    ``histories`` is a list of runs; each run is a list of ``(seconds, S)`` points,
    like the ``"history"`` lists from ``bg.compare_optimizers``.
    """
    ax = _new_ax(ax)
    t_end = max(h[-1][0] for h in histories if h)
    grid = np.linspace(0, t_end, 200)
    curves = []
    for h in histories:
        times = np.array([p[0] for p in h])
        best = np.maximum.accumulate(np.array([p[1] for p in h]))
        idx = np.searchsorted(times, grid, side="right") - 1
        curves.append(np.where(idx >= 0, best[np.clip(idx, 0, None)], np.nan))
    curves = np.array(curves)
    mean = np.nanmean(curves, axis=0)
    std = np.nanstd(curves, axis=0)
    color = SERIES[sum(1 for line in ax.get_lines() if line.get_gid() == "conv") % len(SERIES)]
    first = not any(line.get_gid() == "conv" for line in ax.get_lines())
    line, = ax.plot(grid, mean, color=color, linewidth=2, label=label, zorder=3)
    line.set_gid("conv")
    ax.fill_between(grid, mean - std, mean + std, color=color, alpha=0.18, linewidth=0, zorder=2)
    if first:
        reference_lines(ax, "S")
        _style(ax, grid_axis="both")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("best S so far")
    if label is not None:
        ax.legend(frameon=False, loc="center right")
    return ax


def plot_network(net, ax=None, highlight=None):
    """Draw the network: nodes, fiber links labeled in km. ``highlight`` = a path to color."""
    ax = _new_ax(ax, figsize=(7, 4.5))
    g = nx.Graph()
    g.add_nodes_from(node_names(net))
    for q in net["qconnections"]:
        g.add_edge(q["node1"], q["node2"], km=q["distance"] / KM)
    pos = nx.kamada_kawai_layout(g, weight=None) if g.number_of_edges() else nx.circular_layout(g)
    path_edges = set()
    if highlight:
        path_edges = {frozenset(e) for e in zip(highlight, highlight[1:])}
    edge_colors = [SERIES[1] if frozenset(e) in path_edges else REFERENCE for e in g.edges]
    widths = [3 if frozenset(e) in path_edges else 1.5 for e in g.edges]
    nx.draw_networkx_edges(g, pos, ax=ax, edge_color=edge_colors, width=widths)
    nx.draw_networkx_nodes(g, pos, ax=ax, node_color=SERIES[0], node_size=700,
                           edgecolors=SURFACE, linewidths=2)
    nx.draw_networkx_labels(g, pos, ax=ax, font_size=8, font_color="white")
    nx.draw_networkx_edge_labels(g, pos, ax=ax, font_size=8, font_color=TEXT_SECONDARY,
                                 edge_labels={e: f"{d['km']:g} km" for e, d in g.edges.items()},
                                 bbox={"boxstyle": "round", "fc": SURFACE, "ec": "none"})
    ax.set_title("Network" + (f" (path {' > '.join(highlight)})" if highlight else ""),
                 color=TEXT, fontsize=10, loc="left")
    ax.axis("off")
    return ax


def plot_photon_numbers(link_params, ax=None):
    """Heatmap of how many photons reach Alice and Bob per pulse (shows multi-pair emission)."""
    ax = _new_ax(ax, figsize=(4.8, 4))
    p = photon_numbers(link_params)
    shown = np.where(p > 0, p, np.nan)
    floor = max(np.nanmin(shown), 1e-12) if np.any(p > 0) else 1e-12
    image = ax.imshow(shown, cmap=SEQUENTIAL, norm=LogNorm(vmin=floor, vmax=1), origin="lower")
    for i in range(p.shape[0]):
        for j in range(p.shape[1]):
            if p[i, j] > 0:
                light = p[i, j] < 1e-3
                ax.text(j, i, f"{p[i, j]:.1e}", ha="center", va="center", fontsize=7,
                        color=TEXT if light else "white")
    ax.set_xlabel("photons at Bob")
    ax.set_ylabel("photons at Alice")
    ax.set_xticks(range(p.shape[1]))
    ax.set_yticks(range(p.shape[0]))
    ax.set_title(f"Photon numbers per pulse (mean_photon_number = {link_params['mean_photon_number']})",
                 color=TEXT, fontsize=10, loc="left")
    plt.colorbar(image, ax=ax, label="probability (log scale)", shrink=0.85)
    return ax
