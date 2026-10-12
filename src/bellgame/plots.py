"""Plots. Every function takes an optional ``ax=`` and ``title=``, and returns the ``ax`` it drew on.

Charts of win rate or S show the two numbers that matter as gray reference lines:
the **classical limit** (75% / S = 2) and the **quantum (Tsirelson) limit**
(85.4% / S = 2.83).

Example:
    >>> import bellgame as bg
    >>> import matplotlib
    >>> matplotlib.use("Agg")
    >>> ax = bg.plot_correlations(bg.play(bg.optimal_strategy(), bg.bell_pair()))
"""

import itertools

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm

from .game import (
    CLASSICAL_S,
    CLASSICAL_WIN_RATE,
    TSIRELSON_S,
    TSIRELSON_WIN_RATE,
    chsh_value,
    correlations,
    referee_wins,
    win_rate,
)
from .link import photon_numbers
from .network import to_networkx

# Colors, by job. Series in this order: blue, dark red, grey, pink, yellow, green.
SERIES = ["#2a78d6", "#a8322d", "#6f6e69", "#e87ba4", "#eda100", "#4fae73"]
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
REFERENCE = "#a9a8a3"  # lighter than the grey series, so a limit line never reads as data
SURFACE = "#fcfcfb"
DIVERGING = LinearSegmentedColormap.from_list("bellgame_diverging", ["#a8322d", "#f0efec", "#2a78d6"])
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


def _show(value, fmt):
    """``value`` formatted with ``fmt``, or "no data" for NaN."""
    return "no data" if np.isnan(value) else format(value, fmt)


def _limits(metric):
    """[(value, label, line style)] of the classical and quantum limits for 'S' or 'win_rate'."""
    if metric == "S":
        return [
            (CLASSICAL_S, f"classical limit S = {CLASSICAL_S:g}", "--"),
            (TSIRELSON_S, f"quantum limit S = {TSIRELSON_S:.2f}", ":"),
        ]
    if metric == "win_rate":
        return [
            (CLASSICAL_WIN_RATE, f"classical limit {CLASSICAL_WIN_RATE:.0%}", "--"),
            (TSIRELSON_WIN_RATE, f"quantum limit {TSIRELSON_WIN_RATE:.1%}", ":"),
        ]
    return []


def reference_lines(ax, metric="S"):
    """Draw the classical and quantum limits for ``metric`` ('S' or 'win_rate') as labeled lines."""
    for value, label, style in _limits(metric):
        ax.axhline(value, color=REFERENCE, linestyle=style, linewidth=1.2, zorder=1)
        # labels sit in the right margin, so they never collide with the data
        ax.annotate(
            label.replace(" limit ", " limit\n"),
            xy=(1, value),
            xycoords=("axes fraction", "data"),
            xytext=(6, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=8,
            color=TEXT_SECONDARY,
            annotation_clip=False,
        )
    return ax


def plot_results(result, ax=None, title=None):
    """Win and loss share for each question pair (x, y), from a ``bg.play`` result.

    The bars come from the result's table: the exact one, or the measured one for a
    sampled result. A question pair with no rounds is marked "no data". The default
    title gives the win rate and S.
    """
    ax = _new_ax(ax)
    table = result["table"]
    pairs = list(itertools.product((0, 1), repeat=2))
    wins = np.array([sum(table[a, b, x, y] for a, b in pairs if referee_wins(x, y, a, b)) for x, y in pairs])
    asked = ~np.isnan(wins)
    pos = np.arange(4)
    gap = 0.008  # thin surface gap between stacked segments
    ax.bar(pos[asked], wins[asked], width=0.6, color=SERIES[0], label="win", zorder=2)
    ax.bar(
        pos[asked], 1 - wins[asked] - gap, bottom=wins[asked] + gap, width=0.6, color=SERIES[1], label="loss", zorder=2
    )
    for p, w in zip(pos[asked], wins[asked]):
        ax.annotate(
            f"{w:.0%}", xy=(p, w), xytext=(0, -12), textcoords="offset points", ha="center", fontsize=8, color="white"
        )
    for p in pos[~asked]:  # never asked: no data, which is not a loss
        ax.bar(p, 1, width=0.6, fill=False, edgecolor=REFERENCE, zorder=2)
        ax.text(p, 0.5, "no data", ha="center", va="center", fontsize=8, color=TEXT_SECONDARY)
    ax.set_xticks(pos, [f"x={x}, y={y}" for x, y in pairs])
    ax.set_ylim(0, 1)
    ax.set_ylabel("share of rounds")
    if title is None:
        title = f"Win rate {_show(result['win_rate'], '.1%')}, S = {_show(result['S'], '.2f')}"
        if "rounds" in result:
            title += f" ({result['rounds']} rounds)"
    ax.set_title(title, color=TEXT, loc="left")
    reference_lines(ax, "win_rate")
    if asked.any():
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), frameon=False, ncols=2)
    _style(ax)
    return ax


def plot_correlations(result, ax=None, title=None):
    """2x2 heatmap of the correlations E(x, y) (a result dict or a probability table).

    A question pair with no data is marked "no data". The default title gives S and the win rate.
    """
    ax = _new_ax(ax, figsize=(4.2, 3.6))
    table = result["table"] if isinstance(result, dict) else result
    e = correlations(table)
    image = ax.imshow(e, cmap=DIVERGING, vmin=-1, vmax=1)
    for x, y in itertools.product((0, 1), repeat=2):
        ax.text(y, x, _show(e[x, y], "+.2f"), ha="center", va="center", color=TEXT, fontsize=11)
    ax.set_xticks([0, 1], ["y = 0", "y = 1"])
    ax.set_yticks([0, 1], ["x = 0", "x = 1"])
    if title is None:
        (_, classical, _), (_, quantum, _) = _limits("S")
        title = (
            f"Correlations E(x, y): S = {_show(chsh_value(table), '.2f')}, "
            f"win rate {_show(win_rate(table), '.1%')}\n({classical}, {quantum})"
        )
    ax.set_title(title, color=TEXT, fontsize=10, loc="left")
    plt.colorbar(image, ax=ax, label="E = p(same) - p(different)", shrink=0.85)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return ax


def plot_sweep(xs, results, metric="S", ax=None, label=None, xlabel=None, title=None):
    """Plot one number from a list of results against the values you swept.

    Call it again on the same ``ax`` with another ``label`` to compare sweeps.

    Example:
        >>> import bellgame as bg
        >>> xs = [0, 10, 20]
        >>> rs = [bg.play(bg.optimal_strategy(), bg.link(distance_km=d)) for d in xs]
        >>> ax = bg.plot_sweep(xs, rs, metric="S", xlabel="distance (km)")
    """
    ax = _new_ax(ax)
    ys = [r[metric] for r in results]
    previous = sum(1 for line in ax.get_lines() if line.get_gid() == "sweep")
    color = SERIES[previous % len(SERIES)]
    (line,) = ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=5, label=label, zorder=3)
    line.set_gid("sweep")
    if previous == 0:
        reference_lines(ax, metric)
        _style(ax, grid_axis="both")
    ax.set_ylabel(metric)
    if xlabel:
        ax.set_xlabel(xlabel)
    if label is not None:
        ax.legend(frameon=False)
    if title is not None:
        ax.set_title(title, color=TEXT, loc="left")
    return ax


def plot_angles(strategy, ax=None, title="Polarizer angles (H = 0 deg, V = 90 deg)"):
    """Draw each player's measurement directions (polarizer angles) on a half circle."""
    ax = _new_ax(ax, figsize=(4.5, 4.5))
    t = np.linspace(0, np.pi, 200)
    ax.plot(np.cos(t), np.sin(t), color=GRID, linewidth=1)
    ax.plot(np.cos(t), -np.sin(t), color=GRID, linewidth=1)
    for player, color in (("alice", SERIES[0]), ("bob", SERIES[1])):
        for bit, angle in enumerate(strategy[player]):
            a = np.radians(angle)
            ax.plot(
                [-np.cos(a), np.cos(a)],
                [-np.sin(a), np.sin(a)],
                color=color,
                linewidth=2,
                label=f"{player.title()}" if bit == 0 else None,
            )
            ax.annotate(
                f"{player[0].upper()}{bit}: {angle:g} deg",
                xy=(1.2 * np.cos(a), 1.2 * np.sin(a)),
                ha="center",
                va="center",
                fontsize=8,
                color=TEXT,
            )
    ax.set_aspect("equal")
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.axis("off")
    ax.legend(loc="lower right", frameon=False)
    ax.set_title(title, color=TEXT, fontsize=10, loc="left")
    return ax


def plot_convergence(histories, ax=None, label=None, title=None):
    """Best S found so far vs. time: mean (line) and +/- one standard deviation (band).

    ``histories`` is a list of runs; each run is a list of ``(seconds, S)`` points,
    like the ``"history"`` lists from ``bg.compare_optimizers``.
    """
    if not any(histories):
        raise ValueError("plot_convergence needs at least one history with (seconds, S) points")
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
    (line,) = ax.plot(grid, mean, color=color, linewidth=2, label=label, zorder=3)
    line.set_gid("conv")
    ax.fill_between(grid, mean - std, mean + std, color=color, alpha=0.18, linewidth=0, zorder=2)
    if first:
        reference_lines(ax, "S")
        _style(ax, grid_axis="both")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("best S so far")
    if label is not None:
        ax.legend(frameon=False, loc="center right")
    if title is not None:
        ax.set_title(title, color=TEXT, loc="left")
    return ax


def plot_network(net, ax=None, highlight=None, title=None):
    """Draw the network: nodes, fiber links labeled in km. ``highlight`` = a path to color."""
    ax = _new_ax(ax, figsize=(7, 4.5))
    g = to_networkx(net)
    pos = nx.kamada_kawai_layout(g, weight=None) if g.number_of_edges() else nx.circular_layout(g)
    path_edges = set()
    if highlight:
        path_edges = {frozenset(e) for e in zip(highlight, highlight[1:])}
    edge_colors = [SERIES[1] if frozenset(e) in path_edges else REFERENCE for e in g.edges]
    widths = [3 if frozenset(e) in path_edges else 1.5 for e in g.edges]
    nx.draw_networkx_edges(g, pos, ax=ax, edge_color=edge_colors, width=widths)
    nx.draw_networkx_nodes(g, pos, ax=ax, node_color=SERIES[0], node_size=700, edgecolors=SURFACE, linewidths=2)
    nx.draw_networkx_labels(g, pos, ax=ax, font_size=8, font_color="white")
    nx.draw_networkx_edge_labels(
        g,
        pos,
        ax=ax,
        font_size=8,
        font_color=TEXT_SECONDARY,
        edge_labels={e: f"{d['length']:g} km" for e, d in g.edges.items()},
        bbox={"boxstyle": "round", "fc": SURFACE, "ec": "none"},
    )
    if title is None:
        title = "Network" + (f" (path {' > '.join(highlight)})" if highlight else "")
    ax.set_title(title, color=TEXT, fontsize=10, loc="left")
    ax.axis("off")
    return ax


def plot_photon_numbers(link_params, ax=None, title=None):
    """Heatmap of how many photons reach Alice and Bob per pulse (shows multi-pair emission).

    ``link_params`` is a photonic link (``bg.link``) or a network link (``bg.link_params``).
    """
    ax = _new_ax(ax, figsize=(4.8, 4))
    p = photon_numbers(link_params)
    shown = np.where(p > 0, p, np.nan)
    floor = max(np.nanmin(shown), 1e-12) if np.any(p > 0) else 1e-12
    image = ax.imshow(shown, cmap=SEQUENTIAL, norm=LogNorm(vmin=floor, vmax=1), origin="lower")
    for i in range(p.shape[0]):
        for j in range(p.shape[1]):
            if p[i, j] > 0:
                light = p[i, j] < 1e-3
                ax.text(j, i, f"{p[i, j]:.1e}", ha="center", va="center", fontsize=7, color=TEXT if light else "white")
    ax.set_xlabel("photons at Bob")
    ax.set_ylabel("photons at Alice")
    ax.set_xticks(range(p.shape[1]))
    ax.set_yticks(range(p.shape[0]))
    if title is None:
        title = f"Photon numbers per pulse (mean_photon_number = {link_params['mean_photon_number']})"
    ax.set_title(title, color=TEXT, fontsize=10, loc="left")
    plt.colorbar(image, ax=ax, label="probability (log scale)", shrink=0.85)
    return ax
