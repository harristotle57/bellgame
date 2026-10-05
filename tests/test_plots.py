import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

import bellgame as bg  # noqa: E402


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_every_plot_returns_its_axes():
    exact = bg.play_chsh(bg.optimal_strategy(), bg.bell_pair(0.9))
    sampled = bg.play_chsh(bg.optimal_strategy(), bg.bell_pair(0.9), rounds=500, seed=1)
    _, ax = plt.subplots()
    assert bg.plot_results(sampled, ax=ax) is ax
    assert bg.plot_results(exact) is not None
    assert bg.plot_correlations(exact["table"]) is not None
    assert bg.plot_angles(bg.optimal_strategy()) is not None
    rs = [bg.play_chsh(bg.optimal_strategy(), bg.bell_pair(f)) for f in (0.7, 0.9)]
    ax = bg.plot_sweep([0.7, 0.9], rs, label="S")
    bg.plot_sweep([0.7, 0.9], rs, metric="S", ax=ax, label="again")
    assert len([line for line in ax.get_lines() if line.get_gid() == "sweep"]) == 2
    assert bg.plot_convergence([[(0, 2.0), (1, 2.5)], [(0, 2.1), (2, 2.7)]]) is not None
    net = bg.two_star("HubA", ["A1"], "HubB", ["B1"])
    assert bg.plot_network(net, highlight=["A1", "HubA", "HubB", "B1"]) is not None
    assert bg.plot_photon_numbers(bg.link()) is not None
