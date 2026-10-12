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
    exact = bg.play(bg.optimal_strategy(), bg.bell_pair(0.9))
    sampled = bg.play(bg.optimal_strategy(), bg.bell_pair(0.9), rounds=500, seed=1)
    _, ax = plt.subplots()
    assert bg.plot_results(sampled, ax=ax) is ax
    assert bg.plot_results(exact) is not None
    assert bg.plot_correlations(exact["table"]) is not None
    assert bg.plot_angles(bg.optimal_strategy()) is not None
    rs = [bg.play(bg.optimal_strategy(), bg.bell_pair(f)) for f in (0.7, 0.9)]
    ax = bg.plot_sweep([0.7, 0.9], rs, label="S")
    bg.plot_sweep([0.7, 0.9], rs, metric="S", ax=ax, label="again")
    assert len([line for line in ax.get_lines() if line.get_gid() == "sweep"]) == 2
    assert bg.plot_convergence([[(0, 2.0), (1, 2.5)], [(0, 2.1), (2, 2.7)]]) is not None
    net = bg.two_star("HubA", ["A1"], "HubB", ["B1"])
    assert bg.plot_network(net, highlight=["A1", "HubA", "HubB", "B1"]) is not None
    assert bg.plot_photon_numbers(bg.link()) is not None


def test_unasked_question_pairs_show_no_data_not_losses():
    few = bg.play(bg.optimal_strategy(), bg.bell_pair(), rounds=2, seed=1)  # two of four pairs never asked
    unasked = sum(sum(c.values()) == 0 for c in few["counts"].values())
    ax = bg.plot_results(few)
    assert [t.get_text() for t in ax.texts].count("no data") == unasked > 0
    assert "S = no data" in ax.get_title(loc="left")
    ax = bg.plot_correlations(few)
    assert [t.get_text() for t in ax.texts].count("no data") == unasked


def test_titles_show_win_rate_and_s_and_can_be_replaced():
    result = bg.play(bg.optimal_strategy(), bg.bell_pair())
    assert "Win rate 85.4%, S = 2.83" in bg.plot_results(result).get_title(loc="left")
    assert "S = 2.83, win rate 85.4%" in bg.plot_correlations(result).get_title(loc="left")
    calls = [
        lambda t: bg.plot_results(result, title=t),
        lambda t: bg.plot_correlations(result, title=t),
        lambda t: bg.plot_angles(bg.optimal_strategy(), title=t),
        lambda t: bg.plot_sweep([0, 1], [result, result], title=t),
        lambda t: bg.plot_convergence([[(0, 1.0), (1, 2.0)]], title=t),
        lambda t: bg.plot_network(bg.two_player_network(), title=t),
        lambda t: bg.plot_photon_numbers(bg.link(), title=t),
    ]
    for call in calls:
        assert call("mine").get_title(loc="left") == "mine"


def test_convergence_needs_points():
    with pytest.raises(ValueError, match="at least one history"):
        bg.plot_convergence([[], []])
