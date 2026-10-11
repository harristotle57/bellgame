import numpy as np
import pytest

import bellgame as bg


def test_a_link_with_no_clicks_has_no_data():
    blind = bg.link(detector_efficiency=0, dark_count_rate_hz=0)
    result = bg.play(bg.optimal_strategy(), blind)
    assert np.isnan(result["S"]) and np.isnan(result["win_rate"]) and result["coincidence_prob"] == 0


def test_cached_arriving_state_is_read_only():
    rho = bg.arriving_state(bg.link())
    with pytest.raises(ValueError):
        rho[0, 0] = 1
    assert np.allclose(bg.arriving_state(bg.link()), rho)


@pytest.mark.parametrize("change", [
    {"heralded": "no"}, {"truncation": 1.5}, {"distance_km": -5}, {"detector_efficiency": 2},
    {"no_click": "skip"}, {"alice_misalignment_deg": [1, 2]}, {"mean_photon_number": float("nan")},
])
def test_link_rejects_bad_values_where_they_are_passed(change):
    with pytest.raises(ValueError, match=f"link: {next(iter(change))}"):
        bg.link(**change)


def test_play_rejects_a_link_edited_to_a_bad_value():
    edited = bg.link()
    edited["heralded"] = "no"  # would silently count as True
    with pytest.raises(ValueError, match="heralded"):
        bg.play(bg.optimal_strategy(), edited)


def test_a_zero_km_link_is_allowed():
    assert bg.play(bg.optimal_strategy(), bg.link(distance_km=0))["S"] > 2.8


def test_coincidence_does_not_depend_on_the_angles():
    link = bg.link(mean_photon_number=0.1)
    twisted_angles = {"alice": [10, 70], "bob": [-30, 5]}
    assert bg.play(twisted_angles, link)["coincidence_prob"] == pytest.approx(
        bg.play(bg.optimal_strategy(), link)["coincidence_prob"])


def test_fock_pair_weights_refuses_a_twisted_link():
    with pytest.raises(ValueError, match="untwisted"):
        bg.fock_pair_weights(bg.link(bob_misalignment_deg=[0, 30, 0]))
