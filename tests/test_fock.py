import numpy as np
import pytest

import bellgame as bg
from bellgame import fock

IDEAL = dict(distance_km=0, mean_photon_number=1e-5, detector_efficiency=1.0, dark_count_rate_hz=0)


def s_of(**changes):
    return bg.play_chsh(bg.optimal_strategy(), bg.link(**changes))["S"]


def test_ideal_limit_reaches_tsirelson():
    assert s_of(**IDEAL) == pytest.approx(2 * np.sqrt(2), abs=1e-3)


def test_s_falls_with_mean_photon_number():
    values = [s_of(mean_photon_number=mu) for mu in [0.001, 0.01, 0.05, 0.1, 0.2]]
    assert all(a > b for a, b in zip(values, values[1:]))


def test_s_falls_with_distance_when_dark_counts_present():
    values = [s_of(distance_km=d, dark_count_rate_hz=1000) for d in [0, 50, 100, 150, 200]]
    assert all(a > b for a, b in zip(values, values[1:]))


def test_s_falls_with_misalignment():
    values = [s_of(bob_misalignment_deg=[0, a, 0]) for a in [0, 5, 10, 20]]
    assert all(a > b for a, b in zip(values, values[1:]))


@pytest.mark.parametrize("eta,violates", [(0.80, False), (0.82, False), (0.84, True), (0.90, True)])
def test_detection_loophole_threshold(eta, violates):
    s = s_of(**{**IDEAL, "detector_efficiency": eta}, heralded=True, no_click="zero")
    assert (s > 2) == violates


def test_truncation_converges():
    assert s_of(truncation=2) == pytest.approx(s_of(truncation=3), abs=1e-3)


def test_single_photon_rotation_matches_qubit_rotation():
    angles = [12, -30, 47]
    big = fock.polarization_unitary(angles, truncation=1)
    # one photon in (H, V): |1,0> is index 2, |0,1> is index 1 in the 2x2-level basis
    idx = [2, 1]
    assert np.allclose(big[np.ix_(idx, idx)], bg.polarization_rotation(angles))


def test_loss_lowers_coincidences():
    near = bg.play_chsh(bg.optimal_strategy(), bg.link(distance_km=0))["coincidence_prob"]
    far = bg.play_chsh(bg.optimal_strategy(), bg.link(distance_km=50))["coincidence_prob"]
    assert far == pytest.approx(near * fock.fiber_transmissivity(50, 0.2), rel=0.05)


def test_fock_pair_fidelity_drops_with_multi_pairs():
    low, _ = bg.fock_pair_weights(bg.link(mean_photon_number=0.001, distance_km=50))
    high, _ = bg.fock_pair_weights(bg.link(mean_photon_number=0.1, distance_km=50))
    assert low[0] > high[0]


def test_bad_parameter_name():
    with pytest.raises(ValueError, match="not a link parameter"):
        bg.link(distanse_km=3)
