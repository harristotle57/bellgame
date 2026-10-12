import numpy as np
import pytest

import bellgame as bg


def same_rotation(u, v):
    return abs(np.trace(u.conj().T @ v)) / 2 == pytest.approx(1, abs=1e-12)


@pytest.mark.parametrize(
    "angles", [[10, 45, 30], [10, -45, 30], [-70, 45, -5], [20, 30, 10], [0, 0, 0], [10, 44.99, 30], [90, -45, 90]]
)
def test_polarization_angles_inverts_polarization_rotation(angles):
    u = bg.polarization_rotation(angles)
    assert same_rotation(u, bg.polarization_rotation(bg.polarization_angles(u)))


def test_misalign_twists_a_stack_of_states_one_by_one():
    states = np.stack([bg.bell_pair(), bg.bell_pair(0.8)])
    twisted = bg.misalign(states, bob_deg=[10, 20, 30])
    assert twisted.shape == (2, 4, 4)
    for state, one in zip(states, twisted):
        assert np.allclose(one, bg.misalign(state, bob_deg=[10, 20, 30]))


def test_misalign_with_no_twist_changes_nothing():
    rho = bg.bell_pair(0.9)
    assert np.allclose(bg.misalign(rho), rho)
    assert np.allclose(bg.misalign(rho, bob_deg=[5, 0, 0]), bg.misalign(rho, [0, 0, 0], [5, 0, 0]))


@pytest.mark.parametrize("bad", [5, "abc", [0, "x", 0], [0, 45], [0, np.nan, 0]])
def test_polarization_rotation_needs_three_angles(bad):
    with pytest.raises(ValueError, match="rotation_deg"):
        bg.polarization_rotation(bad)


@pytest.mark.parametrize("make", [bg.bell_pair, bg.werner])
def test_state_makers_reject_bad_values(make):
    for bad in ("0.9x", np.nan, 2):
        with pytest.raises(ValueError):
            make(bad)
