import numpy as np
import pytest

import bellgame as bg
from bellgame.qubits import polarization_rotation
from bellgame.strategies import measurement_setting, projectors


@pytest.mark.parametrize(
    "bad",
    [
        {"alice": 0, "bob": [0, 45]},
        {"alice": "ab", "bob": [0, 45]},
        {"alice": [0, "x"], "bob": [0, 45]},
        {"alice": [0, 45], "bob": [0, 45], "bob_correction": 5},
    ],
)
def test_check_strategy_rejects_non_angles(bad):
    with pytest.raises(ValueError, match=r"strategy\['(alice|bob_correction)'\]: (angles|rotation)_deg"):
        bg.check_strategy(bad)


@pytest.mark.parametrize(
    "bad",
    [
        {"alice": [0, 45, 90], "bob": [0, 45]},
        {"alice": [0, 45], "bob": [0, 45], "alice_correction": [0, 0]},
        {"alice": [0, 45], "bob": [0, 45], "carol": [0, 45]},
        {"alice": [0, 45]},
    ],
)
def test_check_strategy_rejects_wrong_counts_and_keys(bad):
    with pytest.raises(ValueError):
        bg.check_strategy(bad)


def test_check_strategy_accepts_numpy_angles():
    bg.check_strategy({"alice": np.array([0, 45]), "bob": (22.5, -22.5), "bob_correction": np.zeros(3)})


@pytest.mark.parametrize("setting", [([0, 45],), (0, 5), (0, [0, 0])])
def test_qubit_measure_rejects_bad_settings(setting):
    with pytest.raises(ValueError, match="angle|correction"):
        measurement_setting(*setting)
    with pytest.raises(ValueError, match="angle|correction"):
        bg.play((lambda bit, history, qubit: qubit.measure(*setting), bg.copy_bit), bg.bell_pair())


@pytest.mark.parametrize("angle", [0, 22.5, 45, -30, 90])
def test_projectors_pick_out_the_measured_polarization(angle):
    p0, p1 = projectors(angle)
    t = np.radians(angle)
    along = np.array([np.cos(t), np.sin(t)])
    across = np.array([-np.sin(t), np.cos(t)])
    assert np.allclose(p0 @ along, along) and np.allclose(p1 @ across, across)
    assert np.allclose(p0 + p1, np.eye(2)) and np.allclose(p0 @ p1, 0)


def test_projectors_apply_the_correction_first():
    correction = [10, -20, 30]
    u = polarization_rotation(correction)
    p0, p1 = projectors(45, correction)
    plain0, plain1 = projectors(45)
    assert np.allclose(p0, u.conj().T @ plain0 @ u) and np.allclose(p1, u.conj().T @ plain1 @ u)
