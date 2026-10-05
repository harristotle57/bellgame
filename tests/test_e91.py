import numpy as np
import pytest

import bellgame as bg


def test_ideal_pair_gives_perfect_key_and_tsirelson():
    exact = bg.run_e91(bg.bell_pair(), rounds=None)
    assert exact["qber"] == pytest.approx(0)
    assert exact["S"] == pytest.approx(2 * np.sqrt(2))
    r = bg.run_e91(bg.bell_pair(), rounds=5000, seed=2)
    assert r["key_alice"] == r["key_bob"]
    assert r["key_length"] == pytest.approx(5000 * 2 / 9, rel=0.1)
    assert r["S"] == pytest.approx(2 * np.sqrt(2), abs=0.15)


def test_werner_qber_and_s():
    f = 0.9
    exact = bg.run_e91(bg.bell_pair(f), rounds=None)
    v = (4 * f - 1) / 3
    assert exact["qber"] == pytest.approx((1 - v) / 2)
    assert exact["S"] == pytest.approx(2 * np.sqrt(2) * v)


def test_old_settings_have_built_in_errors():
    old = bg.run_e91(bg.bell_pair(), rounds=None, settings=bg.E91_OLD_SETTINGS)
    assert old["qber"] == pytest.approx(np.sin(np.radians(22.5)) ** 2)  # 14.6%
    assert old["S"] == pytest.approx(2 * np.sqrt(2))


def test_seeded():
    a = bg.run_e91(bg.bell_pair(0.8), rounds=1000, seed=5)
    b = bg.run_e91(bg.bell_pair(0.8), rounds=1000, seed=5)
    assert a["key_alice"] == b["key_alice"] and a["S"] == b["S"]


def test_misalignment_breaks_then_correction_fixes():
    twisted = bg.misaligned_source(bg.bell_pair(), bg.random_misalignment(1))
    assert bg.run_e91(twisted, rounds=None)["qber"] > 0.05
    found = bg.find_alignment(twisted, seed=1)
    fixed = bg.run_e91(twisted, rounds=None, corrections={"alice_correction": found["correction"]})
    assert fixed["qber"] < 1e-4 and fixed["S"] > 2.8


def test_link_source():
    r = bg.run_e91(bg.link(), rounds=None)
    assert 0 < r["qber"] < 0.05 and r["secure"]
