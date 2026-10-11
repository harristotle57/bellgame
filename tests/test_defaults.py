import inspect

import numpy as np
import pytest

import bellgame as bg
from bellgame.defaults import ARGUMENTS, DEFAULTS, check


@pytest.mark.parametrize("bad", ["ten", None, True, np.nan, [1]])
def test_numbers_refuse_non_numbers_and_nan(bad):
    with pytest.raises(ValueError, match="sim_time_s"):
        check("sim_time_s", bad)


@pytest.mark.parametrize("name, good, bad", [
    ("fidelity", [0, 1], [-0.1, 1.1]),
    ("time_limit_s", [1e-9, np.inf], [0]),
    ("sim_time_s", [1e-9], [0, np.inf]),
    ("visibility", [-1 / 3, 1], [-0.34, 1.01]),
    ("angle_deg", [-720, 45], [np.inf]),
])
def test_ranges(name, good, bad):
    for value in good:
        check(name, value)
    for value in bad:
        with pytest.raises(ValueError, match=f"{name} must be in"):
            check(name, value)


def test_whole_numbers():
    assert check("rounds", np.float64(5.0)) == 5 and type(check("rounds", 5.0)) is int
    with pytest.raises(ValueError, match="whole number"):
        check("rounds", 2.5)


def test_lists():
    assert check("rotation_deg", np.array([1, 2, 3])) == [1.0, 2.0, 3.0]
    for bad in ([1, 2], "abc", 5, [1, np.nan, 0]):
        with pytest.raises(ValueError, match="rotation_deg"):
            check("rotation_deg", bad)


def test_where_starts_the_message():
    with pytest.raises(ValueError, match=r"^strategy\['bob'\]: angles_deg"):
        check("angles_deg", [1, 2, 3], where="strategy['bob']")


def test_every_entry_is_described_and_checkable():
    for name, entry in {**DEFAULTS, **ARGUMENTS}.items():
        assert entry["meaning"] and "unit" in entry
        assert bg.describe(name).startswith(name)
        if "value" in entry and entry["value"] is not None:
            assert check(name, entry["value"]) == entry["value"]


def test_function_defaults_obey_their_rules():
    for function in (bg.play, bg.run_e91, bg.run_network, bg.find_alignment, bg.compare_optimizers, bg.bell_pair):
        for name, parameter in inspect.signature(function).parameters.items():
            if name in ARGUMENTS and parameter.default not in (None, inspect.Parameter.empty):
                check(name, parameter.default)
