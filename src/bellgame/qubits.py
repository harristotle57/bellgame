"""Two-qubit states and gates as plain numpy arrays.

Conventions used everywhere in bellgame:

* Qubit ``|0>`` is horizontal polarization (H), ``|1>`` is vertical (V).
* All angles are in **degrees**.
* A two-qubit state is a 4x4 density matrix. Qubit 0 is Alice's, qubit 1 is Bob's.

Noise *in the network* (memories decaying, entanglement swapping) is simulated
by SeQUeNCe; see ``bg.run_network``. The functions here build states and turn
polarization.

Example:
    >>> import bellgame as bg
    >>> rho = bg.bell_pair(fidelity=0.9)
    >>> round(bg.fidelity(rho), 3)
    0.9
"""

import numpy as np

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

# Phi+ = (|00> + |11>) / sqrt(2) = (|HH> + |VV>) / sqrt(2)
PHI_PLUS = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
# Phi+, then Phi+ hit by a Z, an X, and a Y error (SeQUeNCe's Bell-diagonal order)
BELL_STATES = [
    PHI_PLUS,
    np.array([1, 0, 0, -1], dtype=complex) / np.sqrt(2),  # Phi-
    np.array([0, 1, 1, 0], dtype=complex) / np.sqrt(2),   # Psi+
    np.array([0, 1, -1, 0], dtype=complex) / np.sqrt(2),  # Psi-
]


def ket_to_dm(ket):
    """Turn a state vector into a density matrix |ket><ket|.

    Example:
        >>> ket_to_dm(np.array([1, 0])).real
        array([[1., 0.],
               [0., 0.]])
    """
    ket = np.asarray(ket, dtype=complex)
    return np.outer(ket, ket.conj())


def bell_pair(fidelity=1.0):
    """A Bell pair Phi+ mixed with the other three Bell states.

    ``fidelity=1`` gives the perfect pair. Lower fidelity spreads the rest
    evenly over the other Bell states (a "Werner state").

    Example:
        >>> rho = bell_pair(0.8)
        >>> rho.shape
        (4, 4)
    """
    if not 0 <= fidelity <= 1:
        raise ValueError(f"fidelity must be between 0 and 1, got {fidelity}")
    visibility = (4 * fidelity - 1) / 3
    return werner(visibility)


def werner(visibility):
    """Werner state: ``visibility * |Phi+><Phi+| + (1 - visibility) * I/4``.

    ``visibility=1`` is a perfect Bell pair, ``visibility=0`` is pure noise.

    Example:
        >>> round(fidelity(werner(1.0)), 3)
        1.0
    """
    if not -1 / 3 <= visibility <= 1:
        raise ValueError(f"visibility must be between -1/3 and 1, got {visibility}")
    return visibility * ket_to_dm(PHI_PLUS) + (1 - visibility) * np.eye(4) / 4


def fidelity(rho, target=PHI_PLUS):
    """Fidelity <target|rho|target> of a two-qubit state with a pure target (default Phi+).

    Example:
        >>> round(fidelity(bell_pair(0.75)), 3)
        0.75
    """
    target = np.asarray(target, dtype=complex)
    return float(np.real(target.conj() @ rho @ target))


def rx(angle_deg):
    """Single-qubit X rotation exp(-i angle X / 2), angle in degrees."""
    t = np.radians(angle_deg) / 2
    return np.array([[np.cos(t), -1j * np.sin(t)], [-1j * np.sin(t), np.cos(t)]])


def ry(angle_deg):
    """Single-qubit Y rotation exp(-i angle Y / 2), angle in degrees."""
    t = np.radians(angle_deg) / 2
    return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]], dtype=complex)


def rz(angle_deg):
    """Single-qubit Z rotation exp(-i angle Z / 2), angle in degrees."""
    t = np.radians(angle_deg) / 2
    return np.array([[np.exp(-1j * t), 0], [0, np.exp(1j * t)]])


def polarization_rotation(angles_deg):
    """A general polarization change, given by three angles in degrees.

    This is what a twisted fiber does to a photon (misalignment), and also what
    a set of wave plates does when you try to undo it (correction).
    ``angles_deg = [a, b, c]`` means: rotate by ``a`` about X, then ``b`` about Y,
    then ``c`` about Z, all on the polarization (Poincare) sphere at half angle,
    so ``[0, b, 0]`` turns linear polarization by ``b`` degrees.

    Example:
        >>> U = polarization_rotation([0, 90, 0])   # H -> V
        >>> np.round(np.abs(U @ [1, 0]), 3)
        array([0., 1.])
    """
    a, b, c = _three_angles(angles_deg)
    return rz(2 * c) @ ry(2 * b) @ rx(2 * a)


def _three_angles(angles_deg):
    angles = list(angles_deg)
    if len(angles) != 3:
        raise ValueError(f"expected 3 angles in degrees, got {angles_deg}")
    return angles


def apply_local(rho, alice=None, bob=None):
    """Apply a 2x2 unitary to Alice's qubit and/or Bob's qubit.

    Example:
        >>> rho = apply_local(bell_pair(), bob=ry(30))
    """
    ua = I2 if alice is None else alice
    ub = I2 if bob is None else bob
    u = np.kron(ua, ub)
    return u @ rho @ u.conj().T


def misalign(rho, alice_deg=(0, 0, 0), bob_deg=(0, 0, 0)):
    """Twist Alice's and Bob's polarization by three angles each (see polarization_rotation).

    Example:
        >>> rho = misalign(bell_pair(), bob_deg=[10, 5, 0])
        >>> fidelity(rho) < 1
        True
    """
    return apply_local(rho, polarization_rotation(alice_deg), polarization_rotation(bob_deg))


def bell_diagonal(weights):
    """A mixture of the four Bell states, given their weights in SeQUeNCe's order.

    ``weights = [Phi+, Phi-, Psi+, Psi-]``, which is SeQUeNCe's Bell-diagonal order
    "I, Z, X, Y": the error that turns Phi+ into each state. The first weight is
    the fidelity.

    Example:
        >>> rho = bell_diagonal([0.9, 0.1, 0, 0])
        >>> round(fidelity(rho), 3)
        0.9
    """
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (4,) or np.any(weights < -1e-12) or not np.isclose(weights.sum(), 1):
        raise ValueError(f"need 4 non-negative weights that add up to 1, got {list(weights)}")
    return sum(w * ket_to_dm(ket) for w, ket in zip(weights, BELL_STATES))
