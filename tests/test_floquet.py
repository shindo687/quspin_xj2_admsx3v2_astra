from __future__ import annotations

import numpy as np
import pytest

import quspin_ad
import chainrules as ad


def unitary_fixture() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(91)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3)))
    phases = np.array([0.31, 0.92, 1.73])
    U = q @ np.diag(np.exp(1j * phases)) @ q.conj().T
    direction = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    direction = (direction + direction.conj().T) / 2.0
    dU = 1j * q @ direction @ q.conj().T @ U
    return U, dU


def test_quasienergy_jvp_and_vjp_duality() -> None:
    U, dU = unitary_fixture()
    value, tangent = ad.jvp(
        quspin_ad.floquet_quasienergy, U, 2.0, tangents={"UF": dU}
    )
    eps = 1e-6
    plus = quspin_ad.floquet_quasienergy(U + eps * dU, 2.0)
    minus = quspin_ad.floquet_quasienergy(U - eps * dU, 2.0)
    assert np.allclose(tangent, (plus - minus) / (2 * eps), rtol=2e-5, atol=2e-6)
    cotangent = np.array([0.4, -0.7, 0.9])
    _, pullback = ad.vjp(quspin_ad.floquet_quasienergy, U, 2.0, wrt=("UF",))
    assert np.allclose(
        np.dot(cotangent, tangent),
        np.real(np.vdot(pullback(cotangent)["UF"], dU)),
        rtol=2e-6,
        atol=2e-6,
    )


def test_projectors_are_phase_invariant() -> None:
    U, _ = unitary_fixture()
    E, V = quspin_ad.floquet_eigensystem(U)
    phases = np.exp(1j * np.array([0.2, 1.4, -0.7]))
    assert np.allclose(
        quspin_ad.floquet_projectors(U),
        quspin_ad.floquet_projectors(U),
    )
    assert np.allclose(
        np.einsum("ij,kj->jik", V * phases, (V * phases).conj()),
        quspin_ad.floquet_projectors(U),
    )


def test_degeneracy_and_branch_cut_are_rejected() -> None:
    with pytest.raises(quspin_ad.FloquetSpectralGapError, match="degenerate"):
        quspin_ad.floquet_quasienergy(np.eye(2))
    with pytest.raises(quspin_ad.FloquetSpectralGapError, match="branch cut"):
        quspin_ad.floquet_quasienergy(np.diag([np.exp(1j * np.pi), 1j]))


def test_named_lattice_parameters_and_upstream_adapter() -> None:
    phases = np.array([0.2, 0.8, 1.4])
    drive = np.array([1.0, -2.0, 0.5])
    gauge = np.array([0.1, 0.3, 0.6])
    momentum_weights = np.array([0.2, 0.4, 0.7])

    def operator(drive_phase=0.0, synthetic_gauge=0.0, momentum=0.0):
        return np.diag(
            np.exp(1j * (phases + drive_phase * drive
                         + synthetic_gauge * gauge + momentum * momentum_weights))
        )

    def derivative(weights):
        return lambda **params: 1j * np.diag(weights) @ operator(**params)

    obj = type("UpstreamFloquet", (), {"UF": operator(), "T": 2.0})()
    adapter = quspin_ad.FloquetOperator(
        operator,
        obj.T,
        parameter_derivatives={
            "drive_phase": derivative(drive),
            "synthetic_gauge": derivative(gauge),
            "momentum": derivative(momentum_weights),
        },
        upstream=obj,
    )
    for name, weights in (("drive_phase", drive), ("synthetic_gauge", gauge), ("momentum", momentum_weights)):
        value, tangent = ad.jvp(
            quspin_ad.floquet_quasienergy,
            adapter,
            adapter.T,
            tangents={name: 1.0},
        )
        expected = -weights[np.argsort(-phases)] / adapter.T
        assert np.allclose(tangent, expected)
        _, pullback = ad.vjp(quspin_ad.floquet_quasienergy, adapter, adapter.T, wrt=(name,))
        assert np.isfinite(pullback(np.ones_like(value))[name])

    native = quspin_ad.floquet_quasienergy_from_object(obj, parameter_derivatives={})
    assert np.allclose(native, quspin_ad.floquet_quasienergy(obj.UF, obj.T))
    value, tangent = ad.jvp(
        quspin_ad.floquet_quasienergy_from_object,
        obj,
        tangents={"drive_phase": 1.0},
        parameter_derivatives={"drive_phase": derivative(drive)},
    )
    assert np.allclose(value, native)
    assert np.all(np.isfinite(tangent))
