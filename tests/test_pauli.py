import jax.numpy as jnp
import pytest

from piqcx.pauli import embed, infer_n_qubits, local_controls, pauli


def test_pauli_algebra(paulis):
    x, y, z = paulis["x"], paulis["y"], paulis["z"]
    assert jnp.allclose(x @ x, paulis["i"])
    assert jnp.allclose(x @ y, 1j * z)


def test_embed_x_on_second_qubit():
    x1 = embed(pauli("x"), 1, 2)
    expected = jnp.kron(pauli("i"), pauli("x"))
    assert x1.shape == (4, 4)
    assert jnp.allclose(x1, expected)


def test_local_controls_count():
    ha = local_controls(3, "xy")
    assert ha.shape == (6, 8, 8)


def test_infer_n_qubits():
    assert infer_n_qubits(8) == 3
    with pytest.raises(ValueError):
        infer_n_qubits(3)
