# standard.py

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, thermal_relaxation_error, ReadoutError


# ─────────────────────────────────────────────
# 1. BUILD TELEPORTATION CIRCUIT (PURE LOGIC)
# ─────────────────────────────────────────────
def build_standard_circuit(theta, phi):
    """
    Build the standard 3-qubit teleportation circuit.

    Parameters
    ----------
    theta, phi : float
        Define input state |phi⟩

    Returns
    -------
    qc : QuantumCircuit
    sv_input : Statevector
    """

    # Input state (for reference)
    qc_in = QuantumCircuit(1)
    qc_in.ry(theta, 0)
    qc_in.rz(phi, 0)
    sv_input = Statevector.from_instruction(qc_in)

    # Registers
    q = QuantumRegister(3, 'q')
    c = ClassicalRegister(2, 'c')
    qc = QuantumCircuit(q, c)

    # Input state
    qc.ry(theta, q[0])
    qc.rz(phi, q[0])

    # Bell pair
    qc.h(q[1])
    qc.cx(q[1], q[2])

    # Bell measurement
    qc.cx(q[0], q[1])
    qc.h(q[0])

    # Measurement
    qc.measure(q[0], c[0])
    qc.measure(q[1], c[1])

    # Conditional correction
    with qc.if_test((c[1], 1)):
        qc.x(q[2])
    with qc.if_test((c[0], 1)):
        qc.z(q[2])

    return qc, sv_input


# ─────────────────────────────────────────────
# 2. RUN CIRCUIT (OPTIONAL EXECUTION LAYER)
# ─────────────────────────────────────────────
def run_circuit(qc, noise_model=None, shots=1024):
    """
    Execute circuit with optional noise.

    Parameters
    ----------
    qc : QuantumCircuit
    noise_model : optional
    shots : int

    Returns
    -------
    result, counts
    """

    simulator = AerSimulator(noise_model=noise_model)
    qc_t = transpile(qc, simulator)

    result = simulator.run(qc_t, shots=shots).result()
    counts = result.get_counts()

    return result, counts


# ─────────────────────────────────────────────
# 3. NOISE MODEL (OPTIONAL EXTENSION)
# ─────────────────────────────────────────────
def build_noise_model(p1q=0.001, p2q=0.01, T1=50e3, T2=70e3, p_meas=0.02):
    """
    Build layered noise model.
    """

    assert T2 <= 2 * T1, "T2 must be ≤ 2*T1"

    nm = NoiseModel()

    # Depolarizing errors
    dep_1q = depolarizing_error(p1q, 1)
    dep_2q = depolarizing_error(p2q, 2)

    for gate in ['rz', 'sx', 'x']:
        nm.add_all_qubit_quantum_error(dep_1q, gate)

    for gate in ['cx']:
        nm.add_all_qubit_quantum_error(dep_2q, gate)

    # Thermal relaxation
    therm_1q = thermal_relaxation_error(T1, T2, 50)
    therm_2q = thermal_relaxation_error(T1, T2, 350)

    nm.add_all_qubit_quantum_error(therm_1q, ['rz', 'sx', 'x'])
    nm.add_all_qubit_quantum_error(therm_2q.expand(therm_2q), ['cx'])

    # Readout error
    readout = ReadoutError([[1 - p_meas/2, p_meas/2],
                            [p_meas, 1 - p_meas]])
    nm.add_all_qubit_readout_error(readout)

    return nm