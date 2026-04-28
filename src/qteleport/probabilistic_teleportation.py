from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Statevector
import numpy as np


def build_probabilistic_circuit(theta, phi, n):
    """
    Probabilistic quantum teleportation (Agrawal & Pati).

    Parameters
    ----------
    theta, phi : input state parameters
    n          : entanglement parameter (0 < n ≤ 1)

    Returns
    -------
    qc         : QuantumCircuit
    sv_input   : reference statevector
    """

    # Reference input state
    qc_in = QuantumCircuit(1)
    qc_in.ry(theta, 0)
    qc_in.rz(phi, 0)
    sv_input = Statevector.from_instruction(qc_in)

    # Registers
    q  = QuantumRegister(4, 'q')    # q0=input, q1=res1, q2=Bob, q3=ancilla
    ca = ClassicalRegister(2, 'ca') # Alice bits
    cs = ClassicalRegister(1, 'cs') # success flag

    qc = QuantumCircuit(q, ca, cs)

    # ── Input state ──
    qc.ry(theta, q[0])
    qc.rz(phi, q[0])

    # ── Partially entangled resource ──
    theta_n = 2 * np.arctan(n)
    qc.ry(theta_n, q[1])
    qc.cx(q[1], q[2])

    # ── Alice operations ──
    qc.cx(q[0], q[1])
    qc.h(q[0])

    # ── R_n operation (ancilla) ──
    rn_angle = 2 * np.arcsin(n / np.sqrt(1 + n**2))
    qc.cry(rn_angle, q[1], q[3])

    # ── Measurements ──
    qc.measure(q[0], ca[0])
    qc.measure(q[1], ca[1])
    qc.measure(q[3], cs[0])  # success flag

    # ── Bob correction (ONLY on success) ──
    with qc.if_test((cs[0], 0)):
        with qc.if_test((ca[1], 1)):
            qc.x(q[2])
        with qc.if_test((ca[0], 1)):
            qc.z(q[2])

    return qc, sv_input