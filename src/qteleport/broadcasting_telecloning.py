from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Statevector
import numpy as np


def build_telecloning_circuit(theta, phi):
    """
    1 → 2 quantum telecloning (Murao et al.)

    Produces two approximate clones of the input state.

    Qubits:
    q0 : input (P)
    q1 : port qubit (PA)
    q2 : clone 1 (C1)
    q3 : clone 2 (C2)
    q4 : ancilla (A)
    """

    # Reference input state
    qc_in = QuantumCircuit(1)
    qc_in.ry(theta, 0)
    qc_in.rz(phi, 0)
    sv_input = Statevector.from_instruction(qc_in)

    # Registers
    q  = QuantumRegister(5, 'q')
    cm = ClassicalRegister(2, 'cm')

    qc = QuantumCircuit(q, cm)

    # ── Input state ──
    qc.ry(theta, q[0])
    qc.rz(phi, q[0])

    # ── Telecloning channel state ──
    angle_PA = 2 * np.arccos(np.sqrt(2/3))

    qc.ry(angle_PA, q[1])
    qc.cx(q[1], q[4])

    qc.h(q[2])
    qc.cx(q[2], q[3])

    qc.ry(np.pi / 4, q[2])

    qc.cx(q[1], q[2])
    qc.cx(q[1], q[3])

    # ── Bell measurement ──
    qc.cx(q[0], q[1])
    qc.h(q[0])

    qc.measure(q[0], cm[0])
    qc.measure(q[1], cm[1])

    # ── Corrections (applied to BOTH clones) ──
    # Clone 1
    with qc.if_test((cm[1], 1)):
        qc.x(q[2])
    with qc.if_test((cm[0], 1)):
        qc.z(q[2])

    # Clone 2
    with qc.if_test((cm[1], 1)):
        qc.x(q[3])
    with qc.if_test((cm[0], 1)):
        qc.z(q[3])

    return qc, sv_input