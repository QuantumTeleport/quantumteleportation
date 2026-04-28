from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Statevector


def build_multiparty_circuit(theta, phi):
    """
    Multi-party teleportation of a 2-qubit entangled state (EPR-like)
    from Alice → Bob & Claire.

    Qubits:
    q0, q1 : Alice input (EPR pair)
    q2     : Alice GHZ qubit
    q3     : Bob output
    q4     : Claire output
    """

    # Reference input state (2-qubit entangled state)
    qc_in = QuantumCircuit(2)
    qc_in.ry(theta, 0)
    qc_in.cx(0, 1)
    qc_in.rz(phi, 0)
    sv_input = Statevector.from_instruction(qc_in)

    # Registers
    q  = QuantumRegister(5, 'q')
    cm = ClassicalRegister(3, 'cm')

    qc = QuantumCircuit(q, cm)

    # ── Step 1: Input EPR pair ──
    qc.ry(theta, q[0])
    qc.cx(q[0], q[1])
    qc.rz(phi, q[0])

    # ── Step 2: GHZ state ──
    qc.h(q[2])
    qc.cx(q[2], q[3])
    qc.cx(q[2], q[4])

    # ── Step 3: Bell-type operations ──
    qc.cx(q[0], q[2])
    qc.cx(q[1], q[2])
    qc.h(q[0])
    qc.h(q[1])

    # ── Step 4: Measurement ──
    qc.measure(q[0], cm[0])  # i
    qc.measure(q[1], cm[1])  # j
    qc.measure(q[2], cm[2])  # check bit

    # ── Step 5: Corrections (applied to BOTH receivers) ──
    # Bob (q3)
    with qc.if_test((cm[0], 1)):
        qc.x(q[3])
    with qc.if_test((cm[1], 1)):
        qc.z(q[3])

    # Claire (q4)
    with qc.if_test((cm[0], 1)):
        qc.x(q[4])
    with qc.if_test((cm[1], 1)):
        qc.z(q[4])

    return qc, sv_input