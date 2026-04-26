# controlled_teleportation.py

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Statevector


def build_controlled_circuit(theta, phi):
    """
    Controlled quantum teleportation circuit.

    Qubit roles:
    q0 → Alice input
    q1,q2,q3 → GHZ resource
    q2 → Bob
    q3 → Charlie (controller)
    """

    # Input reference state (for later fidelity use if needed)
    qc_in = QuantumCircuit(1)
    qc_in.ry(theta, 0)
    qc_in.rz(phi, 0)
    sv_input = Statevector.from_instruction(qc_in)

    # Registers
    q  = QuantumRegister(4, 'q')
    ca = ClassicalRegister(2, 'ca')  # Alice bits
    cc = ClassicalRegister(1, 'cc')  # Charlie bit
    qc = QuantumCircuit(q, ca, cc)

    # ── Input state ──
    qc.ry(theta, q[0])
    qc.rz(phi,   q[0])

    # ── GHZ state ──
    qc.h(q[1])
    qc.cx(q[1], q[2])
    qc.cx(q[1], q[3])

    # ── Alice operations ──
    qc.cx(q[0], q[1])
    qc.h(q[0])

    # ── Measurements ──
    qc.measure(q[0], ca[0])   # i
    qc.measure(q[1], ca[1])   # j
    qc.measure(q[3], cc[0])   # c

    # ── Bob correction ──
    # Z if i = 1
    with qc.if_test((ca[0], 1)):
        qc.z(q[2])

    # X if j XOR c = 1
    with qc.if_test((ca[1], 0)):
        with qc.if_test((cc[0], 1)):
            qc.x(q[2])

    with qc.if_test((ca[1], 1)):
        with qc.if_test((cc[0], 0)):
            qc.x(q[2])

    return qc, sv_input