from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Statevector


def build_bidirectional_circuit(theta_a, phi_a, theta_b, phi_b):
    """
    Bidirectional quantum teleportation.

    Alice sends |ψ⟩ → Bob
    Bob sends |φ⟩ → Alice

    Qubit roles:
    q0 : Alice input
    q1 : Bob output
    q2 : GHZ middle
    q3 : Alice output
    q4 : Bob input
    """

    # Reference states (optional, useful later)
    qc_psi = QuantumCircuit(1)
    qc_psi.ry(theta_a, 0)
    qc_psi.rz(phi_a, 0)
    sv_psi = Statevector.from_instruction(qc_psi)

    qc_phi = QuantumCircuit(1)
    qc_phi.ry(theta_b, 0)
    qc_phi.rz(phi_b, 0)
    sv_phi = Statevector.from_instruction(qc_phi)

    # Registers
    q  = QuantumRegister(5, 'q')
    ca = ClassicalRegister(2, 'ca')  # Alice bits
    cb = ClassicalRegister(2, 'cb')  # Bob bits
    qc = QuantumCircuit(q, ca, cb)

    # ── Input states ──
    qc.ry(theta_a, q[0])
    qc.rz(phi_a, q[0])

    qc.ry(theta_b, q[4])
    qc.rz(phi_b, q[4])

    # ── GHZ state ──
    qc.h(q[2])
    qc.cx(q[2], q[1])
    qc.cx(q[2], q[3])

    # ── Teleportation operations ──
    qc.cx(q[0], q[1])   # Alice → Bob
    qc.cx(q[4], q[3])   # Bob → Alice

    qc.h(q[0])
    qc.h(q[4])

    # ── Measurements ──
    qc.measure(q[0], ca[0])
    qc.measure(q[1], ca[1])

    qc.measure(q[4], cb[0])
    qc.measure(q[3], cb[1])

    # ── Corrections ──
    # Bob corrects q1 using Alice bits
    with qc.if_test((ca[1], 1)):
        qc.z(q[1])
    with qc.if_test((ca[0], 1)):
        qc.x(q[1])

    # Alice corrects q3 using Bob bits
    with qc.if_test((cb[0], 1)):
        qc.z(q[3])
    with qc.if_test((cb[1], 1)):
        qc.x(q[3])

    return qc, sv_psi, sv_phi