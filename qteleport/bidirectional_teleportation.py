#!/usr/bin/env python
# coding: utf-8

# # Bidirectional Quantum Teleportation in Qiskit
# 
# This notebook implements bidirectional quantum teleportation, where Alice and Bob exchange unknown one-qubit states simultaneously using a shared GHZ entangled channel.
# 
# The workflow is organised from setup and hardware inspection, to circuit construction and compilation, to ideal/noisy simulation, hardware-oriented sampling, and finally a backend-by-backend results sweep.
# 
# The goal is to study both the protocol itself and how realistic compilation and noise affect the fidelity of each teleportation direction.

# ## 1 · Imports and Required Modules
# 
# This section imports the Qiskit tools needed for circuit construction, transpilation, simulation, visualization, fidelity analysis, and IBM Runtime sampling. It also prepares the fake backends that will be used later as hardware targets.

# In[22]:


import numpy as np
import matplotlib.pyplot as plt

# Qiskit core
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.quantum_info import Statevector, DensityMatrix, state_fidelity, partial_trace

# Transpiler — preset + custom pass manager
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.transpiler import PassManager, CouplingMap
from qiskit.transpiler.passes import (
    SabreLayout, SabreSwap, BasisTranslator, Optimize1qGatesDecomposition
)
from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary

# Aer simulator
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (
    NoiseModel, depolarizing_error, thermal_relaxation_error, ReadoutError
)

# IBM Runtime — Fake backends + SamplerV2
from qiskit_ibm_runtime.fake_provider import (
    FakeKawasaki,
    FakeSherbrooke,
    FakeKyiv)
from qiskit_ibm_runtime import SamplerV2

# Visualisation
from qiskit.visualization import plot_gate_map, plot_circuit_layout

# -------------------------------
# ✅ BACKEND DICTIONARY (IMPORTANT)
# -------------------------------
backend_dict = {
    "kawasaki"  : FakeKawasaki(),
    "sherbrooke": FakeSherbrooke(),
    "kyiv"      : FakeKyiv()
}

# -------------------------------
# ✅ EXTRACT HARDWARE PROPERTIES
# -------------------------------
backend_data = {}

for name, backend in backend_dict.items():
    config = backend.configuration()

    backend_data[name] = {
        "backend"     : backend,
        "coupling_map": CouplingMap(config.coupling_map),
        "basis_gates" : config.basis_gates,
        "num_qubits"  : config.num_qubits
    }

# -------------------------------
# DEBUG PRINTS
# -------------------------------
print("All imports OK\n")

for name, data in backend_data.items():
    print(f"{name.upper()}:")
    print(f"  Qubits: {data['num_qubits']}")
    print(f"  Basis gates: {data['basis_gates']}")
    print(f"  Coupling map size: {len(data['coupling_map'].get_edges())}")
    print()


# ## 2 · Backend Information and Chip Visualization
# 
# Here the notebook inspects the selected fake IBM backends, including their qubit counts, basis gates, and coupling maps. The chip layouts are plotted so we can see the hardware connectivity that the transpiler must respect when mapping the five logical qubits.

# In[23]:


import matplotlib.pyplot as plt

for name, data in backend_data.items():
    backend = data["backend"]

    print(f"\n{name.upper()}")
    print(f"Qubits: {data['num_qubits']}")
    print(f"Basis gates: {data['basis_gates']}")

    plt.figure()  # ⭐ THIS FIXES IT
    fig = plot_gate_map(backend)
    display(fig)


# ## 3 · Constructing the Bidirectional Teleportation Circuit
# 
# This is the main protocol section. The code builds a five-qubit circuit that performs teleportation in both directions at once.
# 
# **Qubit roles:**
# 
# | Qubit | Role in the code |
# |---|---|
# | `q[0]` | Alice's input qubit carrying `|psi>` |
# | `q[1]` | Bob's output location for Alice's state |
# | `q[2]` | Middle GHZ qubit that distributes entanglement |
# | `q[3]` | Alice's output location for Bob's state |
# | `q[4]` | Bob's input qubit carrying `|phi>` |
# 
# **Classical registers:**
# - `ca[0]` and `ca[1]` store Alice's measurement bits.
# - `cb[0]` and `cb[1]` store Bob's measurement bits.
# 
# **What the code does in order:**
# 1. It prepares Alice's input state on `q[0]` and Bob's input state on `q[4]`.
# 2. It creates a GHZ state across `q[1]`, `q[2]`, and `q[3]`.
# 3. Alice applies `CX(q[0], q[1])` while Bob applies `CX(q[4], q[3])`.
# 4. Alice and Bob each apply a Hadamard gate to their own input qubit.
# 5. Alice measures `(q[0], q[1])` and Bob measures `(q[4], q[3])`.
# 6. They exchange classical bits.
# 7. Bob corrects `q[1]` using Alice's bits, and Alice corrects `q[3]` using Bob's bits.
# 
# At the end, Bob should recover Alice's original state on `q[1]`, and Alice should recover Bob's original state on `q[3]`.

# In[24]:


# ── Input state parameters ───────────────────────────────────────────────────
# |ψ⟩_A  (Alice sends to Bob)
theta_a = np.pi / 4
phi_a   = np.pi / 3

# |φ⟩_B  (Bob sends to Alice)
theta_b = np.pi / 3
phi_b   = np.pi / 6


def create_bidirectional_teleportation_circuit(theta_a, phi_a, theta_b, phi_b):
    # ── Reference statevectors (for fidelity) ────────────────────────────────
    qc_psi = QuantumCircuit(1)
    qc_psi.ry(theta_a, 0);  qc_psi.rz(phi_a, 0)
    sv_psi = Statevector.from_instruction(qc_psi)   # Alice → Bob

    qc_phi = QuantumCircuit(1)
    qc_phi.ry(theta_b, 0);  qc_phi.rz(phi_b, 0)
    sv_phi = Statevector.from_instruction(qc_phi)   # Bob → Alice

    print("Alice input |ψ⟩:", np.round(sv_psi.data, 4))
    print("Bob   input |φ⟩:", np.round(sv_phi.data, 4))

    # ── Registers ─────────────────────────────────────────────────────────────
    # q0=Alice_A, q1=GHZ_1(Bob target), q2=GHZ_2, q3=GHZ_3(Alice target), q4=Bob_B
    q  = QuantumRegister(5, 'q')
    ca = ClassicalRegister(2, 'ca')   # Alice measures (A,1) → bits i,j
    cb = ClassicalRegister(2, 'cb')   # Bob   measures (B,3) → bits i,j
    qc = QuantumCircuit(q, ca, cb)

    # ── Step 1: Prepare input states ──────────────────────────────────────────
    qc.ry(theta_a, q[0]);  qc.rz(phi_a, q[0])   # |ψ⟩_A
    qc.ry(theta_b, q[4]);  qc.rz(phi_b, q[4])   # |φ⟩_B
    qc.barrier(label="Input |ψ⟩_A, |φ⟩_B")

    # ── Step 2: Prepare GHZ state on q1, q2, q3 ──────────────────────────────
    # |GHZ⟩_123 = (1/√2)(|000⟩ + |111⟩)
    qc.h(q[2])
    qc.cx(q[2], q[1])
    qc.cx(q[2], q[3])
    qc.barrier(label="GHZ_123")

    # ── Step 3: Alice CNOT(A,1), Bob CNOT(B,3) → |Ψ₁⟩ ──────────────────────
    qc.cx(q[0], q[1])   # Alice: CNOT(A → GHZ_1)
    qc.cx(q[4], q[3])   # Bob:   CNOT(B → GHZ_3)
    qc.barrier(label="|Ψ₁⟩")

    # ── Step 4: Alice H(A), Bob H(B) ─────────────────────────────────────────
    qc.h(q[0])   # Alice H
    qc.h(q[4])   # Bob   H
    qc.barrier(label="H(A), H(B)")

    # ── Step 5: Alice measures (A,1); Bob measures (B,3) ─────────────────────
    # Alice: ca[0]=meas(A)=i,  ca[1]=meas(1)=j
    qc.measure(q[0], ca[0])
    qc.measure(q[1], ca[1])
    # Bob:   cb[0]=meas(B)=i,  cb[1]=meas(3)=j
    qc.measure(q[4], cb[0])
    qc.measure(q[3], cb[1])
    qc.barrier(label="Measure + Classical exchange (i,j)")

    # ── Step 6: Corrections ───────────────────────────────────────────────────
    # Bob corrects q[1] using Alice's bits ca[0]=i, ca[1]=j
    #   Bob applies Z^j X^i to q[1]
    with qc.if_test((ca[1], 1)):   # Z if j=1
        qc.z(q[1])
    with qc.if_test((ca[0], 1)):   # X if i=1
        qc.x(q[1])

    # Alice corrects q[3] using Bob's bits cb[0]=i, cb[1]=j
    #   Alice applies Z^i X^j to q[3]
    with qc.if_test((cb[0], 1)):   # Z if i=1
        qc.z(q[3])
    with qc.if_test((cb[1], 1)):   # X if j=1
        qc.x(q[3])
    qc.barrier(label="Bob: Z^j X^i on q1 | Alice: Z^i X^j on q3")

    return qc, sv_psi, sv_phi


#  Create circuit
qc, sv_psi, sv_phi = create_bidirectional_teleportation_circuit(
    theta_a, phi_a, theta_b, phi_b)

#  Circuit info
print(f"\nCircuit depth : {qc.depth()}")
print(f"Gate counts   : {dict(qc.count_ops())}")

#  Draw circuit
qc.draw('mpl', style="iqp", fold=-1)


# ## 4 · Transpilation with the Preset Pass Manager
# 
# This section compiles the bidirectional teleportation circuit with Qiskit's preset pass manager at `optimization_level = 2`. The goal is to convert the logical circuit into a backend-compatible form while respecting native gate and connectivity constraints.

# In[25]:


# Preset pass manager — mirrors professor's approach
def transpile_with_preset(qc, backend):
    pm_preset = generate_preset_pass_manager(
        backend=backend,
        optimization_level=2
    )

    qc_t = pm_preset.run(qc)

    print(f"Preset transpiled depth : {qc_t.depth()}")
    print(f"Gate counts : {dict(qc_t.count_ops())}")

    return qc_t

backend = backend_data["kawasaki"]["backend"]
qc_t_preset = transpile_with_preset(qc, backend)
qc_t_preset.draw('mpl', style="iqp", fold=-1)


# ## 5 · Transpilation with a Custom 4-Stage Pass Manager
# 
# To make the compilation flow more explicit, the notebook also defines a custom pass manager with four stages:
# 
# - `SabreLayout` for initial logical-to-physical qubit placement.
# - `SabreSwap` for routing through limited hardware connectivity.
# - `BasisTranslator` for rewriting the circuit into the backend's native gate set.
# - `Optimize1qGatesDecomposition` for simplifying single-qubit gate sequences.
# 
# This provides a second compiled version of the same protocol for comparison.

# In[26]:


coupling_map = CouplingMap(backend.coupling_map)
basis_gates  = list(backend.operation_names)

pm_custom = PassManager([
    # Stage 1 – Layout: map logical → physical qubits
    SabreLayout(coupling_map, seed=42),
    # Stage 2 – Routing: insert SWAP gates for connectivity
    SabreSwap(coupling_map, heuristic='decay', seed=42),
    # Stage 3 – Translation: rewrite to backend basis gates
    BasisTranslator(SessionEquivalenceLibrary, basis_gates),
    # Stage 4 – 1-qubit optimisation: merge/cancel single-qubit gates
    Optimize1qGatesDecomposition(basis=basis_gates),
])

qc_t_custom = pm_custom.run(qc)
print(f"Custom transpiled depth : {qc_t_custom.depth()}")
print(f"Gate counts : {dict(qc_t_custom.count_ops())}")

qc_t_custom.draw('mpl', style="iqp", fold=-1)


# ## 6 · Circuit Depth Comparison
# 
# This section compares the original circuit depth with the preset-transpiled and custom-transpiled depths. Since deeper circuits usually accumulate more noise on real hardware, the plot gives a quick view of how expensive the bidirectional teleportation protocol becomes after hardware-aware compilation.

# In[27]:


labels  = ['Original', 'Preset PM (opt=2)', 'Custom 4-stage PM']
depths  = [qc.depth(), qc_t_preset.depth(), qc_t_custom.depth()]
colors  = ['#5B9BD5', '#ED7D31', '#70AD47']

fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(labels, depths, color=colors, width=0.5, edgecolor='black', linewidth=0.6)
for bar, d in zip(bars, depths):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            str(d), ha='center', va='bottom', fontweight='bold')
ax.set_ylabel("Circuit Depth")
ax.set_title("Circuit Depth: Original vs Transpiled")
ax.set_ylim(0, max(depths) * 1.25)
plt.tight_layout()
plt.show()


# ## 7 · Building a Layered Noise Model
# 
# To study more realistic execution, the notebook creates a layered Aer noise model. It combines several important hardware error sources so that both teleportation directions can be tested under non-ideal conditions.
# 
# | Layer | Noise Type | Applied To |
# |---|---|---|
# | 1 | Depolarizing error | Single-qubit and two-qubit gates |
# | 2 | Thermal relaxation with T1 and T2 | Gates with finite duration |
# | 3 | Readout or SPAM error | Measurements |

# In[28]:


def build_noise_model(p1q=0.001, p2q=0.01,
                      T1=50e3, T2=70e3,
                      p_meas=0.02):
    """
    Build a layered noise model.

    Parameters
    ----------
    p1q   : depolarizing probability for 1-qubit gates
    p2q   : depolarizing probability for 2-qubit gates  (cx, ecr …)
    T1    : amplitude damping time (ns)
    T2    : dephasing time (ns)  — must be ≤ 2*T1
    p_meas: readout (SPAM) error probability per qubit
    """
    assert T2 <= 2 * T1, "T2 must be ≤ 2*T1 (physical constraint)"
    nm = NoiseModel()

    # ── Layer 1: Depolarizing gate errors ──────────────────────────────────────
    dep_1q = depolarizing_error(p1q, 1)
    dep_2q = depolarizing_error(p2q, 2)
    for gate in ['u', 'u1', 'u2', 'u3', 'id', 'rz', 'sx', 'x']:
        nm.add_all_qubit_quantum_error(dep_1q, gate)
    for gate in ['cx', 'ecr', 'cz']:
        nm.add_all_qubit_quantum_error(dep_2q, gate)

    # ── Layer 2: Thermal relaxation (T1/T2) ────────────────────────────────────
    # Gate times (ns) — typical IBM values
    gate_times = {
        'u'  : 50,   'u1': 0,  'u2': 50,  'u3': 100,
        'id' : 50,   'rz': 0,  'sx': 50,  'x' : 50,
        'cx' : 350,  'ecr': 350,
        'measure': 1000,
    }
    for gate, t_gate in gate_times.items():
        if t_gate == 0:
            continue
        therm = thermal_relaxation_error(T1, T2, t_gate)
        n_q = 2 if gate in ('cx', 'ecr') else 1
        if n_q == 2:
            therm2 = therm.expand(thermal_relaxation_error(T1, T2, t_gate))
            nm.add_all_qubit_quantum_error(therm2, gate)
        else:
            nm.add_all_qubit_quantum_error(therm, gate)

    # ── Layer 3: Readout / SPAM errors ─────────────────────────────────────────
    # Asymmetric: p(1|0) = p_meas/2,  p(0|1) = p_meas
    readout = ReadoutError([[1 - p_meas/2, p_meas/2],
                            [p_meas,       1 - p_meas]])
    nm.add_all_qubit_readout_error(readout)

    return nm

# Default noise model
noise_model = build_noise_model()
print(noise_model)


# ## 8 · Ideal Density-Matrix Simulation
# 
# Here the circuit is simulated without noise using the density-matrix method. After the protocol finishes, the notebook traces out all unused qubits to isolate Bob's received state on `q[1]` and Alice's received state on `q[3]`, then compares each one with its corresponding original input state using state fidelity.

# In[29]:


qc_ideal = qc.copy()
qc_ideal.save_density_matrix()

sim_ideal    = AerSimulator(method='density_matrix')
result_ideal = sim_ideal.run(qc_ideal, shots=4096).result()

dm_ideal = DensityMatrix(result_ideal.data()['density_matrix'])

# 5 qubits: q0(A), q1(GHZ_1/Bob-target), q2(GHZ_2), q3(GHZ_3/Alice-target), q4(B)
# Bob receives |ψ⟩ on q1  → trace out all except q1  = [0,2,3,4]
# Alice receives |φ⟩ on q3 → trace out all except q3  = [0,1,2,4]
rho_q1_ideal = partial_trace(dm_ideal, [0, 2, 3, 4])   # Bob's received state
rho_q3_ideal = partial_trace(dm_ideal, [0, 1, 2, 4])   # Alice's received state

fid_q1_ideal = state_fidelity(rho_q1_ideal, sv_psi)    # |ψ⟩ arrived at Bob q1
fid_q3_ideal = state_fidelity(rho_q3_ideal, sv_phi)    # |φ⟩ arrived at Alice q3

print(f"Ideal fidelity — Bob  q[1] vs |ψ⟩_A : {fid_q1_ideal:.6f}")
print(f"Ideal fidelity — Alice q[3] vs |φ⟩_B : {fid_q3_ideal:.6f}")

counts_ideal = result_ideal.get_counts()
print("\nMeasurement counts (ideal):")
for outcome, cnt in sorted(counts_ideal.items()):
    print(f"  |{outcome}>  {cnt:5d}  ({100*cnt/4096:.1f}%)")


# ## 9 · Noisy Density-Matrix Simulation
# 
# This section repeats the same bidirectional fidelity analysis with the layered noise model included. The circuit is first transpiled into an Aer-compatible basis, then simulated with noise, so the loss in teleportation quality can be measured in each direction.

# In[30]:


# Transpile to AerSimulator basis so noise model gates match
basis_aer  = ['u', 'cx', 'id', 'rz', 'sx', 'x', 'measure', 'if_else']
qc_noisy_t = transpile(qc, basis_gates=basis_aer, optimization_level=1)
qc_noisy_t.save_density_matrix()

sim_noisy    = AerSimulator(method='density_matrix', noise_model=noise_model)
result_noisy = sim_noisy.run(qc_noisy_t, shots=4096).result()

dm_noisy     = DensityMatrix(result_noisy.data()['density_matrix'])
rho_q1_noisy = partial_trace(dm_noisy, [0, 2, 3, 4])
rho_q3_noisy = partial_trace(dm_noisy, [0, 1, 2, 4])

fid_q1_noisy = state_fidelity(rho_q1_noisy, sv_psi)
fid_q3_noisy = state_fidelity(rho_q3_noisy, sv_phi)

print(f"Noisy fidelity — Bob  q[1] vs |ψ⟩_A : {fid_q1_noisy:.6f}")
print(f"Noisy fidelity — Alice q[3] vs |φ⟩_B : {fid_q3_noisy:.6f}")
print(f"\nFidelity degradation (q1) : {fid_q1_ideal - fid_q1_noisy:.6f}")
print(f"Fidelity degradation (q3) : {fid_q3_ideal - fid_q3_noisy:.6f}")


# ## 10 · Hardware-Oriented Sampling with SamplerV2
# 
# In this part, the circuit is converted into ISA form and executed through `SamplerV2` on the `FakeKawasaki` backend. The measurement counts are reported separately for Alice's classical register and Bob's classical register, matching the two sets of classical outcomes used for the correction step.

# In[31]:


# Build a version of the circuit with measure_all for SamplerV2
qc_sample = qc.copy()

# Transpile to ISA form required by SamplerV2
pm_sampler = generate_preset_pass_manager(backend=backend, optimization_level=2)
qc_isa = pm_sampler.run(qc_sample)

print("ISA circuit (hardware-native) drawn with IQP style:")
qc_isa.draw('mpl', style="iqp", fold=-1)


# In[32]:


# Run via SamplerV2 — professor's exact pattern
sampler    = SamplerV2(backend)
job        = sampler.run([qc_isa], shots=4096)
pub_result = job.result()[0]

# SamplerV2 PubResult → BitArray → counts
# Two classical registers: ca (Alice bits) and cb (Bob bits)
counts_ca = pub_result.data.ca.get_counts()   # Alice's (i,j) bits
counts_cb = pub_result.data.cb.get_counts()   # Bob's   (i,j) bits

print("SamplerV2 — Alice measurement counts (FakeKawasaki hardware noise):")
total_ca = sum(counts_ca.values())
for outcome, cnt in sorted(counts_ca.items()):
    print(f"  ca=|{outcome}>  {cnt:5d}  ({100*cnt/total_ca:.1f}%)")

print("\nSamplerV2 — Bob measurement counts:")
total_cb = sum(counts_cb.values())
for outcome, cnt in sorted(counts_cb.items()):
    print(f"  cb=|{outcome}>  {cnt:5d}  ({100*cnt/total_cb:.1f}%)")


# ## 11 · Fidelity Comparison for Both Teleportation Directions
# 
# This section tests several representative input states instead of only one example state. The fidelities are evaluated separately for the `A -> Bob` direction and the `B -> Alice` direction, both in ideal and noisy conditions.

# In[33]:


qc_psi_ref = QuantumCircuit(1);  qc_psi_ref.ry(theta_a, 0);  qc_psi_ref.rz(phi_a, 0)
qc_phi_ref = QuantumCircuit(1);  qc_phi_ref.ry(theta_b, 0);  qc_phi_ref.rz(phi_b, 0)

# Test states for Alice (A→Bob direction)
test_states_a = {
    '|0⟩'  : QuantumCircuit(1),
    '|1⟩'  : (lambda c: (c.x(0), c)[1])(QuantumCircuit(1)),
    '|+⟩'  : (lambda c: (c.h(0), c)[1])(QuantumCircuit(1)),
    '|-⟩'  : (lambda c: (c.h(0), c.z(0), c)[2])(QuantumCircuit(1)),
    '|ψ⟩'  : qc_psi_ref,
}

def run_bidir_fidelity(input_a_qc, input_b_qc, noisy=False):
    """Build & run bidirectional teleportation for given 1-qubit input circuits."""
    q_  = QuantumRegister(5, 'q')
    ca_ = ClassicalRegister(2, 'ca')
    cb_ = ClassicalRegister(2, 'cb')
    qc_ = QuantumCircuit(q_, ca_, cb_)

    qc_.compose(input_a_qc, qubits=[0], inplace=True)   # |ψ⟩_A on q0
    qc_.compose(input_b_qc, qubits=[4], inplace=True)   # |φ⟩_B on q4
    qc_.barrier()

    # GHZ on q1,q2,q3
    qc_.h(q_[2]);  qc_.cx(q_[2], q_[1]);  qc_.cx(q_[2], q_[3])
    qc_.barrier()

    # Alice CNOT(A,1), Bob CNOT(B,3)
    qc_.cx(q_[0], q_[1]);  qc_.cx(q_[4], q_[3])
    qc_.barrier()

    # H on A and B
    qc_.h(q_[0]);  qc_.h(q_[4])
    qc_.barrier()

    # Measure
    qc_.measure(q_[0], ca_[0]);  qc_.measure(q_[1], ca_[1])
    qc_.measure(q_[4], cb_[0]);  qc_.measure(q_[3], cb_[1])
    qc_.barrier()

    # Bob corrects q1: Z^j X^i using Alice bits ca[0]=i, ca[1]=j
    with qc_.if_test((ca_[1], 1)): qc_.z(q_[1])
    with qc_.if_test((ca_[0], 1)): qc_.x(q_[1])
    # Alice corrects q3: Z^i X^j using Bob bits cb[0]=i, cb[1]=j
    with qc_.if_test((cb_[0], 1)): qc_.z(q_[3])
    with qc_.if_test((cb_[1], 1)): qc_.x(q_[3])

    sv_a = Statevector.from_instruction(input_a_qc)
    sv_b = Statevector.from_instruction(input_b_qc)
    nm   = noise_model if noisy else None
    sim  = AerSimulator(method='density_matrix', noise_model=nm)
    if noisy:
        basis_aer = ['u', 'cx', 'id', 'rz', 'sx', 'x', 'measure', 'if_else']
        qc_ = transpile(qc_, basis_gates=basis_aer, optimization_level=1)
    qc_.save_density_matrix()
    res  = sim.run(qc_, shots=2048).result()
    dm_  = DensityMatrix(res.data()['density_matrix'])
    rho1 = partial_trace(dm_, [0, 2, 3, 4])   # Bob q1
    rho3 = partial_trace(dm_, [0, 1, 2, 4])   # Alice q3
    return state_fidelity(rho1, sv_a), state_fidelity(rho3, sv_b)


fid_q1_ideal_per, fid_q3_ideal_per = {}, {}
fid_q1_noisy_per, fid_q3_noisy_per = {}, {}

for label, input_a in test_states_a.items():
    f1i, f3i = run_bidir_fidelity(input_a, qc_phi_ref, noisy=False)
    f1n, f3n = run_bidir_fidelity(input_a, qc_phi_ref, noisy=True)
    fid_q1_ideal_per[label] = f1i;  fid_q3_ideal_per[label] = f3i
    fid_q1_noisy_per[label] = f1n;  fid_q3_noisy_per[label] = f3n

labels_ch = list(fid_q1_ideal_per.keys())
x = np.arange(len(labels_ch));  width = 0.2

fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
for ax, fid_id, fid_no, title in zip(
        axes,
        [fid_q1_ideal_per, fid_q3_ideal_per],
        [fid_q1_noisy_per, fid_q3_noisy_per],
        ["A→Bob (q1) receives |ψ⟩", "B→Alice (q3) receives |φ⟩"]):
    vals_id = [fid_id[k] for k in labels_ch]
    vals_no = [fid_no[k] for k in labels_ch]
    b1 = ax.bar(x - width/2, vals_id, width, label='Ideal', color='#5B9BD5', edgecolor='black', lw=0.6)
    b2 = ax.bar(x + width/2, vals_no, width, label='Noisy', color='#ED7D31', edgecolor='black', lw=0.6)
    for bar in [*b1, *b2]:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f"{bar.get_height():.3f}", ha='center', va='bottom', fontsize=7)
    ax.set_xticks(x);  ax.set_xticklabels(labels_ch)
    ax.set_ylabel("State Fidelity");  ax.set_ylim(0.8, 1.05)
    ax.set_title(title)
    ax.legend();  ax.axhline(1.0, color='grey', linestyle='--', lw=0.8)

fig.suptitle("Bidirectional Teleportation — Per-Channel Fidelity: Ideal vs Noisy", fontsize=12)
plt.tight_layout();  plt.show()


# ## 12 · Fidelity as Noise Strength Increases
# 
# To examine robustness, the notebook sweeps the two-qubit depolarizing error rate and recalculates the resulting fidelities for both output channels. This shows how sensitive each direction of bidirectional teleportation is to stronger entangling-gate noise.

# In[34]:


p2q_sweep = np.linspace(0.001, 0.15, 20)
fids_q1_sweep = []
fids_q3_sweep = []

for p2q in p2q_sweep:
    nm_sweep = build_noise_model(p1q=p2q/10, p2q=p2q, p_meas=p2q/5)
    q_  = QuantumRegister(5, 'q')
    ca_ = ClassicalRegister(2, 'ca')
    cb_ = ClassicalRegister(2, 'cb')
    qc_ = QuantumCircuit(q_, ca_, cb_)

    qc_.ry(theta_a, q_[0]);  qc_.rz(phi_a, q_[0])
    qc_.ry(theta_b, q_[4]);  qc_.rz(phi_b, q_[4]);  qc_.barrier()
    qc_.h(q_[2]);  qc_.cx(q_[2], q_[1]);  qc_.cx(q_[2], q_[3]);  qc_.barrier()
    qc_.cx(q_[0], q_[1]);  qc_.cx(q_[4], q_[3]);  qc_.barrier()
    qc_.h(q_[0]);  qc_.h(q_[4]);  qc_.barrier()
    qc_.measure(q_[0], ca_[0]);  qc_.measure(q_[1], ca_[1])
    qc_.measure(q_[4], cb_[0]);  qc_.measure(q_[3], cb_[1]);  qc_.barrier()
    with qc_.if_test((ca_[1], 1)): qc_.z(q_[1])
    with qc_.if_test((ca_[0], 1)): qc_.x(q_[1])
    with qc_.if_test((cb_[0], 1)): qc_.z(q_[3])
    with qc_.if_test((cb_[1], 1)): qc_.x(q_[3])

    basis_aer = ['u', 'cx', 'id', 'rz', 'sx', 'x', 'measure', 'if_else']
    qc_t = transpile(qc_, basis_gates=basis_aer, optimization_level=1)
    qc_t.save_density_matrix()
    sim_ = AerSimulator(method='density_matrix', noise_model=nm_sweep)
    res_ = sim_.run(qc_t, shots=2048).result()
    dm_  = DensityMatrix(res_.data()['density_matrix'])
    fids_q1_sweep.append(state_fidelity(partial_trace(dm_, [0, 2, 3, 4]), sv_psi))
    fids_q3_sweep.append(state_fidelity(partial_trace(dm_, [0, 1, 2, 4]), sv_phi))

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(p2q_sweep, fids_q1_sweep, 'o-', color='#5B9BD5', lw=2, ms=5, label='A→Bob q1 (|ψ⟩)')
ax.plot(p2q_sweep, fids_q3_sweep, 's--', color='#70AD47', lw=2, ms=5, label='B→Alice q3 (|φ⟩)')
ax.axhline(2/3, color='#ED7D31', linestyle='--', lw=1.5, label='Classical limit (2/3)')
ax.axhline(1.0, color='grey', linestyle=':', lw=1)
ax.fill_between(p2q_sweep, fids_q1_sweep, 2/3,
                where=[f > 2/3 for f in fids_q1_sweep],
                alpha=0.10, color='#5B9BD5')
ax.fill_between(p2q_sweep, fids_q3_sweep, 2/3,
                where=[f > 2/3 for f in fids_q3_sweep],
                alpha=0.10, color='#70AD47')
ax.set_xlabel("2-qubit Gate Depolarizing Error (p₂q)")
ax.set_ylabel("State Fidelity")
ax.set_title("Bidirectional Teleportation — Fidelity vs Noise Strength")
ax.legend();  ax.set_ylim(0.5, 1.05)
ax.grid(True, alpha=0.3)
plt.tight_layout();  plt.show()


# ## 13 · Final Results and Summary
# 
# The notebook ends by running the protocol across multiple fake backends, collecting the directional fidelities, and reporting an average bidirectional fidelity. Together, these results summarize how well the protocol works, how compilation changes the circuit, and how noise affects simultaneous two-way teleportation.

# ### Backend Sweep Helper Functions
# 
# The final section reuses three small helper functions so the backend comparison stays readable:
# 
# - `get_qubit_layout(...)` picks a simple five-qubit layout when a larger backend offers many choices.
# - `transpile_with_layout(...)` wraps `transpile(...)` so an optional initial layout can be supplied consistently.
# - `run_simulation(...)` executes the compiled circuit and returns both raw counts and normalized probabilities.
# 
# Placing these helpers here keeps them close to the backend sweep that uses them.
# 

# In[35]:


def get_qubit_layout(coupling_map, num_qubits, manual_map=None):
    if manual_map:
        return manual_map

    edges = coupling_map.get_edges()

    for edge in edges:
        if len(edge) >= num_qubits:
            return list(edge[:num_qubits])

    return list(range(num_qubits))


# In[36]:


def transpile_with_layout(qc, backend, layout=None):
    return transpile(
        qc,
        backend=backend,
        initial_layout=layout,
        optimization_level=2
    )


# In[37]:


def run_simulation(qc):
    sim = AerSimulator()
    result = sim.run(qc, shots=1024).result()

    counts = result.get_counts()
    probs = {k: v / 1024 for k, v in counts.items()}

    return counts, probs


# In[38]:


fidelity_results = {}
error_rates = []
all_results = []

for name, data in backend_data.items():
    backend = data["backend"]

    print(f"\n===== {name.upper()} =====")

    # 🔁 Create circuit
    qc, sv_psi, sv_phi = create_bidirectional_teleportation_circuit(
        theta_a, phi_a, theta_b, phi_b)

    # 🔧 choose layout (manual OR automatic)
    layout = None
    if backend.num_qubits >= 20:
        layout = get_qubit_layout(data["coupling_map"], 5)   # 5 qubits

    # Use layout-enabled transpile
    qc_t = transpile_with_layout(qc, backend, layout)

    # ▶ Run simulation
    counts, probs = run_simulation(qc_t)

    print("\nMeasurement outcomes:")
    for outcome, p in probs.items():
        print(f"  {outcome} → {p:.3f}")

    # Fidelity via AerSimulator density matrix with backend noise model
    nm_backend = NoiseModel.from_backend(backend)
    basis_aer  = ['u', 'cx', 'id', 'rz', 'sx', 'x', 'measure', 'if_else']
    qc_dm = transpile(qc, basis_gates=basis_aer, optimization_level=1)
    qc_dm.save_density_matrix()
    sim_dm = AerSimulator(method='density_matrix', noise_model=nm_backend)
    res_dm = sim_dm.run(qc_dm, shots=2048).result()
    dm_b   = DensityMatrix(res_dm.data()['density_matrix'])

    fid_q1 = float(state_fidelity(partial_trace(dm_b, [0, 2, 3, 4]), sv_psi))
    fid_q3 = float(state_fidelity(partial_trace(dm_b, [0, 1, 2, 4]), sv_phi))
    fidelity = (fid_q1 + fid_q3) / 2   # average bidirectional fidelity
    error    = 1 - fidelity

    fidelity_results[name] = {"q1": fid_q1, "q3": fid_q3, "avg": fidelity}
    error_rates.append(error)
    all_results.append({
        "backend"  : name,
        "counts"   : counts,
        "probs"    : probs,
        "fid_q1"   : fid_q1,
        "fid_q3"   : fid_q3,
        "fidelity" : fidelity,
        "error"    : error
    })
    print(f"Fidelity A→Bob  (q1): {fid_q1:.4f}")
    print(f"Fidelity B→Alice (q3): {fid_q3:.4f}")
    print(f"Average fidelity     : {fidelity:.4f}")
    print(f"Error                : {error:.4f}")


# ### Backend Sweep Outputs
# 
# After the helper definitions and backend loop run, the notebook finishes with three outputs:
# 
# - a bar chart comparing directional and average fidelities across backends,
# - a concise printed summary for each backend, and
# - a detailed dump of the sampled counts and fidelity values.
# 

# In[39]:


names    = list(fidelity_results.keys())
fids_q1  = [fidelity_results[n]["q1"]  for n in names]
fids_q3  = [fidelity_results[n]["q3"]  for n in names]
fids_avg = [fidelity_results[n]["avg"] for n in names]

x = np.arange(len(names));  width = 0.25
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(x - width, fids_q1,  width, label='A→Bob q1 (|ψ⟩)',  color='#5B9BD5', edgecolor='black')
ax.bar(x,         fids_q3,  width, label='B→Alice q3 (|φ⟩)', color='#70AD47', edgecolor='black')
ax.bar(x + width, fids_avg, width, label='Average',           color='#ED7D31', edgecolor='black')
ax.axhline(2/3, color='red', linestyle='--', lw=1.5, label='Classical limit (2/3)')
ax.set_xticks(x);  ax.set_xticklabels(names)
ax.set_title("Bidirectional Teleportation — Fidelity across Backends")
ax.set_xlabel("Backend");  ax.set_ylabel("Fidelity")
ax.set_ylim(0, 1.1);  ax.legend()
plt.tight_layout();  plt.show()


# In[40]:


print("=" * 55)
print("  Bidirectional Quantum Teleportation")
print("  Results Summary")
print("=" * 55)

for name in fidelity_results:
    r = fidelity_results[name]
    print(f"\nBackend : {name}")
    print(f"Fidelity A→Bob  (q1) : {r['q1']:.6f}")
    print(f"Fidelity B→Alice (q3) : {r['q3']:.6f}")
    print(f"Average fidelity      : {r['avg']:.6f}")
    print(f"Error                 : {1 - r['avg']:.6f}")
    if r["avg"] > 2/3:
        print("✓ Beats classical limit")

print("=" * 55)


# In[41]:


print("\nDetailed Results:")
for res in all_results:
    print(f"\nBackend: {res['backend']}")
    print(f"Counts: {res['counts']}")
    print(f"Fidelity A→Bob  (q1): {res['fid_q1']:.4f}")
    print(f"Fidelity B→Alice (q3): {res['fid_q3']:.4f}")

