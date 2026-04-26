#!/usr/bin/env python
# coding: utf-8

# # Controlled Quantum Teleportation in Qiskit
# 
# This notebook implements the controlled quantum teleportation protocol, where an unknown quantum state is transferred from Alice to Bob with Charlie acting as a controller.
# 
# The notebook is organised from setup and hardware inspection, to circuit construction and compilation, to ideal/noisy simulation, hardware-oriented sampling, and finally a backend-by-backend comparison.
# 
# The goal is to show both the protocol logic and how compilation and noise affect Bob's recovered state when Charlie's participation is required.
# 

# ## 1 · Imports and Required Modules
# 
# This section imports the Qiskit tools needed for circuit construction, transpilation, simulation, visualization, fidelity analysis, and IBM Runtime sampling. It also prepares the fake backends that will later be used as hardware targets.

# In[1]:


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
# Here the notebook inspects the selected fake IBM backends, including their qubit count, basis gates, and coupling maps. The chip layouts are plotted so we can see the hardware connectivity that the transpiler must respect when mapping the controlled teleportation circuit.

# In[2]:


import matplotlib.pyplot as plt

for name, data in backend_data.items():
    backend = data["backend"]

    print(f"\n{name.upper()}")
    print(f"Qubits: {data['num_qubits']}")
    print(f"Basis gates: {data['basis_gates']}")

    plt.figure()
    fig = plot_gate_map(backend)
    display(fig)


# ## 3 · Constructing the Controlled Teleportation Circuit
# 
# This is the core protocol section. The code builds the circuit step by step in the same order the controlled teleportation process is performed.
# 
# **Qubit roles:**
# - `q[0]` stores Alice's unknown input state `|psi>`.
# - `q[1]`, `q[2]`, and `q[3]` form the GHZ entangled channel.
# - Bob receives the teleported state on `q[2]`.
# - Charlie controls completion of the protocol through `q[3]`.
# 
# **Classical registers:**
# - `ca[0]` stores Alice's first measurement bit `i`.
# - `ca[1]` stores Alice's second measurement bit `j`.
# - `cc[0]` stores Charlie's measurement bit `c`.
# 
# **What the code does in order:**
# 1. It prepares Alice's arbitrary input state using `ry(theta)` and `rz(phi)`.
# 2. It creates a GHZ state across `q[1]`, `q[2]`, and `q[3]`.
# 3. Alice performs `CX(q[0], q[1])` followed by `H(q[0])`.
# 4. Alice measures her two qubits and sends the classical bits `i` and `j` to Bob.
# 5. Charlie measures `q[3]` and sends the bit `c` to Bob.
# 6. Bob applies the conditional correction on `q[2]`: a `Z` gate if `i = 1`, and an `X` gate when `j XOR c = 1`.
# 
# This ordering is important because Bob can reconstruct the state only after both Alice's and Charlie's classical information are available.

# In[3]:


theta = np.pi / 4
phi   = np.pi / 3

def create_controlled_teleportation_circuit(theta, phi):
    # ── Ideal input state (for fidelity) ──
    qc_in = QuantumCircuit(1)
    qc_in.ry(theta, 0)
    qc_in.rz(phi, 0)
    sv_input = Statevector.from_instruction(qc_in)

    print("Ideal input statevector:")
    print(np.round(sv_input.data, 4))

    # ── Registers ────────────────────────────────────────────────────────────
    q  = QuantumRegister(4, 'q')    # q0=Alice_A, q1=GHZ_1, q2=Bob, q3=Charlie
    ca = ClassicalRegister(2, 'ca') # Alice: i=ca[0], j=ca[1]
    cc = ClassicalRegister(1, 'cc') # Charlie: c=cc[0]
    qc = QuantumCircuit(q, ca, cc)

    # ── Step 1: Prepare input |ψ⟩_A ─────────────────────────────────────────
    qc.ry(theta, q[0])
    qc.rz(phi,   q[0])
    qc.barrier(label="Input |ψ⟩")

    # ── Step 2: Prepare GHZ state on q1, q2, q3 ─────────────────────────────
    # |GHZ⟩_123 = (1/√2)(|000⟩ + |111⟩)
    qc.h(q[1])
    qc.cx(q[1], q[2])
    qc.cx(q[1], q[3])
    qc.barrier(label="GHZ")

    # ── Step 3: Alice CNOT(A,1) then H(A) ───────────────────────────────────
    qc.cx(q[0], q[1])
    qc.h(q[0])
    qc.barrier(label="Alice ops")

    # ── Step 4: Alice measures A and 1 → sends (i,j) to Bob ─────────────────
    qc.measure(q[0], ca[0])   # i
    qc.measure(q[1], ca[1])   # j
    qc.barrier(label="Alice meas → Bob")

    # ── Step 5: Charlie measures qubit 3 → sends c to Bob ───────────────────
    qc.measure(q[3], cc[0])   # c
    qc.barrier(label="Charlie meas → Bob")

    # ── Step 6: Bob applies U^{-1}_{ij,c} to q2 ────────────────────────────
    # Z correction: depends only on i (independent of Charlie)
    with qc.if_test((ca[0], 1)):
        qc.z(q[2])
    # X correction: depends on (j XOR c)
    #   j=0, c=1  →  X
    with qc.if_test((ca[1], 0)):
        with qc.if_test((cc[0], 1)):
            qc.x(q[2])
    #   j=1, c=0  →  X
    with qc.if_test((ca[1], 1)):
        with qc.if_test((cc[0], 0)):
            qc.x(q[2])
    qc.barrier(label="Bob correction")

    return qc, sv_input


#  Create circuit
qc, sv_input = create_controlled_teleportation_circuit(theta, phi)

#  Circuit info
print(f"\nCircuit depth : {qc.depth()}")
print(f"Gate counts   : {dict(qc.count_ops())}")

#  Draw circuit
qc.draw('mpl', style="iqp", fold=-1)


# ## 4 · Transpilation with the Preset Pass Manager
# 
# In this section the circuit is compiled with Qiskit's built-in preset pass manager using `optimization_level = 2`. The goal is to translate the controlled teleportation circuit into a backend-compatible form while reducing unnecessary overhead where possible.

# In[4]:


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
# To make the compilation process more explicit, the notebook also builds a custom pass manager with four stages:
# 
# - `SabreLayout` chooses an initial placement of logical qubits on physical qubits.
# - `SabreSwap` inserts routing swaps when connectivity constraints require them.
# - `BasisTranslator` rewrites gates into the backend's native gate set.
# - `Optimize1qGatesDecomposition` simplifies sequences of single-qubit gates.
# 
# This allows a direct comparison between Qiskit's default strategy and a manually structured compilation flow.

# In[5]:


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


# ## 6 · Depth Comparison of the Original and Transpiled Circuits
# 
# This section compares the depth of the original circuit with the preset-transpiled and custom-transpiled versions. Since deeper circuits are generally more vulnerable to decoherence and gate noise, this gives a quick indicator of how hardware-ready each compiled circuit is.

# In[6]:


labels  = ['Original', 'Preset PM (opt=2)', 'Custom 4-stage PM']
depths  = [qc.depth(), qc_t_preset.depth(), qc_t_custom.depth()]
colors  = ['#5B9BD5', '#ED7D31', '#70AD47']

fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(labels, depths, color=colors, width=0.5, edgecolor='black', linewidth=0.6)
for bar, d in zip(bars, depths):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            str(d), ha='center', va='bottom', fontweight='bold')
ax.set_ylabel("Circuit Depth")
ax.set_title("Controlled Teleportation — Circuit Depth: Original vs Transpiled")
ax.set_ylim(0, max(depths) * 1.25)
plt.tight_layout()
plt.show()


# ## 7 · Building a Layered Noise Model
# 
# To study realistic execution, the notebook creates a noise model for Aer simulation. The model combines several common hardware error sources so we can measure how much controlled teleportation fidelity degrades when noise is present.
# 
# | Layer | Noise Type | Applied To |
# |---|---|---|
# | 1 | Depolarizing error | Single-qubit and two-qubit gates |
# | 2 | Thermal relaxation with T1 and T2 | Gates with finite duration |
# | 3 | Readout or SPAM error | Measurements |
# 

# In[7]:


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

    # ── Layer 1: Depolarizing gate errors ──────────────────────────────────
    dep_1q = depolarizing_error(p1q, 1)
    dep_2q = depolarizing_error(p2q, 2)
    for gate in ['u', 'u1', 'u2', 'u3', 'id', 'rz', 'sx', 'x']:
        nm.add_all_qubit_quantum_error(dep_1q, gate)
    for gate in ['cx', 'ecr', 'cz']:
        nm.add_all_qubit_quantum_error(dep_2q, gate)

    # ── Layer 2: Thermal relaxation (T1/T2) ────────────────────────────────
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

    # ── Layer 3: Readout / SPAM errors ─────────────────────────────────────
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
# Here the controlled teleportation circuit is simulated without noise using the density-matrix method. After execution, all qubits except Bob's target qubit are traced out so the recovered state on `q[2]` can be directly compared with Alice's original input state through state fidelity.

# In[8]:


qc_ideal = qc.copy()
qc_ideal.save_density_matrix()

sim_ideal = AerSimulator(method='density_matrix')
result_ideal = sim_ideal.run(qc_ideal, shots=4096).result()

dm_ideal = DensityMatrix(result_ideal.data()['density_matrix'])
# 4 qubits: trace out q0(Alice_A), q1(GHZ_1), q3(Charlie) — keep q2(Bob)
rho_bob_ideal = partial_trace(dm_ideal, [0, 1, 3])

fidelity_ideal = state_fidelity(rho_bob_ideal, sv_input)
print(f"Ideal fidelity (Bob q[2] vs |ψ⟩): {fidelity_ideal:.6f}")

counts_ideal = result_ideal.get_counts()
print("\nMeasurement counts (ideal):")
for outcome, cnt in sorted(counts_ideal.items()):
    print(f"  |{outcome}>  {cnt:5d}  ({100*cnt/4096:.1f}%)")


# ## 9 · Noisy Density-Matrix Simulation
# 
# This section repeats the same analysis under the layered noise model. The circuit is first transpiled into an Aer-compatible basis, then simulated with noise, and Bob's reduced state is again compared with the original input state to quantify the loss in teleportation quality.

# In[9]:


# Transpile to AerSimulator basis so noise model gates match
basis_aer = ['u', 'cx', 'id', 'rz', 'sx', 'x', 'measure', 'if_else']
qc_noisy_t = transpile(qc, basis_gates=basis_aer, optimization_level=1)
qc_noisy_t.save_density_matrix()

sim_noisy = AerSimulator(method='density_matrix', noise_model=noise_model)
result_noisy = sim_noisy.run(qc_noisy_t, shots=4096).result()

dm_noisy = DensityMatrix(result_noisy.data()['density_matrix'])
# Trace out q0(Alice_A), q1(GHZ_1), q3(Charlie) — keep q2(Bob)
rho_bob_noisy = partial_trace(dm_noisy, [0, 1, 3])

fidelity_noisy = state_fidelity(rho_bob_noisy, sv_input)
print(f"Noisy fidelity (Bob q[2] vs |ψ⟩): {fidelity_noisy:.6f}")
print(f"Ideal fidelity                   : {fidelity_ideal:.6f}")
print(f"Fidelity degradation             : {fidelity_ideal - fidelity_noisy:.6f}")


# ## 10 · Hardware-Oriented Sampling with SamplerV2
# 
# In this section the circuit is converted into ISA form and executed through `SamplerV2` on the `FakeKawasaki` backend. The returned counts are separated into Alice's classical register and Charlie's classical register, reflecting the two sources of classical information required for Bob's correction step.

# In[10]:


# Build ISA circuit for SamplerV2
qc_sample = qc.copy()

pm_sampler = generate_preset_pass_manager(backend=backend, optimization_level=2)
qc_isa = pm_sampler.run(qc_sample)

print("ISA circuit (hardware-native) drawn with IQP style:")
qc_isa.draw('mpl', style="iqp", fold=-1)


# In[11]:


# Run via SamplerV2
sampler    = SamplerV2(backend)
job        = sampler.run([qc_isa], shots=4096)
pub_result = job.result()[0]

# SamplerV2 PubResult → BitArray → counts
# Classical registers: ca (2 bits) and cc (1 bit)
counts_ca = pub_result.data.ca.get_counts()  # Alice bits
counts_cc = pub_result.data.cc.get_counts()  # Charlie bits

print("SamplerV2 — Alice measurement counts (FakeKawasaki hardware noise):")
total = sum(counts_ca.values())
for outcome, cnt in sorted(counts_ca.items()):
    print(f"  ca=|{outcome}>  {cnt:5d}  ({100*cnt/total:.1f}%)")

print("\nSamplerV2 — Charlie measurement counts:")
total_cc = sum(counts_cc.values())
for outcome, cnt in sorted(counts_cc.items()):
    print(f"  cc=|{outcome}>  {cnt:5d}  ({100*cnt/total_cc:.1f}%)")


# ## 11 · Fidelity Comparison Across Different Input States
# 
# Controlled teleportation is tested on multiple representative input states rather than only one example state. This helps show whether the protocol behaves consistently across basis states, superposition states, and the chosen arbitrary state used earlier in the notebook.

# In[12]:


qc_in = QuantumCircuit(1)
qc_in.ry(theta, 0)
qc_in.rz(phi, 0)

# Fidelity across multiple input states
test_states = {
    '|0⟩'  : QuantumCircuit(1),
    '|1⟩'  : (lambda c: (c.x(0), c)[1])(QuantumCircuit(1)),
    '|+⟩'  : (lambda c: (c.h(0), c)[1])(QuantumCircuit(1)),
    '|-⟩'  : (lambda c: (c.h(0), c.z(0), c)[2])(QuantumCircuit(1)),
    '|ψ⟩'  : qc_in,
}

def run_controlled_teleport_fidelity(input_qc, noisy=False):
    """Build & run controlled teleportation for a given 1-qubit input circuit."""
    q_  = QuantumRegister(4, 'q')
    ca_ = ClassicalRegister(2, 'ca')
    cc_ = ClassicalRegister(1, 'cc')
    qc_ = QuantumCircuit(q_, ca_, cc_)

    qc_.compose(input_qc, qubits=[0], inplace=True)
    qc_.barrier()

    # GHZ on q1,q2,q3
    qc_.h(q_[1]);  qc_.cx(q_[1], q_[2]);  qc_.cx(q_[1], q_[3])
    qc_.barrier()

    # Alice: CNOT(A,1), H(A)
    qc_.cx(q_[0], q_[1]);  qc_.h(q_[0])
    qc_.barrier()

    # Measure Alice and Charlie
    qc_.measure(q_[0], ca_[0]);  qc_.measure(q_[1], ca_[1])
    qc_.measure(q_[3], cc_[0])
    qc_.barrier()

    # Bob correction: Z if i=1; X if (j XOR c)=1
    with qc_.if_test((ca_[0], 1)): qc_.z(q_[2])
    with qc_.if_test((ca_[1], 0)):
        with qc_.if_test((cc_[0], 1)): qc_.x(q_[2])
    with qc_.if_test((ca_[1], 1)):
        with qc_.if_test((cc_[0], 0)): qc_.x(q_[2])

    sv_ref = Statevector.from_instruction(input_qc)
    nm = noise_model if noisy else None
    sim = AerSimulator(method='density_matrix', noise_model=nm)
    if noisy:
        basis_aer = ['u', 'cx', 'id', 'rz', 'sx', 'x', 'measure', 'if_else']
        qc_ = transpile(qc_, basis_gates=basis_aer, optimization_level=1)
    qc_.save_density_matrix()
    res = sim.run(qc_, shots=2048).result()
    dm_ = DensityMatrix(res.data()['density_matrix'])
    rho = partial_trace(dm_, [0, 1, 3])   # keep Bob q2
    return state_fidelity(rho, sv_ref)

fid_ideal_per = {k: run_controlled_teleport_fidelity(v, noisy=False) for k, v in test_states.items()}
fid_noisy_per = {k: run_controlled_teleport_fidelity(v, noisy=True)  for k, v in test_states.items()}

labels_ch = list(fid_ideal_per.keys())
vals_id   = [fid_ideal_per[k] for k in labels_ch]
vals_no   = [fid_noisy_per[k] for k in labels_ch]

x = np.arange(len(labels_ch))
width = 0.35
fig, ax = plt.subplots(figsize=(9, 5))
b1 = ax.bar(x - width/2, vals_id, width, label='Ideal', color='#5B9BD5', edgecolor='black', lw=0.6)
b2 = ax.bar(x + width/2, vals_no, width, label='Noisy', color='#ED7D31', edgecolor='black', lw=0.6)
for bar in [*b1, *b2]:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
            f"{bar.get_height():.3f}", ha='center', va='bottom', fontsize=8)
ax.set_xticks(x);  ax.set_xticklabels(labels_ch)
ax.set_ylabel("State Fidelity");  ax.set_ylim(0.8, 1.05)
ax.set_title("Controlled Teleportation — Per-Channel Fidelity: Ideal vs Noisy")
ax.legend();  ax.axhline(1.0, color='grey', linestyle='--', lw=0.8)
plt.tight_layout();  plt.show()


# ## 12 · Fidelity as Noise Strength Increases
# 
# To evaluate robustness, this section sweeps the two-qubit noise level across a range of values and recalculates the teleportation fidelity each time. Because entangling gates are central to GHZ-state preparation and correction logic, this sweep highlights how sensitive the controlled teleportation protocol is to stronger hardware noise.

# In[13]:


p2q_sweep = np.linspace(0.001, 0.15, 20)
fidelities_sweep = []

for p2q in p2q_sweep:
    nm_sweep = build_noise_model(p1q=p2q/10, p2q=p2q, p_meas=p2q/5)
    q_  = QuantumRegister(4, 'q')
    ca_ = ClassicalRegister(2, 'ca')
    cc_ = ClassicalRegister(1, 'cc')
    qc_ = QuantumCircuit(q_, ca_, cc_)

    qc_.ry(theta, q_[0]);  qc_.rz(phi, q_[0]);  qc_.barrier()
    qc_.h(q_[1]);  qc_.cx(q_[1], q_[2]);  qc_.cx(q_[1], q_[3]);  qc_.barrier()
    qc_.cx(q_[0], q_[1]);  qc_.h(q_[0]);  qc_.barrier()
    qc_.measure(q_[0], ca_[0]);  qc_.measure(q_[1], ca_[1])
    qc_.measure(q_[3], cc_[0]);  qc_.barrier()
    with qc_.if_test((ca_[0], 1)): qc_.z(q_[2])
    with qc_.if_test((ca_[1], 0)):
        with qc_.if_test((cc_[0], 1)): qc_.x(q_[2])
    with qc_.if_test((ca_[1], 1)):
        with qc_.if_test((cc_[0], 0)): qc_.x(q_[2])

    basis_aer = ['u', 'cx', 'id', 'rz', 'sx', 'x', 'measure', 'if_else']
    qc_t = transpile(qc_, basis_gates=basis_aer, optimization_level=1)
    qc_t.save_density_matrix()
    sim_ = AerSimulator(method='density_matrix', noise_model=nm_sweep)
    res_ = sim_.run(qc_t, shots=2048).result()
    dm_  = DensityMatrix(res_.data()['density_matrix'])
    rho_ = partial_trace(dm_, [0, 1, 3])
    fidelities_sweep.append(state_fidelity(rho_, sv_input))

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(p2q_sweep, fidelities_sweep, 'o-', color='#5B9BD5', lw=2, ms=5, label='Controlled teleportation fidelity')
ax.axhline(2/3, color='#ED7D31', linestyle='--', lw=1.5, label='Classical limit (2/3)')
ax.axhline(1.0, color='grey', linestyle=':', lw=1)
ax.fill_between(p2q_sweep, fidelities_sweep, 2/3,
                where=[f > 2/3 for f in fidelities_sweep],
                alpha=0.15, color='#5B9BD5', label='Quantum advantage region')
ax.set_xlabel("2-qubit Gate Depolarizing Error (p₂q)")
ax.set_ylabel("State Fidelity F(ρ_Bob, |ψ⟩)")
ax.set_title("Controlled Teleportation — Fidelity vs Noise Strength")
ax.legend();  ax.set_ylim(0.5, 1.05)
ax.grid(True, alpha=0.3)
plt.tight_layout();  plt.show()


# ## 13 · Final Results and Summary
# 
# The notebook ends by collecting the fidelity results across the tested backends and simulations. Taken together, the sections show how controlled teleportation is built, how the protocol depends on Alice's and Charlie's measurement outcomes, how transpilation changes the circuit, and how noise reduces the fidelity of the recovered state on Bob's qubit.

# ### Backend Sweep Helper Functions
# 
# The final backend comparison reuses three helper functions so the sweep stays easy to read:
# 
# - `get_qubit_layout(...)` picks a simple four-qubit layout when a larger backend offers many possible mappings.
# - `transpile_with_layout(...)` wraps `transpile(...)` so an optional initial layout can be passed in a consistent way.
# - `run_simulation(...)` executes the compiled circuit and returns both raw counts and normalized probabilities.
# 
# Placing these helpers here keeps them close to the section that actually uses them.
# 

# In[14]:


def get_qubit_layout(coupling_map, num_qubits, manual_map=None):
    if manual_map:
        return manual_map

    edges = coupling_map.get_edges()

    for edge in edges:
        if len(edge) >= num_qubits:
            return list(edge[:num_qubits])

    return list(range(num_qubits))


# In[15]:


def transpile_with_layout(qc, backend, layout=None):
    return transpile(
        qc,
        backend=backend,
        initial_layout=layout,
        optimization_level=2
    )


# In[16]:


def run_simulation(qc):
    sim = AerSimulator()
    result = sim.run(qc, shots=1024).result()

    counts = result.get_counts()
    probs = {k: v / 1024 for k, v in counts.items()}

    return counts, probs


# In[17]:


fidelity_results = {}
error_rates = []
all_results = []

for name, data in backend_data.items():
    backend = data["backend"]

    print(f"\n===== {name.upper()} =====")

    # Recreate circuit for each backend
    qc, sv_input = create_controlled_teleportation_circuit(theta, phi)

    # Choose layout
    layout = None
    if backend.num_qubits >= 20:
        layout = get_qubit_layout(data["coupling_map"], 4)  # 4 qubits for controlled

    # Transpile and run
    qc_t = transpile_with_layout(qc, backend, layout)
    counts, probs = run_simulation(qc_t)

    print("\nBob outcomes (marginal over Alice+Charlie bits):")
    for outcome, p in probs.items():
        print(f"  {outcome} → {p:.3f}")

    print("Counts:", counts)

    # Fidelity via density matrix (AerSimulator)
    nm_backend = NoiseModel.from_backend(backend)
    basis_aer = ['u', 'cx', 'id', 'rz', 'sx', 'x', 'measure', 'if_else']
    qc_dm = transpile(qc, basis_gates=basis_aer, optimization_level=1)
    qc_dm.save_density_matrix()
    sim_dm = AerSimulator(method='density_matrix', noise_model=nm_backend)
    res_dm = sim_dm.run(qc_dm, shots=2048).result()
    dm_b   = DensityMatrix(res_dm.data()['density_matrix'])
    rho_b  = partial_trace(dm_b, [0, 1, 3])
    fidelity = float(state_fidelity(rho_b, sv_input))
    error    = 1 - fidelity

    fidelity_results[name] = fidelity
    error_rates.append(error)
    all_results.append({
        "backend" : name,
        "counts"  : counts,
        "probs"   : probs,
        "fidelity": fidelity,
        "error"   : error
    })
    print(f"Fidelity: {fidelity:.4f}")
    print(f"Error:    {error:.4f}")


# ### Backend Sweep Outputs
# 
# After the helper definitions and backend loop run, the notebook finishes with three outputs:
# 
# - a bar chart comparing the teleportation fidelity across backends,
# - a short printed summary for each backend, and
# - a detailed dump of the sampled counts and fidelity values.
# 

# In[18]:


plt.figure(figsize=(8, 5))
plt.bar(fidelity_results.keys(), fidelity_results.values(),
        color='#5B9BD5', edgecolor='black')
plt.axhline(2/3, color='red', linestyle='--', lw=1.5, label='Classical limit (2/3)')
plt.title("Controlled Teleportation — Fidelity across Backends")
plt.xlabel("Backend")
plt.ylabel("Fidelity")
plt.legend()
plt.ylim(0, 1.1)
plt.show()


# In[19]:


print("=" * 55)
print("  Controlled Teleportation (Gorbachev–Trubilko)")
print("  Results Summary")
print("=" * 55)

for name in fidelity_results:
    print(f"\nBackend : {name}")
    print(f"Fidelity : {fidelity_results[name]:.6f}")
    print(f"Error    : {1 - fidelity_results[name]:.6f}")

    if fidelity_results[name] > 2/3:
        print("✓ Beats classical limit")

print("=" * 55)


# In[20]:


print("\nDetailed Results:")
for res in all_results:
    print(f"\nBackend: {res['backend']}")
    print(f"Counts: {res['counts']}")
    print(f"Fidelity: {res['fidelity']:.4f}")

