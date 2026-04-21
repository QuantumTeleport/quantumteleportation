#!/usr/bin/env python
# coding: utf-8

# # Standard Quantum Teleportation — FakeKawasaki + Noise Simulation
# 
# This notebook walks through a full quantum teleportation workflow, starting from circuit construction and ending with hardware-aware and noisy evaluations.
# 
# ### What this notebook covers
# - importing the Qiskit and IBM Runtime tools used throughout the analysis
# - inspecting fake IBM backends and their chip connectivity
# - building the standard 3-qubit teleportation circuit for an arbitrary input state
# - comparing preset transpilation with a custom 4-stage pass manager
# - studying ideal, noisy, and backend-aware execution behaviour
# - measuring teleportation fidelity across channels, noise strengths, and multiple backends
# 
# The overall goal is to show not just that teleportation works in theory, but also how compilation strategy and hardware noise affect the final transmitted state.
# 

# ## 1 · Imports
# 
# This section loads the numerical, visualization, simulation, transpiler, and backend tools used in the notebook. It also builds a small backend dictionary so the rest of the workflow can switch between fake IBM devices in a consistent way.
# 

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
    "kawasaki": FakeKawasaki(),
    "sherbrooke": FakeSherbrooke(),
    "kyiv": FakeKyiv()
}

# -------------------------------
# ✅ EXTRACT HARDWARE PROPERTIES
# -------------------------------
backend_data = {}

for name, backend in backend_dict.items():
    config = backend.configuration()

    backend_data[name] = {
        "backend": backend,
        "coupling_map": CouplingMap(config.coupling_map),
        "basis_gates": config.basis_gates,
        "num_qubits": config.num_qubits
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


# ## 2 · Backend & Chip Visualisation
# 
# Before running teleportation, this section inspects the target hardware models. The printed backend information and chip map help show how many qubits are available, which basis gates are native, and how qubits are connected on the device.
# 

# In[2]:


import matplotlib.pyplot as plt

for name, data in backend_data.items():
    backend = data["backend"]

    print(f"\n{name.upper()}")
    print(f"Qubits: {data['num_qubits']}")
    print(f"Basis gates: {data['basis_gates']}")

    plt.figure()  # ⭐ THIS FIXES IT
    fig = plot_gate_map(backend)
    display(fig)


# ### Layout Helper
# 
# This helper picks a physical-qubit layout for the teleportation circuit. It is used later when trying layout-aware transpilation across different backends.
# 

# In[3]:


def get_qubit_layout(coupling_map, num_qubits, manual_map=None):
    if manual_map:
        return manual_map

    edges = coupling_map.get_edges()

    for edge in edges:
        if len(edge) >= num_qubits:
            return list(edge[:num_qubits])

    return list(range(num_qubits))


# ## 3 · Build the Teleportation Circuit
# 
# Here the notebook prepares an arbitrary single-qubit input state using `theta` and `phi`, then builds the standard 3-qubit teleportation protocol. The circuit includes Bell-pair creation, Alice's measurement stage, and Bob's conditional correction operations.
# 

# In[4]:


theta = np.pi / 4
phi   = np.pi / 3

def create_teleportation_circuit(theta, phi):
    # ── Ideal input state (for fidelity) ──
    qc_in = QuantumCircuit(1)
    qc_in.ry(theta, 0)
    qc_in.rz(phi, 0)
    sv_input = Statevector.from_instruction(qc_in)

    print("Ideal input statevector:")
    print(np.round(sv_input.data, 4))

    # ── Registers ────────────────────────
    q = QuantumRegister(3, 'q')   # q0=input, q1=Alice, q2=Bob
    c = ClassicalRegister(2, 'c')
    qc = QuantumCircuit(q, c)

    # ── Step 1: Input state ─────────────
    qc.ry(theta, q[0])
    qc.rz(phi,   q[0])
    qc.barrier(label="Input |ψ⟩")

    # ── Step 2: Bell pair ──────────────
    qc.h(q[1])
    qc.cx(q[1], q[2])
    qc.barrier(label="Bell pair")

    # ── Step 3: Bell measurement ───────
    qc.cx(q[0], q[1])
    qc.h(q[0])
    qc.barrier(label="Bell meas")

    # ── Step 4: Classical communication ─
    qc.measure(q[0], c[0])
    qc.measure(q[1], c[1])
    qc.barrier(label="Classical comm")

    # ── Step 5: Bob correction ─────────
    with qc.if_test((c[1], 1)):
        qc.x(q[2])
    with qc.if_test((c[0], 1)):
        qc.z(q[2])

    qc.barrier(label="Bob correction")

    print(f"Logical circuit depth : {qc.depth()}")
    print(f"Gate counts           : {dict(qc.count_ops())}")

    return qc, sv_input

qc, sv_input = create_teleportation_circuit(theta, phi)
qc.draw('mpl', style="iqp", fold=-1)


# ## 4 · Preset Pass Manager (`optimization_level=2`)
# 
# This section uses Qiskit's preset pass manager to compile the teleportation circuit for the selected backend. It gives a convenient baseline for depth and gate-count comparisons before trying a more manual transpilation strategy.
# 

# In[5]:


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

qc_t_preset = transpile_with_preset(qc, backend)
qc_t = qc_t_preset

qc_t_preset.draw('mpl', style="iqp", fold=-1)


# ### Layout-Aware Transpilation Helper
# 
# This short helper wraps `transpile(...)` with an optional initial layout so the backend comparison later in the notebook can test manual or automatic qubit placement without rewriting the compilation call each time.
# 

# In[6]:


def transpile_with_layout(qc, backend, layout=None):
    return transpile(
        qc,
        backend=backend,
        initial_layout=layout,
        optimization_level=2
    )


# ## 5 · Custom 4-Stage Pass Manager
# 
# This section recreates the compilation flow more explicitly so each stage is easier to study.
# 
# Stages: **SabreLayout → SabreSwap → BasisTranslator → Optimize1q**
# 
# The idea is to control how logical qubits are placed, how routing is introduced, how gates are translated into the backend basis, and how single-qubit operations are simplified afterward.
# 

# In[7]:


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


# ### Quick Simulation Utility
# 
# This utility runs a shot-based simulator and converts raw counts into probabilities. It keeps the later backend summary code cleaner and easier to read.
# 

# In[8]:


def run_simulation(qc):
    sim = AerSimulator()
    result = sim.run(qc, shots=1024).result()

    counts = result.get_counts()
    probs = {k: v / 1024 for k, v in counts.items()}

    return counts, probs


# ## 6 · Depth Comparison
# 
# After building both transpiled versions, this section compares their circuit depths against the original circuit. It gives a quick visual sense of how much compilation overhead each strategy adds.
# 

# In[9]:


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


# ## 7 · Layered Noise Model
# 
# This section builds a more realistic simulator by combining several independent noise sources. Instead of using a single abstract error channel, the notebook separates gate noise, relaxation effects, and measurement errors so their combined impact is easier to interpret.
# 
# | Layer | Type | Applies to |
# |---|---|---|
# | 1 | Depolarizing gate error | 1-qubit and 2-qubit gates |
# | 2 | T1/T2 thermal relaxation | Gates through finite gate times |
# | 3 | SPAM / readout error | Measurement outcomes |
# 

# In[10]:


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

    # ── Layer 2: Thermal relaxation ────────────────────────────────────────────
    gate_times = {
        'u': 50, 'u1': 0, 'u2': 50, 'u3': 100, 'id': 50,
        'rz': 0, 'sx': 50, 'x': 50, 'cx': 350, 'ecr': 350, 'measure': 1000
    }
    for gate, t in gate_times.items():
        if t == 0:
            continue
        therm = thermal_relaxation_error(T1, T2, t)
        if gate in ('cx', 'ecr'):
            nm.add_all_qubit_quantum_error(therm.expand(therm), gate)
        else:
            nm.add_all_qubit_quantum_error(therm, gate)

    # ── Layer 3: Readout error ─────────────────────────────────────────────────
    readout = ReadoutError([[1 - p_meas/2, p_meas/2],
                            [p_meas, 1 - p_meas]])
    nm.add_all_qubit_readout_error(readout)

    return nm

noise_model = build_noise_model()
print(noise_model)


# ## 8 · Ideal Simulation (Density Matrix, No Noise)
# 
# This section evaluates the teleportation circuit in an ideal density-matrix simulator. The result acts as the reference case, showing the fidelity Bob should achieve when no hardware imperfections are present.
# 

# In[11]:


qc_ideal = qc.copy()
qc_ideal.save_density_matrix()

sim_ideal = AerSimulator(method='density_matrix')
result_ideal = sim_ideal.run(qc_ideal, shots=4096).result()

dm_ideal = DensityMatrix(result_ideal.data()['density_matrix'])
rho_bob_ideal = partial_trace(dm_ideal, [0, 1])   # trace out q0, q1

fidelity_ideal = state_fidelity(rho_bob_ideal, sv_input)
print(f"Ideal fidelity (Bob q[2] vs |ψ⟩): {fidelity_ideal:.6f}")

counts_ideal = result_ideal.get_counts()
print("\nMeasurement counts (ideal):")
for outcome, cnt in sorted(counts_ideal.items()):
    print(f"  |{outcome}>  {cnt:5d}  ({100*cnt/4096:.1f}%)")


# ## 9 · Noisy Simulation (Density Matrix + Layered Noise)
# 
# Here the same teleportation protocol is run again, but with the layered noise model enabled. Comparing this section with the ideal case makes the fidelity loss from realistic noise immediately visible.
# 

# In[12]:


# Transpile to AerSimulator basis so noise model gates match
basis_aer = ['u', 'cx', 'id', 'rz', 'sx', 'x', 'measure', 'if_else']
qc_noisy_t = transpile(qc, basis_gates=basis_aer, optimization_level=1)
qc_noisy_t.save_density_matrix()

sim_noisy = AerSimulator(method='density_matrix', noise_model=noise_model)
result_noisy = sim_noisy.run(qc_noisy_t, shots=4096).result()

dm_noisy = DensityMatrix(result_noisy.data()['density_matrix'])
rho_bob_noisy = partial_trace(dm_noisy, [0, 1])

fidelity_noisy = state_fidelity(rho_bob_noisy, sv_input)
print(f"Noisy fidelity (Bob q[2] vs |ψ⟩): {fidelity_noisy:.6f}")
print(f"Ideal fidelity                   : {fidelity_ideal:.6f}")
print(f"Fidelity degradation             : {fidelity_ideal - fidelity_noisy:.6f}")


# ## 10 · SamplerV2 on FakeKawasaki (IBM Runtime Primitive)
# 
# This section moves from Aer-style simulation to a hardware-aware runtime workflow. The circuit is transpiled into ISA form for `FakeKawasaki`, then sampled through `SamplerV2` to mimic how execution would look on an IBM-style backend.
# 

# In[13]:


# Build a version of the circuit with measure_all for SamplerV2
qc_sample = qc.copy()

# Transpile to ISA form required by SamplerV2
pm_sampler = generate_preset_pass_manager(backend=backend, optimization_level=2)
qc_isa = pm_sampler.run(qc_sample)

print("ISA circuit (hardware-native) drawn with IQP style:")
qc_isa.draw('mpl', style="iqp", fold=-1)


# In[14]:


# Run via SamplerV2 — professor's exact pattern
sampler = SamplerV2(backend)
job     = sampler.run([qc_isa], shots=4096)
pub_result = job.result()[0]

# SamplerV2 PubResult → BitArray → counts
counts_sampler = pub_result.data.c.get_counts()   # 'c' = ClassicalRegister name

print("SamplerV2 measurement counts (FakeKawasaki hardware noise):")
total = sum(counts_sampler.values())
for outcome, cnt in sorted(counts_sampler.items()):
    print(f"  |{outcome}>  {cnt:5d}  ({100*cnt/total:.1f}%)")


# ## 11 · Per-Channel Fidelity Comparison
# 
# Teleportation should work for more than one specially chosen state. This section tests several representative one-qubit inputs and compares how well the protocol preserves each state after transmission.
# 

# In[15]:


qc_in = QuantumCircuit(1)
qc_in.ry(theta, 0)
qc_in.rz(phi, 0)

# Fidelity across multiple input states (X, Y, Z, H, T axes)
test_states = {
    '|0⟩'  : QuantumCircuit(1),
    '|1⟩'  : (lambda c: (c.x(0), c)[1])(QuantumCircuit(1)),
    '|+⟩'  : (lambda c: (c.h(0), c)[1])(QuantumCircuit(1)),
    '|-⟩'  : (lambda c: (c.h(0), c.z(0), c)[2])(QuantumCircuit(1)),
    '|ψ⟩'  : qc_in,   # our theta/phi state
}

def run_teleport_fidelity(input_qc, noisy=False):
    """Build and run teleportation for a given 1-qubit input circuit."""
    q_ = QuantumRegister(3, 'q')
    c_ = ClassicalRegister(2, 'c')
    qc_ = QuantumCircuit(q_, c_)
    qc_.compose(input_qc, qubits=[0], inplace=True)
    qc_.barrier()
    qc_.h(q_[1]);  qc_.cx(q_[1], q_[2]);  qc_.barrier()
    qc_.cx(q_[0], q_[1]);  qc_.h(q_[0]);  qc_.barrier()
    qc_.measure(q_[0], c_[0]);  qc_.measure(q_[1], c_[1]);  qc_.barrier()
    with qc_.if_test((c_[1], 1)): qc_.x(q_[2])
    with qc_.if_test((c_[0], 1)): qc_.z(q_[2])
    qc_.save_density_matrix()

    input_sv = Statevector.from_instruction(input_qc)
    if noisy:
        qc_run = transpile(qc_, optimization_level=1)
        sim = AerSimulator(method='density_matrix', noise_model=noise_model)
    else:
        qc_run = qc_
        sim = AerSimulator(method='density_matrix')
    result = sim.run(qc_run, shots=2048).result()
    dm = DensityMatrix(result.data()['density_matrix'])
    rho_bob = partial_trace(dm, [0, 1])
    return state_fidelity(rho_bob, input_sv)

labels = []
fidelities_ideal_states = []
fidelities_noisy_states = []

for label, input_qc in test_states.items():
    labels.append(label)
    fidelities_ideal_states.append(run_teleport_fidelity(input_qc, noisy=False))
    fidelities_noisy_states.append(run_teleport_fidelity(input_qc, noisy=True))

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(x - width/2, fidelities_ideal_states, width, label='Ideal', color='#5B9BD5')
ax.bar(x + width/2, fidelities_noisy_states, width, label='Noisy', color='#ED7D31')
ax.axhline(2/3, color='grey', linestyle='--', linewidth=1, label='Classical limit')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(0.5, 1.05)
ax.set_ylabel('Fidelity')
ax.set_title('Teleportation Fidelity Across Input States')
ax.legend()
ax.grid(True, axis='y', alpha=0.25)
plt.tight_layout()
plt.show()


# ## 12 · Fidelity vs Noise Strength Sweep
# 
# This section studies how teleportation fidelity changes as the two-qubit error rate increases. The sweep helps show when the protocol remains clearly quantum and when noise starts pushing performance toward the classical limit.
# 

# In[16]:


p2q_sweep = np.linspace(0.001, 0.15, 20)
fidelities_sweep = []

for p2q in p2q_sweep:
    nm_sweep = build_noise_model(p1q=p2q/10, p2q=p2q, p_meas=p2q/5)
    q_ = QuantumRegister(3, 'q');  c_ = ClassicalRegister(2, 'c')
    qc_ = QuantumCircuit(q_, c_)
    qc_.ry(theta, q_[0]);  qc_.rz(phi, q_[0]);  qc_.barrier()
    qc_.h(q_[1]);  qc_.cx(q_[1], q_[2]);  qc_.barrier()
    qc_.cx(q_[0], q_[1]);  qc_.h(q_[0]);  qc_.barrier()
    qc_.measure(q_[0], c_[0]);  qc_.measure(q_[1], c_[1]);  qc_.barrier()
    with qc_.if_test((c_[1], 1)): qc_.x(q_[2])
    with qc_.if_test((c_[0], 1)): qc_.z(q_[2])
    basis_aer = ['u', 'cx', 'id', 'rz', 'sx', 'x', 'measure', 'if_else']
    qc_t = transpile(qc_, basis_gates=basis_aer, optimization_level=1)
    qc_t.save_density_matrix()
    sim_ = AerSimulator(method='density_matrix', noise_model=nm_sweep)
    res_ = sim_.run(qc_t, shots=2048).result()
    dm_  = DensityMatrix(res_.data()['density_matrix'])
    rho_ = partial_trace(dm_, [0, 1])
    fidelities_sweep.append(state_fidelity(rho_, sv_input))

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(p2q_sweep, fidelities_sweep, 'o-', color='#5B9BD5', lw=2, ms=5, label='Teleportation fidelity')
ax.axhline(2/3, color='#ED7D31', linestyle='--', lw=1.5, label='Classical limit (2/3)')
ax.axhline(1.0, color='grey', linestyle=':', lw=1)
ax.fill_between(p2q_sweep, fidelities_sweep, 2/3,
                where=[f > 2/3 for f in fidelities_sweep],
                alpha=0.15, color='#5B9BD5', label='Quantum advantage region')
ax.set_xlabel("2-qubit Gate Depolarizing Error (p₂q)")
ax.set_ylabel("State Fidelity F(ρ_Bob, |ψ⟩)")
ax.set_title("Fidelity vs Noise Strength")
ax.legend();  ax.set_ylim(0.5, 1.05)
ax.grid(True, alpha=0.3)
plt.tight_layout();  plt.show()


# ## 13 · Multi-Backend Summary
# 
# The final section gathers the workflow into a backend-by-backend comparison. It reruns the teleportation analysis across the available fake IBM devices, then summarizes fidelities, error rates, counts, and visual comparisons in one place.
# 

# In[17]:


fidelity_results = {}
error_rates = []
bob_results = []

for name, data in backend_data.items():
    backend = data["backend"]

    print(f"\n===== {name.upper()} =====")

    # 🔁 Create circuit
    qc, sv_input = create_teleportation_circuit(theta, phi)

    # 🔧 choose layout (manual OR automatic)
    layout = None   # default

    # Example manual layout (spread qubits apart)
    if backend.num_qubits >= 20:
        layout = [0, 5, 10]

    # Use layout-enabled transpile
    qc_t = transpile_with_layout(qc, backend, layout)

    # ▶ Run simulation
    counts, probs = run_simulation(qc_t)

    print("\nBob outcomes:")
    for outcome, p in probs.items():
        print(f"  {outcome} → {p:.3f}")

    print("Counts:", counts)
    print("Probabilities:", probs)

    # 🎯 Teleportation success (Bob correct outcomes)
    fidelity = probs.get('00', 0) + probs.get('11', 0)
    error = 1 - fidelity

    fidelity_results[name] = fidelity
    error_rates.append(error)
    all_results = []
    all_results.append({
        "backend": name,
        "counts": counts,
        "probs": probs,
        "fidelity": fidelity,
        "error": error
    })
    print(f"Fidelity: {fidelity:.4f}")
    print(f"Error: {error:.4f}")


# ### Plots and Printed Summary
# 
# These final cells turn the collected backend results into a bar chart and a compact text summary, making it easier to compare teleportation quality across the tested devices.
# 

# In[18]:


plt.figure(figsize=(8,5))
plt.bar(fidelity_results.keys(), fidelity_results.values())

plt.title("Fidelity across backends")
plt.xlabel("Backend")
plt.ylabel("Fidelity")

plt.show()


# In[19]:


print("=" * 55)
print("  Quantum Teleportation — Results Summary")
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

