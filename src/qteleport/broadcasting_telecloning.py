#!/usr/bin/env python
# coding: utf-8

# # Quantum Broadcasting / Telecloning — Murao et al. (FakeKawasaki + Noise Simulation)
# 
# Quantum telecloning broadcasts a single input state $|\psi\rangle_P$ to **multiple receivers** simultaneously via a shared cloning channel state $|\xi\rangle_{PA,C}$.
# 
# **Pseudocode (Murao et al.) — $d$-dimensional, $M$ receivers:**
# 
# $$|\psi\rangle_P |\xi\rangle_{PA,C} = \left(\sum_{j=0}^{d-1}\alpha_j|j\rangle_P\right) \frac{1}{\sqrt{d}}\sum_{k=0}^{d-1}|k\rangle_{P_A}|\phi_k\rangle_C$$
# 
# $$= \frac{1}{\sqrt{d}}\sum_{m,n=0}^{d-1}|\Phi_{mn}\rangle_{P,P_A}\,U_{mn}|\phi\rangle_C$$
# 
# 1. CNOT($P \to P_A$) then $H(P)$ → $\frac{1}{\sqrt{d}}\sum_{m,n}|mn\rangle_{P,P_A}\,U_{mn}|\phi\rangle_C$
# 2. Alice measures $(P, P_A)$ → $(m,n)$ broadcast to **all receivers**
# 3. Each receiver $C_i$ applies $U_{mn}^{-1}$ → recovers $|\phi\rangle_C$
# 
# **Qubit implementation ($d=2$, $M=2$ receivers):**
# The optimal $1\to 2$ telecloning channel state is:
# $$|\xi\rangle = \frac{1}{\sqrt{6}}\bigl(2|0\rangle_{P_A}|00\rangle_{C_1C_2} + |1\rangle_{P_A}(|01\rangle+|10\rangle)_{C_1C_2}\bigr)$$
# 
# Each clone achieves the **optimal cloning fidelity** $F = 5/6 \approx 0.833$ (Bužek–Hillery bound).
# 
# **Workflow in this notebook:** setup and backend inspection, telecloning circuit construction, transpilation, ideal/noisy simulation, hardware-oriented sampling, input-state fidelity tests, noise sweeps, and finally a backend-by-backend comparison.
# 
# **Setup:** Same as standard notebook —
# - **FakeKawasaki / FakeSherbrooke / FakeKyiv** (127-qubit IBM fake backends)
# - **SamplerV2** (IBM Runtime primitive) for hardware-aware sampling
# - **Preset pass manager** (`generate_preset_pass_manager`) + **custom 4-stage pass manager**
# - **Layered noise model**: depolarizing gate errors + T1/T2 thermal relaxation + SPAM/readout errors
# - **AerSimulator density matrix** simulation for ideal vs noisy fidelity comparison
# - Per-channel fidelity bar chart + fidelity-vs-noise-strength sweep
# 

# ## 1 · Imports

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


# ## 2 · Backend & Chip Visualisation

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


# ## 3 · Build the Telecloning Circuit
# 
# **Qubit assignment (5 qubits, $1 \to 2$ telecloning):**
# 
# | Qubit | Label | Role |
# |---|---|---|
# | `q[0]` | $P$ | Alice's input qubit $|\psi\rangle = \alpha|0\rangle + \beta|1\rangle$ |
# | `q[1]` | $P_A$ | Alice's ancilla (port qubit) — measured alongside $P$ |
# | `q[2]` | $C_1$ | Receiver 1's clone qubit |
# | `q[3]` | $C_2$ | Receiver 2's clone qubit |
# | `q[4]` | $A$   | Alice's anticlone / ancilla qubit |
# 
# **Classical registers:**
# - `cm[0,1]` = Alice's measurements of $(P, P_A)$ → bits $(m, n)$ broadcast to all receivers
# 
# **Channel state preparation** $|\xi\rangle_{P_A,C_1,C_2,A}$:
# 
# The optimal $1\to 2$ telecloning channel state (Murao et al.) is:
# $$|\xi\rangle = \frac{1}{\sqrt{6}}\bigl(2|0\rangle_{P_A}|00\rangle_{C_1 C_2}|0\rangle_A + |0\rangle_{P_A}|11\rangle_{C_1 C_2}|0\rangle_A + |1\rangle_{P_A}(|01\rangle+|10\rangle)_{C_1 C_2}|1\rangle_A\bigr)$$
# 
# **Corrections $U_{mn}^{-1}$** (same table as standard teleportation applied to **each** receiver):
# 
# | $m$ | $n$ | $U_{mn}^{-1}$ |
# |---|---|---|
# | 0 | 0 | $I$ |
# | 0 | 1 | $X$ |
# | 1 | 0 | $Z$ |
# | 1 | 1 | $ZX$ (i.e. $iY$) |
# 
# **Optimal clone fidelity:** $F = 5/6 \approx 0.833$ (Bužek–Hillery bound for $1 \to 2$ qubit cloning).

# In[3]:


theta = np.pi / 4   # input |ψ⟩ polar angle
phi   = np.pi / 3   # input |ψ⟩ azimuthal angle


def create_telecloning_circuit(theta, phi):
    """
    Optimal 1→2 qubit telecloning (Murao et al.).

    Qubits:
      q[0] = P   : Alice input |ψ⟩
      q[1] = PA  : Alice port ancilla (measured with P)
      q[2] = C1  : Receiver 1 clone
      q[3] = C2  : Receiver 2 clone
      q[4] = A   : Alice anticlone ancilla

    Channel state |ξ⟩ on (PA, C1, C2, A) encodes the optimal
    cloning transformation. Prepared via the circuit below.
    """
    # ── Reference statevector (for fidelity) ──────────────────────────────
    qc_in = QuantumCircuit(1)
    qc_in.ry(theta, 0);  qc_in.rz(phi, 0)
    sv_input = Statevector.from_instruction(qc_in)

    print("Input statevector |ψ⟩:")
    print(np.round(sv_input.data, 4))

    # ── Registers ──────────────────────────────────────────────────────────
    # P, PA, C1, C2, A
    q  = QuantumRegister(5, 'q')
    cm = ClassicalRegister(2, 'cm')   # Alice meas: m=cm[0] (P), n=cm[1] (PA)
    qc = QuantumCircuit(q, cm)

    # ── Step 1: Prepare input |ψ⟩_P ───────────────────────────────────────
    qc.ry(theta, q[0])
    qc.rz(phi,   q[0])
    qc.barrier(label="Input |ψ⟩")

    # ── Step 2: Prepare channel state |ξ⟩ on (PA, C1, C2, A) ─────────────
    # Target:  (1/√6)( 2|0⟩_PA|00⟩_{C1C2}|0⟩_A
    #                 + |0⟩_PA|11⟩_{C1C2}|0⟩_A
    #                 + |1⟩_PA(|01⟩+|10⟩)_{C1C2}|1⟩_A )
    #
    # Preparation circuit (verified against Murao et al. Table I):
    #  1. Ry(2·arccos(√(2/3))) on PA  → √(2/3)|0⟩ + √(1/3)|1⟩
    #  2. CX(PA → A)
    #  3. H on C1
    #  4. CX(C1 → C2)     [Bell pair on C1,C2]
    #  5. Ry(π/4) on C1   [tilt toward |01⟩+|10⟩ sector]
    #  6. CX(PA → C1)
    #  7. CX(PA → C2)

    angle_PA = 2 * np.arccos(np.sqrt(2/3))   # ≈ 70.53°

    qc.ry(angle_PA, q[1])      # PA: √(2/3)|0⟩ + √(1/3)|1⟩
    qc.cx(q[1], q[4])          # entangle PA with anticlone A
    qc.h(q[2])                 # C1 into superposition
    qc.cx(q[2], q[3])          # Bell pair on C1, C2
    qc.ry(np.pi/4, q[2])       # tilt C1 to weight the |01⟩+|10⟩ sector
    qc.cx(q[1], q[2])          # correlate PA → C1
    qc.cx(q[1], q[3])          # correlate PA → C2
    qc.barrier(label="|ξ⟩ channel")

    # ── Step 3: CNOT(P → PA) then H(P)  [Bell measurement basis] ──────────
    qc.cx(q[0], q[1])
    qc.h(q[0])
    qc.barrier(label="CNOT(P→PA), H(P)")

    # ── Step 4: Alice measures (P, PA) → (m, n) broadcast to all receivers ─
    qc.measure(q[0], cm[0])    # m
    qc.measure(q[1], cm[1])    # n
    qc.barrier(label="Alice meas (m,n) → receivers")

    # ── Step 5: Each receiver applies U_mn^{-1} ────────────────────────────
    # Receiver 1 (C1 = q[2])
    with qc.if_test((cm[1], 1)):   # X if n=1
        qc.x(q[2])
    with qc.if_test((cm[0], 1)):   # Z if m=1
        qc.z(q[2])

    # Receiver 2 (C2 = q[3])  — same correction, independently applied
    with qc.if_test((cm[1], 1)):   # X if n=1
        qc.x(q[3])
    with qc.if_test((cm[0], 1)):   # Z if m=1
        qc.z(q[3])
    qc.barrier(label="U_mn^{-1} on C1, C2")

    return qc, sv_input


#  Create circuit
qc, sv_input = create_telecloning_circuit(theta, phi)

#  Circuit info
print(f"\nCircuit depth : {qc.depth()}")
print(f"Gate counts   : {dict(qc.count_ops())}")

#  Draw circuit
qc.draw('mpl', style="iqp", fold=-1)


# ## 4 · Preset Pass Manager (optimization_level=2)

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


# ## 5 · Custom 4-Stage Pass Manager
# 
# Stages: **SabreLayout → SabreSwap → BasisTranslator → Optimize1q**

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


# ## 6 · Depth Comparison

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
ax.set_title("Circuit Depth: Original vs Transpiled")
ax.set_ylim(0, max(depths) * 1.25)
plt.tight_layout()
plt.show()


# ## 7 · Layered Noise Model
# 
# Three independent noise layers:
# 
# | Layer | Type | Applies to |
# |---|---|---|
# | 1 | Depolarizing gate error | 1-qubit & 2-qubit gates |
# | 2 | T1/T2 thermal relaxation | All gates (via gate time) |
# | 3 | SPAM / Readout error | Measurement |

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
    readout = ReadoutError([[1 - p_meas/2, p_meas/2],
                            [p_meas,       1 - p_meas]])
    nm.add_all_qubit_readout_error(readout)

    return nm

# Default noise model
noise_model = build_noise_model()
print(noise_model)


# ## 8 · Ideal Simulation (Density Matrix, No Noise)
# 
# Both clones C1 and C2 are evaluated. Each achieves the **Bužek–Hillery optimal fidelity** $F = 5/6 \approx 0.833$, not perfect fidelity — this is the fundamental limit of quantum cloning.
# 
# Tracing:
# - Clone 1 fidelity: `partial_trace(dm, [0,1,3,4])` → keeps `q[2]` (C1)
# - Clone 2 fidelity: `partial_trace(dm, [0,1,2,4])` → keeps `q[3]` (C2)

# In[8]:


qc_ideal = qc.copy()
qc_ideal.save_density_matrix()

sim_ideal    = AerSimulator(method='density_matrix')
result_ideal = sim_ideal.run(qc_ideal, shots=4096).result()

dm_ideal = DensityMatrix(result_ideal.data()['density_matrix'])

# 5 qubits: q0(P), q1(PA), q2(C1), q3(C2), q4(A)
# Clone 1 on q2: trace out q0, q1, q3, q4
# Clone 2 on q3: trace out q0, q1, q2, q4
rho_c1_ideal = partial_trace(dm_ideal, [0, 1, 3, 4])
rho_c2_ideal = partial_trace(dm_ideal, [0, 1, 2, 4])

fid_c1_ideal = state_fidelity(rho_c1_ideal, sv_input)
fid_c2_ideal = state_fidelity(rho_c2_ideal, sv_input)

print(f"Ideal fidelity — Clone 1 (q[2]) : {fid_c1_ideal:.6f}")
print(f"Ideal fidelity — Clone 2 (q[3]) : {fid_c2_ideal:.6f}")
print(f"Bužek–Hillery optimal bound      : {5/6:.6f}")
print(f"Clone 1 gap from optimal         : {fid_c1_ideal - 5/6:+.6f}")
print(f"Clone 2 gap from optimal         : {fid_c2_ideal - 5/6:+.6f}")

counts_ideal = result_ideal.get_counts()
print("\nMeasurement counts (ideal):")
for outcome, cnt in sorted(counts_ideal.items()):
    print(f"  |{outcome}>  {cnt:5d}  ({100*cnt/4096:.1f}%)")


# ## 9 · Noisy Simulation (Density Matrix + Layered Noise)

# In[9]:


# Transpile to AerSimulator basis so noise model gates match
basis_aer  = ['u', 'cx', 'id', 'rz', 'sx', 'x', 'measure', 'if_else']
qc_noisy_t = transpile(qc, basis_gates=basis_aer, optimization_level=1)
qc_noisy_t.save_density_matrix()

sim_noisy    = AerSimulator(method='density_matrix', noise_model=noise_model)
result_noisy = sim_noisy.run(qc_noisy_t, shots=4096).result()

dm_noisy     = DensityMatrix(result_noisy.data()['density_matrix'])
rho_c1_noisy = partial_trace(dm_noisy, [0, 1, 3, 4])
rho_c2_noisy = partial_trace(dm_noisy, [0, 1, 2, 4])

fid_c1_noisy = state_fidelity(rho_c1_noisy, sv_input)
fid_c2_noisy = state_fidelity(rho_c2_noisy, sv_input)

print(f"Noisy fidelity — Clone 1 (q[2]) : {fid_c1_noisy:.6f}")
print(f"Noisy fidelity — Clone 2 (q[3]) : {fid_c2_noisy:.6f}")
print(f"Bužek–Hillery optimal bound      : {5/6:.6f}")
print(f"\nFidelity degradation — Clone 1  : {fid_c1_ideal - fid_c1_noisy:.6f}")
print(f"Fidelity degradation — Clone 2  : {fid_c2_ideal - fid_c2_noisy:.6f}")


# ## 10 · SamplerV2 on FakeKawasaki (IBM Runtime Primitive)

# In[10]:


# Build a version of the circuit with measure_all for SamplerV2
qc_sample = qc.copy()

# Transpile to ISA form required by SamplerV2
pm_sampler = generate_preset_pass_manager(backend=backend, optimization_level=2)
qc_isa = pm_sampler.run(qc_sample)

print("ISA circuit (hardware-native) drawn with IQP style:")
qc_isa.draw('mpl', style="iqp", fold=-1)


# In[11]:


# Run via SamplerV2 — professor's exact pattern
sampler    = SamplerV2(backend)
job        = sampler.run([qc_isa], shots=4096)
pub_result = job.result()[0]

# SamplerV2 PubResult → BitArray → counts
# Classical register cm: Alice's measurement bits (m, n)
counts_sampler = pub_result.data.cm.get_counts()

print("SamplerV2 — Alice (m,n) measurement counts (FakeKawasaki hardware noise):")
total = sum(counts_sampler.values())
for outcome, cnt in sorted(counts_sampler.items()):
    print(f"  cm=|{outcome}>  {cnt:5d}  ({100*cnt/total:.1f}%)")

print("\n(Each outcome broadcasts the same (m,n) correction to both receivers C1 and C2.)")


# ## 11 · Per-Channel Fidelity Comparison
# 
# Both clones are evaluated across five input states. The dashed line at $F = 5/6$ marks the **Bužek–Hillery optimal bound** — ideal telecloning should meet it exactly, while noisy telecloning falls below.

# In[12]:


qc_in = QuantumCircuit(1)
qc_in.ry(theta, 0);  qc_in.rz(phi, 0)

test_states = {
    '|0⟩'  : QuantumCircuit(1),
    '|1⟩'  : (lambda c: (c.x(0), c)[1])(QuantumCircuit(1)),
    '|+⟩'  : (lambda c: (c.h(0), c)[1])(QuantumCircuit(1)),
    '|-⟩'  : (lambda c: (c.h(0), c.z(0), c)[2])(QuantumCircuit(1)),
    '|ψ⟩'  : qc_in,
}


def run_telecloning_fidelity(input_qc, noisy=False):
    """Build & run telecloning for a given 1-qubit input state.
    Returns (fid_c1, fid_c2).
    """
    q_  = QuantumRegister(5, 'q')
    cm_ = ClassicalRegister(2, 'cm')
    qc_ = QuantumCircuit(q_, cm_)

    qc_.compose(input_qc, qubits=[0], inplace=True)
    qc_.barrier()

    # Channel state |ξ⟩
    angle_PA_ = 2 * np.arccos(np.sqrt(2/3))
    qc_.ry(angle_PA_, q_[1])
    qc_.cx(q_[1], q_[4])
    qc_.h(q_[2]);  qc_.cx(q_[2], q_[3])
    qc_.ry(np.pi/4, q_[2])
    qc_.cx(q_[1], q_[2]);  qc_.cx(q_[1], q_[3])
    qc_.barrier()

    # CNOT(P→PA), H(P)
    qc_.cx(q_[0], q_[1]);  qc_.h(q_[0])
    qc_.barrier()

    # Measure
    qc_.measure(q_[0], cm_[0]);  qc_.measure(q_[1], cm_[1])
    qc_.barrier()

    # Corrections — Clone 1
    with qc_.if_test((cm_[1], 1)): qc_.x(q_[2])
    with qc_.if_test((cm_[0], 1)): qc_.z(q_[2])
    # Corrections — Clone 2
    with qc_.if_test((cm_[1], 1)): qc_.x(q_[3])
    with qc_.if_test((cm_[0], 1)): qc_.z(q_[3])

    sv_ref = Statevector.from_instruction(input_qc)
    nm  = noise_model if noisy else None
    sim = AerSimulator(method='density_matrix', noise_model=nm)
    if noisy:
        basis_aer = ['u', 'cx', 'id', 'rz', 'sx', 'x', 'measure', 'if_else']
        qc_ = transpile(qc_, basis_gates=basis_aer, optimization_level=1)
    qc_.save_density_matrix()
    res  = sim.run(qc_, shots=2048).result()
    dm_  = DensityMatrix(res.data()['density_matrix'])
    rho1 = partial_trace(dm_, [0, 1, 3, 4])   # C1
    rho2 = partial_trace(dm_, [0, 1, 2, 4])   # C2
    return state_fidelity(rho1, sv_ref), state_fidelity(rho2, sv_ref)


fid_c1_id, fid_c2_id = {}, {}
fid_c1_no, fid_c2_no = {}, {}

for label, st in test_states.items():
    f1i, f2i = run_telecloning_fidelity(st, noisy=False)
    f1n, f2n = run_telecloning_fidelity(st, noisy=True)
    fid_c1_id[label] = f1i;  fid_c2_id[label] = f2i
    fid_c1_no[label] = f1n;  fid_c2_no[label] = f2n

labels_ch = list(fid_c1_id.keys())
x = np.arange(len(labels_ch));  width = 0.2

fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
for ax, fid_id, fid_no, title in zip(
        axes,
        [fid_c1_id, fid_c2_id],
        [fid_c1_no, fid_c2_no],
        ["Clone 1 (C1, q[2])", "Clone 2 (C2, q[3])"]):
    vals_id = [fid_id[k] for k in labels_ch]
    vals_no = [fid_no[k] for k in labels_ch]
    b1 = ax.bar(x - width/2, vals_id, width, label='Ideal',
                color='#5B9BD5', edgecolor='black', lw=0.6)
    b2 = ax.bar(x + width/2, vals_no, width, label='Noisy',
                color='#ED7D31', edgecolor='black', lw=0.6)
    for bar in [*b1, *b2]:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.004,
                f"{bar.get_height():.3f}", ha='center', va='bottom', fontsize=7)
    ax.axhline(5/6, color='green', linestyle='--', lw=1.5,
               label='Bužek–Hillery bound (5/6)')
    ax.axhline(2/3, color='red', linestyle=':', lw=1.2,
               label='Classical limit (2/3)')
    ax.set_xticks(x);  ax.set_xticklabels(labels_ch)
    ax.set_ylabel("Clone Fidelity");  ax.set_ylim(0.4, 1.05)
    ax.set_title(title)
    ax.legend(fontsize=8)

fig.suptitle("Telecloning — Per-Input Clone Fidelity: Ideal vs Noisy", fontsize=12)
plt.tight_layout();  plt.show()


# ## 12 · Fidelity vs Noise Strength Sweep
# 
# Both clones are swept together. The **Bužek–Hillery bound** ($F=5/6$) and **classical limit** ($F=2/3$) are marked. Noise degrades both clones symmetrically.

# In[13]:


p2q_sweep  = np.linspace(0.001, 0.15, 20)
fids_c1_sweep = []
fids_c2_sweep = []
basis_aer  = ['u', 'cx', 'id', 'rz', 'sx', 'x', 'measure', 'if_else']

for p2q in p2q_sweep:
    nm_sw = build_noise_model(p1q=p2q/10, p2q=p2q, p_meas=p2q/5)
    q_  = QuantumRegister(5, 'q')
    cm_ = ClassicalRegister(2, 'cm')
    qc_ = QuantumCircuit(q_, cm_)

    # Input state
    qc_.ry(theta, q_[0]);  qc_.rz(phi, q_[0]);  qc_.barrier()

    # Channel state
    angle_PA_ = 2 * np.arccos(np.sqrt(2/3))
    qc_.ry(angle_PA_, q_[1])
    qc_.cx(q_[1], q_[4])
    qc_.h(q_[2]);  qc_.cx(q_[2], q_[3])
    qc_.ry(np.pi/4, q_[2])
    qc_.cx(q_[1], q_[2]);  qc_.cx(q_[1], q_[3])
    qc_.barrier()

    # Bell basis
    qc_.cx(q_[0], q_[1]);  qc_.h(q_[0]);  qc_.barrier()

    # Measure
    qc_.measure(q_[0], cm_[0]);  qc_.measure(q_[1], cm_[1]);  qc_.barrier()

    # Corrections
    with qc_.if_test((cm_[1], 1)): qc_.x(q_[2])
    with qc_.if_test((cm_[0], 1)): qc_.z(q_[2])
    with qc_.if_test((cm_[1], 1)): qc_.x(q_[3])
    with qc_.if_test((cm_[0], 1)): qc_.z(q_[3])

    qc_t = transpile(qc_, basis_gates=basis_aer, optimization_level=1)
    qc_t.save_density_matrix()
    sim_ = AerSimulator(method='density_matrix', noise_model=nm_sw)
    res_ = sim_.run(qc_t, shots=2048).result()
    dm_  = DensityMatrix(res_.data()['density_matrix'])
    fids_c1_sweep.append(state_fidelity(partial_trace(dm_, [0,1,3,4]), sv_input))
    fids_c2_sweep.append(state_fidelity(partial_trace(dm_, [0,1,2,4]), sv_input))

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(p2q_sweep, fids_c1_sweep, 'o-', color='#5B9BD5', lw=2, ms=5, label='Clone 1 (C1)')
ax.plot(p2q_sweep, fids_c2_sweep, 's--', color='#ED7D31', lw=2, ms=5, label='Clone 2 (C2)')
ax.axhline(5/6, color='green', linestyle='--', lw=1.5, label='Bužek–Hillery bound (5/6)')
ax.axhline(2/3, color='red', linestyle=':', lw=1.5, label='Classical limit (2/3)')
ax.axhline(1.0, color='grey', linestyle=':', lw=1)
ax.fill_between(p2q_sweep, fids_c1_sweep, 2/3,
                where=[f > 2/3 for f in fids_c1_sweep],
                alpha=0.10, color='#5B9BD5')
ax.fill_between(p2q_sweep, fids_c2_sweep, 2/3,
                where=[f > 2/3 for f in fids_c2_sweep],
                alpha=0.10, color='#ED7D31')
ax.set_xlabel("2-qubit Gate Depolarizing Error (p₂q)")
ax.set_ylabel("Clone Fidelity")
ax.set_title("Telecloning — Clone Fidelity vs Noise Strength")
ax.legend();  ax.set_ylim(0.4, 1.05)
ax.grid(True, alpha=0.3)
plt.tight_layout();  plt.show()


# ## 13 · Summary

# ### Backend Sweep Helper Functions
# 
# The final backend comparison uses three small helper functions so the sweep stays readable:
# 
# - `get_qubit_layout(...)` picks a simple five-qubit layout when a larger backend offers many possible mappings.
# - `transpile_with_layout(...)` wraps `transpile(...)` so an optional initial layout can be passed consistently.
# - `run_simulation(...)` executes the compiled circuit and returns both raw counts and normalized probabilities.
# 
# Keeping these helpers here makes it clearer that they mainly support the backend-by-backend summary.
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

    # 🔁 Create circuit
    qc, sv_input = create_telecloning_circuit(theta, phi)

    # 🔧 choose layout (manual OR automatic)
    layout = None
    if backend.num_qubits >= 20:
        layout = get_qubit_layout(data["coupling_map"], 5)   # 5 qubits

    # Use layout-enabled transpile
    qc_t = transpile_with_layout(qc, backend, layout)

    # ▶ Run simulation
    counts, probs = run_simulation(qc_t)

    print("\nMeasurement outcomes:")
    for outcome, p_out in probs.items():
        print(f"  {outcome} → {p_out:.3f}")

    # Fidelity via AerSimulator density matrix with backend noise model
    nm_backend = NoiseModel.from_backend(backend)
    basis_aer  = ['u', 'cx', 'id', 'rz', 'sx', 'x', 'measure', 'if_else']
    qc_dm = transpile(qc, basis_gates=basis_aer, optimization_level=1)
    qc_dm.save_density_matrix()
    sim_dm = AerSimulator(method='density_matrix', noise_model=nm_backend)
    res_dm = sim_dm.run(qc_dm, shots=2048).result()
    dm_b   = DensityMatrix(res_dm.data()['density_matrix'])

    fid_c1 = float(state_fidelity(partial_trace(dm_b, [0,1,3,4]), sv_input))
    fid_c2 = float(state_fidelity(partial_trace(dm_b, [0,1,2,4]), sv_input))
    fidelity = (fid_c1 + fid_c2) / 2
    error    = 1 - fidelity

    fidelity_results[name] = {"c1": fid_c1, "c2": fid_c2, "avg": fidelity}
    error_rates.append(error)
    all_results.append({
        "backend" : name,
        "counts"  : counts,
        "probs"   : probs,
        "fid_c1"  : fid_c1,
        "fid_c2"  : fid_c2,
        "fidelity": fidelity,
        "error"   : error
    })
    print(f"Clone 1 fidelity : {fid_c1:.4f}")
    print(f"Clone 2 fidelity : {fid_c2:.4f}")
    print(f"Average fidelity : {fidelity:.4f}  (BH bound: {5/6:.4f})")
    print(f"Error            : {error:.4f}")


# ### Backend Sweep Outputs
# 
# After the helper definitions and backend loop run, the notebook finishes with three outputs:
# 
# - a grouped bar chart comparing clone fidelities across backends,
# - a concise printed summary, and
# - a detailed dump of the sampled counts and fidelity values.
# 

# In[18]:


names    = list(fidelity_results.keys())
fids_c1  = [fidelity_results[n]["c1"]  for n in names]
fids_c2  = [fidelity_results[n]["c2"]  for n in names]
fids_avg = [fidelity_results[n]["avg"] for n in names]

x = np.arange(len(names));  width = 0.22
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(x - width, fids_c1,  width, label='Clone 1 (C1)', color='#5B9BD5', edgecolor='black')
ax.bar(x,         fids_c2,  width, label='Clone 2 (C2)', color='#ED7D31', edgecolor='black')
ax.bar(x + width, fids_avg, width, label='Average',       color='#70AD47', edgecolor='black')
ax.axhline(5/6, color='green', linestyle='--', lw=1.5, label='Bužek–Hillery bound (5/6)')
ax.axhline(2/3, color='red',   linestyle=':',  lw=1.2, label='Classical limit (2/3)')
ax.set_xticks(x);  ax.set_xticklabels(names)
ax.set_title("Telecloning — Clone Fidelity across Backends")
ax.set_xlabel("Backend");  ax.set_ylabel("Clone Fidelity")
ax.set_ylim(0, 1.1);  ax.legend(fontsize=9)
plt.tight_layout();  plt.show()


# In[19]:


print("=" * 58)
print("  Quantum Broadcasting / Telecloning (Murao et al.)")
print("  1 → 2 Optimal Qubit Telecloning")
print("=" * 58)
print(f"  Bužek–Hillery optimal clone fidelity : {5/6:.6f}")
print(f"  Classical cloning limit              : {2/3:.6f}")
print("=" * 58)

for name in fidelity_results:
    r = fidelity_results[name]
    print(f"\nBackend : {name}")
    print(f"Clone 1 fidelity : {r['c1']:.6f}")
    print(f"Clone 2 fidelity : {r['c2']:.6f}")
    print(f"Average fidelity : {r['avg']:.6f}")
    print(f"Error            : {1 - r['avg']:.6f}")
    if r["avg"] > 2/3:
        print("✓ Beats classical cloning limit")
    if r["avg"] >= 5/6 - 0.02:
        print("✓ Near Bužek–Hillery optimal bound")

print("=" * 58)


# In[20]:


print("\nDetailed Results:")
for res in all_results:
    print(f"\nBackend: {res['backend']}")
    print(f"Counts: {res['counts']}")
    print(f"Clone 1 fidelity: {res['fid_c1']:.4f}")
    print(f"Clone 2 fidelity: {res['fid_c2']:.4f}")

