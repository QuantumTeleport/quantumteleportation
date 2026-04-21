#!/usr/bin/env python
# coding: utf-8

# # Probabilistic Quantum Teleportation — Agrawal & Pati (FakeKawasaki + Noise Simulation)
# 
# Probabilistic teleportation uses a **partially entangled resource state** $|\Phi_{\text{res}}\rangle_{12}$ instead of a maximally entangled Bell pair. Teleportation succeeds with probability $p < 1$, but when it succeeds the fidelity is perfect.
# 
# **Pseudocode (Agrawal & Pati):**
# 
# $$|\psi\rangle_a |\Phi_{\text{res}}\rangle_{12} = (\alpha|0\rangle_a + \beta|1\rangle_a)\,\frac{1}{\sqrt{1+|n|^2}}(|00\rangle_{12} + n|11\rangle_{12})$$
# 
# 1. Expand over generalised Bell basis $\{|\chi_{ij}\rangle\}$ → $\frac{1}{2}\sum_{i,j}|\chi_{ij}\rangle_{a1}\,U_{ij}|\psi\rangle_2$
# 2. Apply $R_n$ to (a,1) → $\frac{1}{2}\sum_{i,j}|ij\rangle_{a1}\,U_{ij}|\psi\rangle_2$
# 3. Alice measures (a,1) in std. basis, sends $(i,j)$ to Bob
# 4. Bob applies $U_{ij}^{-1}$ — succeeds **with probability $p < 1$**
# 
# **Key physics:** The resource state has entanglement parameter $n \in (0,1]$. When $n=1$ this reduces to standard teleportation ($p=1$). For $n<1$, the $R_n$ disentangling unitary is non-unitary on the full space — implemented here via post-selection on the ancilla.
# 
# **Workflow in this notebook:** setup and backend inspection, protocol construction, transpilation, ideal/noisy simulation, hardware-oriented sampling, input-state fidelity tests, parameter sweeps, and finally a backend-by-backend comparison.
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

# In[21]:


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

# In[22]:


import matplotlib.pyplot as plt

for name, data in backend_data.items():
    backend = data["backend"]

    print(f"\n{name.upper()}")
    print(f"Qubits: {data['num_qubits']}")
    print(f"Basis gates: {data['basis_gates']}")

    plt.figure()  # ⭐ THIS FIXES IT
    fig = plot_gate_map(backend)
    display(fig)


# ## 3 · Build the Probabilistic Teleportation Circuit
# 
# **Qubit assignment (4 qubits total):**
# 
# | Qubit | Label | Role |
# |---|---|---|
# | `q[0]` | a | Alice's input $|\psi\rangle = \alpha|0\rangle + \beta|1\rangle$ |
# | `q[1]` | 1 | Alice's half of partially entangled resource |
# | `q[2]` | 2 | Bob's half of resource (receives $|\psi\rangle$) |
# | `q[3]` | anc | Ancilla — post-selected on $|0\rangle$ to flag success |
# 
# **Classical registers:**
# - `ca[0,1]` = Alice's measurements of (a, 1) → bits (i, j)
# - `cs[0]`   = Ancilla measurement → success flag (0 = success)
# 
# **Resource state** $|\Phi_{\text{res}}\rangle_{12} = \frac{1}{\sqrt{1+|n|^2}}(|00\rangle + n|11\rangle)$
# prepared using a biased rotation angle $\theta_n = 2\arctan(n)$.
# 
# **$R_n$ disentangling unitary:** Implemented as a controlled-$R_y$ on the ancilla, followed by post-selection on ancilla $= |0\rangle$ to enforce the probabilistic success condition.
# 
# **Success probability:** $p = \frac{2|n|^2}{1+|n|^2}$ — rises from 0 to 1 as $n: 0 \to 1$.

# In[23]:


# ── Parameters ───────────────────────────────────────────────────────────────
theta = np.pi / 4   # input state |ψ⟩ polar angle
phi   = np.pi / 3   # input state |ψ⟩ azimuthal angle
n     = 0.7         # entanglement parameter  0 < n ≤ 1
                    # n=1 → standard teleportation (p=1)
                    # n<1 → probabilistic (p<1, but perfect fidelity on success)

# Theoretical success probability
p_success_theory = 2 * n**2 / (1 + n**2)
print(f"Entanglement parameter n  = {n}")
print(f"Theoretical success prob  = {p_success_theory:.4f}")


def create_probabilistic_teleportation_circuit(theta, phi, n):
    """
    Probabilistic teleportation (Agrawal & Pati).

    Resource state: |Φ_res⟩_12 = (1/√(1+n²))(|00⟩ + n|11⟩)

    R_n disentangling: implemented via a controlled-Ry on an ancilla qubit
    followed by post-selection on ancilla=|0⟩ (success branch).

    Bob's corrections U_ij^{-1} = {I, X, Z, iY} for (i,j) in {00,01,10,11}
    — same as standard teleportation because the resource state expansion
      over the generalised Bell basis recovers the standard correction table.
    """
    # ── Reference statevector ──────────────────────────────────────────────
    qc_in = QuantumCircuit(1)
    qc_in.ry(theta, 0);  qc_in.rz(phi, 0)
    sv_input = Statevector.from_instruction(qc_in)

    print("Input statevector |ψ⟩:")
    print(np.round(sv_input.data, 4))

    # ── Registers ─────────────────────────────────────────────────────────
    q  = QuantumRegister(4, 'q')    # q0=a(input), q1=res1, q2=Bob, q3=ancilla
    ca = ClassicalRegister(2, 'ca') # Alice: i=ca[0], j=ca[1]
    cs = ClassicalRegister(1, 'cs') # success flag: cs[0]=0 → success
    qc = QuantumCircuit(q, ca, cs)

    # ── Step 1: Prepare input state |ψ⟩_a ────────────────────────────────
    qc.ry(theta, q[0])
    qc.rz(phi,   q[0])
    qc.barrier(label="Input |ψ⟩")

    # ── Step 2: Prepare partially entangled resource |Φ_res⟩_12 ──────────
    # |Φ_res⟩ = (1/√(1+n²))(|00⟩ + n|11⟩)
    # Angle: θ_n = 2·arctan(n)  so  cos(θ_n/2)=1/√(1+n²), sin(θ_n/2)=n/√(1+n²)
    theta_n = 2 * np.arctan(n)
    qc.ry(theta_n, q[1])       # rotate q1: α|0⟩ + β|1⟩ with β/α = n
    qc.cx(q[1], q[2])          # entangle → partially entangled resource
    qc.barrier(label="|Φ_res⟩")

    # ── Step 3: Alice CNOT(a,1) then H(a) — Bell measurement basis ───────
    qc.cx(q[0], q[1])
    qc.h(q[0])
    qc.barrier(label="Alice ops")

    # ── Step 4: R_n disentangling — controlled-Ry on ancilla ─────────────
    # R_n maps the generalised Bell states |χ_ij⟩ → |ij⟩ (success branch)
    # Implemented as Ry(2·arcsin(n)) on ancilla, controlled on q[1].
    # Post-selection on ancilla=|0⟩ enforces the success condition.
    # The rotation angle encodes the normalisation 1/√(1+n²).
    rn_angle = 2 * np.arcsin(n / np.sqrt(1 + n**2))
    qc.cry(rn_angle, q[1], q[3])   # controlled-Ry(R_n) on ancilla
    qc.barrier(label="R_n")

    # ── Step 5: Alice measures (a,1); ancilla measured for success ────────
    qc.measure(q[0], ca[0])   # i
    qc.measure(q[1], ca[1])   # j
    qc.measure(q[3], cs[0])   # 0 = success, 1 = failure
    qc.barrier(label="Alice meas + success flag")

    # ── Step 6: Bob applies U_ij^{-1} on q[2] (conditioned on success) ───
    # Only meaningful when cs[0]==0 (ancilla = |0⟩, success branch)
    # U_ij table: (0,0)→I, (0,1)→X, (1,0)→Z, (1,1)→iY=XZ
    with qc.if_test((cs[0], 0)):          # proceed only on success
        with qc.if_test((ca[1], 1)):      # X if j=1
            qc.x(q[2])
        with qc.if_test((ca[0], 1)):      # Z if i=1
            qc.z(q[2])
    qc.barrier(label="Bob U_ij^{-1}")

    return qc, sv_input


#  Create circuit
qc, sv_input = create_probabilistic_teleportation_circuit(theta, phi, n)

#  Circuit info
print(f"\nCircuit depth : {qc.depth()}")
print(f"Gate counts   : {dict(qc.count_ops())}")

#  Draw circuit
qc.draw('mpl', style="iqp", fold=-1)


# ## 4 · Preset Pass Manager (optimization_level=2)

# In[24]:


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

# In[25]:


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

# In[26]:


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

# In[27]:


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


# ## 8 · Ideal Simulation (Density Matrix, No Noise)
# 
# **Post-selection:** We keep only shots where `cs[0] == 0` (ancilla measured $|0\rangle$), corresponding to the success branch. The empirical success rate is compared to the theoretical value $p = \frac{2n^2}{1+n^2}$.

# In[28]:


qc_ideal = qc.copy()
qc_ideal.save_density_matrix()

sim_ideal = AerSimulator(method='density_matrix')

# 🔥 KEY FIX
qc_ideal = transpile(qc_ideal, sim_ideal)

result_ideal = sim_ideal.run(qc_ideal, shots=4096).result()

dm_ideal = DensityMatrix(result_ideal.data()['density_matrix'])

# 4 qubits: q0(a), q1(res1), q2(Bob), q3(ancilla)
# Bob's state on q2 — trace out q0, q1, q3
rho_bob_ideal = partial_trace(dm_ideal, [0, 1, 3])

fidelity_ideal = state_fidelity(rho_bob_ideal, sv_input)
print(f"Ideal fidelity (Bob q[2] vs |ψ⟩): {fidelity_ideal:.6f}")

# ── Post-selection on success (cs=0) ─────────────────────────────────────────
counts_ideal = result_ideal.get_counts()
total_shots  = sum(counts_ideal.values())

# Count format: "cs ca" e.g. "0 01" → cs=0 (success), ca=01
success_counts = {k: v for k, v in counts_ideal.items()
                  if k.split()[0] == '0'}   # cs register is leftmost
success_shots  = sum(success_counts.values())
p_empirical    = success_shots / total_shots

print(f"\nTheoretical success prob : {p_success_theory:.4f}")
print(f"Empirical  success prob  : {p_empirical:.4f}")
print(f"\nSuccess-branch counts (cs=0):")
for outcome, cnt in sorted(success_counts.items()):
    print(f"  |{outcome}>  {cnt:5d}  ({100*cnt/total_shots:.1f}%)")

print(f"\nAll measurement counts (ideal):")
for outcome, cnt in sorted(counts_ideal.items()):
    print(f"  |{outcome}>  {cnt:5d}  ({100*cnt/total_shots:.1f}%)")


# ## 9 · Noisy Simulation (Density Matrix + Layered Noise)

# In[29]:


# Transpile to AerSimulator basis so noise model gates match
basis_aer  = ['u', 'cx', 'id', 'rz', 'sx', 'x', 'measure', 'if_else']
qc_noisy_t = transpile(qc, basis_gates=basis_aer, optimization_level=1)
qc_noisy_t.save_density_matrix()

sim_noisy    = AerSimulator(method='density_matrix', noise_model=noise_model)
result_noisy = sim_noisy.run(qc_noisy_t, shots=4096).result()

dm_noisy      = DensityMatrix(result_noisy.data()['density_matrix'])
rho_bob_noisy = partial_trace(dm_noisy, [0, 1, 3])

fidelity_noisy = state_fidelity(rho_bob_noisy, sv_input)
print(f"Noisy fidelity (Bob q[2] vs |ψ⟩): {fidelity_noisy:.6f}")
print(f"Ideal fidelity                   : {fidelity_ideal:.6f}")
print(f"Fidelity degradation             : {fidelity_ideal - fidelity_noisy:.6f}")

# Noisy empirical success rate
counts_noisy   = result_noisy.get_counts()
total_noisy    = sum(counts_noisy.values())
success_noisy  = sum(v for k, v in counts_noisy.items() if k.split()[0] == '0')
p_noisy        = success_noisy / total_noisy
print(f"\nNoisy empirical success prob : {p_noisy:.4f}")
print(f"Ideal empirical success prob : {p_empirical:.4f}")


# ## 10 · SamplerV2 on FakeKawasaki (IBM Runtime Primitive)

# In[30]:


# Build a version of the circuit with measure_all for SamplerV2
qc_sample = qc.copy()

# Transpile to ISA form required by SamplerV2
pm_sampler = generate_preset_pass_manager(backend=backend, optimization_level=2)
qc_isa = pm_sampler.run(qc_sample)

print("ISA circuit (hardware-native) drawn with IQP style:")
qc_isa.draw('mpl', style="iqp", fold=-1)


# In[31]:


# Run via SamplerV2 — professor's exact pattern
sampler    = SamplerV2(backend)
job        = sampler.run([qc_isa], shots=4096)
pub_result = job.result()[0]

# SamplerV2 PubResult → BitArray → counts
# Two classical registers: ca (Alice bits i,j) and cs (success flag)
counts_ca = pub_result.data.ca.get_counts()   # Alice (i,j)
counts_cs = pub_result.data.cs.get_counts()   # success flag

print("SamplerV2 — Alice measurement counts (FakeKawasaki hardware noise):")
total_ca = sum(counts_ca.values())
for outcome, cnt in sorted(counts_ca.items()):
    print(f"  ca=|{outcome}>  {cnt:5d}  ({100*cnt/total_ca:.1f}%)")

print("\nSamplerV2 — Ancilla success flag (0=success, 1=failure):")
total_cs = sum(counts_cs.values())
for outcome, cnt in sorted(counts_cs.items()):
    label = "SUCCESS" if outcome == "0" else "failure"
    print(f"  cs=|{outcome}> ({label})  {cnt:5d}  ({100*cnt/total_cs:.1f}%)")

p_sampler = counts_cs.get("0", 0) / total_cs
print(f"\nSamplerV2 empirical success prob : {p_sampler:.4f}")
print(f"Theoretical success prob         : {p_success_theory:.4f}")


# ## 11 · Per-Channel Fidelity Comparison
# 
# Fidelity is evaluated on the **full** (un-post-selected) density matrix — this shows the degradation when treating the mixed success+failure output as the received state, which is the relevant metric for comparing to a noisy channel.

# In[32]:


qc_in = QuantumCircuit(1)
qc_in.ry(theta, 0);  qc_in.rz(phi, 0)

# Fidelity across multiple input states (X, Y, Z, H, T axes)
test_states = {
    '|0⟩'  : QuantumCircuit(1),
    '|1⟩'  : (lambda c: (c.x(0), c)[1])(QuantumCircuit(1)),
    '|+⟩'  : (lambda c: (c.h(0), c)[1])(QuantumCircuit(1)),
    '|-⟩'  : (lambda c: (c.h(0), c.z(0), c)[2])(QuantumCircuit(1)),
    '|ψ⟩'  : qc_in,
}

def run_prob_teleport_fidelity(input_qc, n_param, noisy=False):
    """Build & run probabilistic teleportation for a given input state."""
    q_  = QuantumRegister(4, 'q')
    ca_ = ClassicalRegister(2, 'ca')
    cs_ = ClassicalRegister(1, 'cs')
    qc_ = QuantumCircuit(q_, ca_, cs_)

    qc_.compose(input_qc, qubits=[0], inplace=True)
    qc_.barrier()

    # Resource state
    theta_n_ = 2 * np.arctan(n_param)
    qc_.ry(theta_n_, q_[1]);  qc_.cx(q_[1], q_[2]);  qc_.barrier()

    # Alice ops
    qc_.cx(q_[0], q_[1]);  qc_.h(q_[0]);  qc_.barrier()

    # R_n on ancilla
    rn_ = 2 * np.arcsin(n_param / np.sqrt(1 + n_param**2))
    qc_.cry(rn_, q_[1], q_[3]);  qc_.barrier()

    # Measure
    qc_.measure(q_[0], ca_[0]);  qc_.measure(q_[1], ca_[1])
    qc_.measure(q_[3], cs_[0]);  qc_.barrier()

    # Bob corrections (on success)
    with qc_.if_test((cs_[0], 0)):
        with qc_.if_test((ca_[1], 1)): qc_.x(q_[2])
        with qc_.if_test((ca_[0], 1)): qc_.z(q_[2])

    sv_ref = Statevector.from_instruction(input_qc)
    nm = noise_model if noisy else None
    sim = AerSimulator(method='density_matrix', noise_model=nm)
    # ✅ ALWAYS transpile (this fixes cry issue)
    qc_ = transpile(qc_, sim, optimization_level=1)
    qc_.save_density_matrix()
    res = sim.run(qc_, shots=2048).result()
    dm_ = DensityMatrix(res.data()['density_matrix'])
    rho = partial_trace(dm_, [0, 1, 3])   # Bob q2
    return state_fidelity(rho, sv_ref)


fid_ideal_per = {k: run_prob_teleport_fidelity(v, n, noisy=False) for k, v in test_states.items()}
fid_noisy_per = {k: run_prob_teleport_fidelity(v, n, noisy=True)  for k, v in test_states.items()}

labels_ch = list(fid_ideal_per.keys())
vals_id   = [fid_ideal_per[k] for k in labels_ch]
vals_no   = [fid_noisy_per[k] for k in labels_ch]

x = np.arange(len(labels_ch));  width = 0.35
fig, ax = plt.subplots(figsize=(9, 5))
b1 = ax.bar(x - width/2, vals_id, width, label='Ideal', color='#5B9BD5', edgecolor='black', lw=0.6)
b2 = ax.bar(x + width/2, vals_no, width, label='Noisy', color='#ED7D31', edgecolor='black', lw=0.6)
for bar in [*b1, *b2]:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
            f"{bar.get_height():.3f}", ha='center', va='bottom', fontsize=8)
ax.set_xticks(x);  ax.set_xticklabels(labels_ch)
ax.set_ylabel("State Fidelity");  ax.set_ylim(0.4, 1.1)
ax.set_title(f"Probabilistic Teleportation — Per-Channel Fidelity (n={n}, p≈{p_success_theory:.2f}): Ideal vs Noisy")
ax.legend();  ax.axhline(1.0, color='grey', linestyle='--', lw=0.8)
ax.axhline(2/3, color='red', linestyle='--', lw=1, label='Classical limit')
plt.tight_layout();  plt.show()


# ## 12 · Fidelity vs Noise Strength Sweep
# 
# Two sweeps are shown together: **fidelity vs noise** (fixed $n$) and **success probability vs $n$** (fixed noise).

# In[33]:


# ── Panel A: Fidelity vs noise strength (fixed n) ────────────────────────────
p2q_sweep    = np.linspace(0.001, 0.15, 20)
fids_sweep   = []
basis_aer    = ['u', 'cx', 'id', 'rz', 'sx', 'x', 'measure', 'if_else']

for p2q in p2q_sweep:
    nm_sw = build_noise_model(p1q=p2q/10, p2q=p2q, p_meas=p2q/5)
    q_  = QuantumRegister(4, 'q')
    ca_ = ClassicalRegister(2, 'ca')
    cs_ = ClassicalRegister(1, 'cs')
    qc_ = QuantumCircuit(q_, ca_, cs_)
    qc_.ry(theta, q_[0]);  qc_.rz(phi, q_[0]);  qc_.barrier()
    theta_n_ = 2 * np.arctan(n)
    qc_.ry(theta_n_, q_[1]);  qc_.cx(q_[1], q_[2]);  qc_.barrier()
    qc_.cx(q_[0], q_[1]);  qc_.h(q_[0]);  qc_.barrier()
    rn_ = 2 * np.arcsin(n / np.sqrt(1 + n**2))
    qc_.cry(rn_, q_[1], q_[3]);  qc_.barrier()
    qc_.measure(q_[0], ca_[0]);  qc_.measure(q_[1], ca_[1])
    qc_.measure(q_[3], cs_[0]);  qc_.barrier()
    with qc_.if_test((cs_[0], 0)):
        with qc_.if_test((ca_[1], 1)): qc_.x(q_[2])
        with qc_.if_test((ca_[0], 1)): qc_.z(q_[2])
    qc_t = transpile(qc_, basis_gates=basis_aer, optimization_level=1)
    qc_t.save_density_matrix()
    sim_ = AerSimulator(method='density_matrix', noise_model=nm_sw)
    res_ = sim_.run(qc_t, shots=2048).result()
    dm_  = DensityMatrix(res_.data()['density_matrix'])
    rho_ = partial_trace(dm_, [0, 1, 3])
    fids_sweep.append(state_fidelity(rho_, sv_input))

# ── Panel B: Success probability vs n (ideal, no noise) ──────────────────────
n_sweep     = np.linspace(0.01, 1.0, 40)
p_theory    = 2 * n_sweep**2 / (1 + n_sweep**2)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# Panel A
ax1.plot(p2q_sweep, fids_sweep, 'o-', color='#5B9BD5', lw=2, ms=5,
         label=f'Prob. teleportation (n={n})')
ax1.axhline(2/3, color='#ED7D31', linestyle='--', lw=1.5, label='Classical limit (2/3)')
ax1.axhline(1.0, color='grey', linestyle=':', lw=1)
ax1.fill_between(p2q_sweep, fids_sweep, 2/3,
                 where=[f > 2/3 for f in fids_sweep],
                 alpha=0.15, color='#5B9BD5', label='Quantum advantage region')
ax1.set_xlabel("2-qubit Gate Depolarizing Error (p₂q)")
ax1.set_ylabel("State Fidelity F(ρ_Bob, |ψ⟩)")
ax1.set_title(f"Fidelity vs Noise Strength (n={n})")
ax1.legend();  ax1.set_ylim(0.4, 1.05)
ax1.grid(True, alpha=0.3)

# Panel B
ax2.plot(n_sweep, p_theory, '-', color='#70AD47', lw=2.5,
         label=r'$p = \frac{2n^2}{1+n^2}$')
ax2.axvline(n, color='#5B9BD5', linestyle='--', lw=1.5, label=f'Current n={n}')
ax2.axhline(p_success_theory, color='#ED7D31', linestyle=':', lw=1.5,
            label=f'p≈{p_success_theory:.3f}')
ax2.set_xlabel("Entanglement parameter n")
ax2.set_ylabel("Success probability p")
ax2.set_title("Success Probability vs Entanglement Parameter")
ax2.legend();  ax2.set_ylim(0, 1.05)
ax2.grid(True, alpha=0.3)

plt.tight_layout();  plt.show()


# ## 13 · Summary

# ### Backend Sweep Helper Functions
# 
# The final backend comparison uses three small helper functions so the sweep stays readable:
# 
# - `get_qubit_layout(...)` picks a simple four-qubit layout when a larger backend offers many possible mappings.
# - `transpile_with_layout(...)` wraps `transpile(...)` so an optional initial layout can be passed consistently.
# - `run_simulation(...)` executes the compiled circuit and returns both raw counts and normalized probabilities.
# 
# Keeping these helpers here makes it clearer that they are mainly for the final backend-by-backend summary.
# 

# In[34]:


def get_qubit_layout(coupling_map, num_qubits, manual_map=None):
    if manual_map:
        return manual_map

    edges = coupling_map.get_edges()

    for edge in edges:
        if len(edge) >= num_qubits:
            return list(edge[:num_qubits])

    return list(range(num_qubits))


# In[35]:


def transpile_with_layout(qc, backend, layout=None):
    return transpile(
        qc,
        backend=backend,
        initial_layout=layout,
        optimization_level=2
    )


# In[36]:


def run_simulation(qc):
    sim = AerSimulator()
    result = sim.run(qc, shots=1024).result()

    counts = result.get_counts()
    probs = {k: v / 1024 for k, v in counts.items()}

    return counts, probs


# In[37]:


fidelity_results = {}
error_rates = []
all_results = []

for name, data in backend_data.items():
    backend = data["backend"]

    print(f"\n===== {name.upper()} =====")

    # 🔁 Create circuit
    qc, sv_input = create_probabilistic_teleportation_circuit(theta, phi, n)

    # 🔧 choose layout (manual OR automatic)
    layout = None
    if backend.num_qubits >= 20:
        layout = get_qubit_layout(data["coupling_map"], 4)   # 4 qubits

    # Use layout-enabled transpile
    qc_t = transpile_with_layout(qc, backend, layout)

    # ▶ Run simulation
    counts, probs = run_simulation(qc_t)

    print("\nMeasurement outcomes:")
    for outcome, p_out in probs.items():
        print(f"  {outcome} → {p_out:.3f}")

    print("Counts:", counts)

    # Fidelity via AerSimulator density matrix with backend noise model
    nm_backend = NoiseModel.from_backend(backend)
    basis_aer  = ['u', 'cx', 'id', 'rz', 'sx', 'x', 'measure', 'if_else']
    qc_dm = transpile(qc, basis_gates=basis_aer, optimization_level=1)
    qc_dm.save_density_matrix()
    sim_dm = AerSimulator(method='density_matrix', noise_model=nm_backend)
    res_dm = sim_dm.run(qc_dm, shots=2048).result()
    dm_b   = DensityMatrix(res_dm.data()['density_matrix'])
    rho_b  = partial_trace(dm_b, [0, 1, 3])   # Bob q2
    fidelity = float(state_fidelity(rho_b, sv_input))
    error    = 1 - fidelity

    # Empirical success rate from AerSimulator
    total_dm  = sum(counts.values())
    success_dm = sum(v for k, v in counts.items() if k.split()[0] == '0')
    p_emp     = success_dm / total_dm if total_dm > 0 else 0.0

    fidelity_results[name] = fidelity
    error_rates.append(error)
    all_results.append({
        "backend"  : name,
        "counts"   : counts,
        "probs"    : probs,
        "fidelity" : fidelity,
        "error"    : error,
        "p_success": p_emp
    })
    print(f"Fidelity   : {fidelity:.4f}")
    print(f"Error      : {error:.4f}")
    print(f"Success prob (empirical): {p_emp:.4f}  (theory: {p_success_theory:.4f})")


# ### Backend Sweep Outputs
# 
# After the helper definitions and backend loop run, the notebook finishes with three outputs:
# 
# - a bar chart comparing fidelity across backends,
# - a concise printed summary, and
# - a detailed dump of the sampled counts, fidelities, and empirical success probabilities.
# 

# In[38]:


plt.figure(figsize=(8, 5))
plt.bar(fidelity_results.keys(), fidelity_results.values(),
        color='#5B9BD5', edgecolor='black')
plt.axhline(2/3, color='red', linestyle='--', lw=1.5, label='Classical limit (2/3)')
plt.title(f"Probabilistic Teleportation — Fidelity across Backends (n={n})")
plt.xlabel("Backend")
plt.ylabel("Fidelity")
plt.ylim(0, 1.1)
plt.legend()
plt.show()


# In[39]:


print("=" * 55)
print("  Probabilistic Quantum Teleportation (Agrawal & Pati)")
print("=" * 55)
print(f"  Entanglement parameter n  = {n}")
print(f"  Theoretical success prob  = {p_success_theory:.4f}")
print("=" * 55)

for name in fidelity_results:
    print(f"\nBackend : {name}")
    print(f"Fidelity : {fidelity_results[name]:.6f}")
    print(f"Error    : {1 - fidelity_results[name]:.6f}")

    if fidelity_results[name] > 2/3:
        print("✓ Beats classical limit")

print("=" * 55)


# In[40]:


print("\nDetailed Results:")
for res in all_results:
    print(f"\nBackend: {res['backend']}")
    print(f"Counts: {res['counts']}")
    print(f"Fidelity: {res['fidelity']:.4f}")
    print(f"Success prob (empirical): {res['p_success']:.4f}")

