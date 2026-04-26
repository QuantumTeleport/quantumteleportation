#!/usr/bin/env python
# coding: utf-8

# # Multiple-Party Quantum Teleportation — EPR pair to Bob & Claire (FakeKawasaki + Noise Simulation)
# 
# Alice teleports $|\Psi\rangle_{12}$ (an **EPR pair**) simultaneously to Bob and Claire using a shared GHZ channel. Both receivers recover one qubit each of the entangled pair.
# 
# **Pseudocode (EPR pair to Bob & Claire):**
# 
# $$|\Psi\rangle_{12}\,|GHZ\rangle_{345} = \frac{1}{\sqrt{2}}(a|00\rangle_{12} + b|11\rangle_{12})\,\frac{1}{\sqrt{2}}(|000\rangle_{345}+|111\rangle_{345})$$
# 
# $$= \frac{1}{2}\sum_{i,j\in\{0,1\}}|\Omega_{ij}\rangle_{123}\,X^i Z^j|\Psi\rangle_{45}$$
# 
# 1. CNOT($1\to3$), CNOT($2\to3$), then $H(1)$, $H(2)$
#    $\longrightarrow \frac{1}{2}\sum_{i,j}|ij0\rangle_{123}\,X^iZ^j|\Psi\rangle_{45}$
# 2. Alice measures $(1,2,3)$ in std. basis → sends $(i,j)$ to Bob & Claire
# 3. Bob & Claire jointly apply $Z^jX^i$ → recover $|\Psi\rangle_{45}$
# 
# **Key physics:** Alice holds a 2-qubit EPR pair and 1 GHZ qubit. Bob and Claire each hold one GHZ qubit. After Alice's Bell-basis measurement, Bob and Claire's qubits collapse into a rotated copy of $|\Psi\rangle$ and they jointly apply corrections to recover it.
# 
# **Workflow in this notebook:** setup and backend inspection, circuit construction, transpilation, ideal/noisy simulation, hardware-oriented sampling, input-state fidelity tests, noise sweeps, and finally a backend-by-backend comparison.
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


# ## 2 · Backend & Chip Visualisation

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


# ## 3 · Build the Multi-Party Teleportation Circuit
# 
# **Qubit assignment (5 qubits total):**
# 
# | Qubit | Label | Party | Role |
# |---|---|---|---|
# | `q[0]` | 1 | Alice | First qubit of input EPR pair $|\Psi\rangle_{12}$ |
# | `q[1]` | 2 | Alice | Second qubit of input EPR pair |
# | `q[2]` | 3 | Alice | Alice's GHZ qubit (measured with 1 and 2) |
# | `q[3]` | 4 | Bob   | Bob's GHZ qubit — receives first qubit of $|\Psi\rangle$ |
# | `q[4]` | 5 | Claire| Claire's GHZ qubit — receives second qubit of $|\Psi\rangle$ |
# 
# **Classical register:**
# - `cm[0]` = Alice's measurement of qubit 1 → bit $i$
# - `cm[1]` = Alice's measurement of qubit 2 → bit $j$
# - `cm[2]` = Alice's measurement of qubit 3 → bit (always 0 post-protocol, kept for verification)
# 
# **Input state** $|\Psi\rangle_{12}$: a parametrised EPR-like pair
# $$|\Psi\rangle_{12} = \cos\tfrac{\theta}{2}|00\rangle + e^{i\phi}\sin\tfrac{\theta}{2}|11\rangle$$
# prepared as $H$ on qubit 1 followed by $CNOT(1\to2)$ with additional $R_y/R_z$ rotations for generality.
# 
# **Corrections** $Z^j X^i$ applied **jointly** to Bob (q[3]) and Claire (q[4]):
# 
# | $i$ | $j$ | Bob applies | Claire applies |
# |---|---|---|---|
# | 0 | 0 | $I$ | $I$ |
# | 0 | 1 | $Z$ | $Z$ |
# | 1 | 0 | $X$ | $X$ |
# | 1 | 1 | $ZX$ | $ZX$ |

# In[24]:


# ── Input EPR pair parameters ────────────────────────────────────────────────
# |Ψ⟩_12 = cos(θ/2)|00⟩ + e^{iφ}·sin(θ/2)|11⟩
# We use θ=π/2 (maximally entangled) with a relative phase φ=π/3
theta = np.pi / 2   # entanglement angle (π/2 → maximally entangled)
phi   = np.pi / 3   # relative phase


def create_multiparty_teleportation_circuit(theta, phi):
    """
    Multi-party teleportation of an EPR pair to Bob & Claire.

    Alice holds qubits 1,2 (input EPR pair) and qubit 3 (GHZ anchor).
    Bob holds qubit 4, Claire holds qubit 5.

    Protocol:
      1. Prepare |Ψ⟩_12 on q[0],q[1]
      2. Prepare GHZ_345 on q[2],q[3],q[4]
      3. CNOT(1→3), CNOT(2→3), H(1), H(2)   [generalised Bell basis]
      4. Alice measures q[0],q[1],q[2] → (i,j,0)
      5. Bob & Claire apply Z^j X^i to q[3],q[4]
    """
    # ── Reference: input EPR state (for fidelity) ─────────────────────────
    qc_epr = QuantumCircuit(2)
    qc_epr.ry(theta, 0)          # rotation on qubit 1
    qc_epr.cx(0, 1)              # entangle
    qc_epr.rz(phi, 0)            # relative phase
    sv_input = Statevector.from_instruction(qc_epr)

    print("Input EPR statevector |Ψ⟩_12:")
    print(np.round(sv_input.data, 4))

    # ── Registers ─────────────────────────────────────────────────────────
    # q0=qubit1(Alice), q1=qubit2(Alice), q2=qubit3(Alice/GHZ),
    # q3=qubit4(Bob),   q4=qubit5(Claire)
    q  = QuantumRegister(5, 'q')
    cm = ClassicalRegister(3, 'cm')   # cm[0]=i, cm[1]=j, cm[2]=qubit3 check
    qc = QuantumCircuit(q, cm)

    # ── Step 1: Prepare input EPR pair |Ψ⟩_12 on q[0], q[1] ─────────────
    qc.ry(theta, q[0])
    qc.cx(q[0], q[1])
    qc.rz(phi,  q[0])
    qc.barrier(label="Input |Ψ⟩_12")

    # ── Step 2: Prepare GHZ state on q[2], q[3], q[4] ────────────────────
    # |GHZ⟩_345 = (1/√2)(|000⟩ + |111⟩)
    qc.h(q[2])
    qc.cx(q[2], q[3])
    qc.cx(q[2], q[4])
    qc.barrier(label="|GHZ⟩_345")

    # ── Step 3: CNOT(1→3), CNOT(2→3) then H(1), H(2) ────────────────────
    # Maps |Ω_ij⟩_123 ⊗ X^iZ^j|Ψ⟩_45 → |ij0⟩_123 ⊗ X^iZ^j|Ψ⟩_45
    qc.cx(q[0], q[2])    # CNOT(qubit1 → qubit3)
    qc.cx(q[1], q[2])    # CNOT(qubit2 → qubit3)
    qc.h(q[0])           # H on qubit 1
    qc.h(q[1])           # H on qubit 2
    qc.barrier(label="CNOT(1→3),CNOT(2→3),H(1),H(2)")

    # ── Step 4: Alice measures qubits 1, 2, 3 → (i, j, 0) ───────────────
    qc.measure(q[0], cm[0])   # i
    qc.measure(q[1], cm[1])   # j
    qc.measure(q[2], cm[2])   # should always be 0
    qc.barrier(label="Alice meas (i,j) → Bob & Claire")

    # ── Step 5: Bob & Claire jointly apply Z^j X^i ────────────────────────
    # Bob corrects q[3]: X if i=1, Z if j=1
    with qc.if_test((cm[0], 1)):   # X if i=1
        qc.x(q[3])
    with qc.if_test((cm[1], 1)):   # Z if j=1
        qc.z(q[3])

    # Claire corrects q[4]: X if i=1, Z if j=1
    with qc.if_test((cm[0], 1)):   # X if i=1
        qc.x(q[4])
    with qc.if_test((cm[1], 1)):   # Z if j=1
        qc.z(q[4])
    qc.barrier(label="Bob & Claire: Z^j X^i")

    return qc, sv_input


#  Create circuit
qc, sv_input = create_multiparty_teleportation_circuit(theta, phi)

#  Circuit info
print(f"\nCircuit depth : {qc.depth()}")
print(f"Gate counts   : {dict(qc.count_ops())}")

#  Draw circuit
qc.draw('mpl', style="iqp", fold=-1)


# ## 4 · Preset Pass Manager (optimization_level=2)

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


# ## 5 · Custom 4-Stage Pass Manager
# 
# Stages: **SabreLayout → SabreSwap → BasisTranslator → Optimize1q**

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


# ## 6 · Depth Comparison

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


# ## 7 · Layered Noise Model
# 
# Three independent noise layers:
# 
# | Layer | Type | Applies to |
# |---|---|---|
# | 1 | Depolarizing gate error | 1-qubit & 2-qubit gates |
# | 2 | T1/T2 thermal relaxation | All gates (via gate time) |
# | 3 | SPAM / Readout error | Measurement |

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
# The output is a **2-qubit state** on Bob's qubit 4 and Claire's qubit 5. We compare the joint $(q[3], q[4])$ state to the input EPR pair $|\Psi\rangle_{12}$.
# 
# Tracing: `partial_trace(dm, [0,1,2])` — keeps q[3] and q[4] (Bob & Claire), traces out Alice's 3 qubits.

# In[29]:


qc_ideal = qc.copy()
qc_ideal.save_density_matrix()

sim_ideal    = AerSimulator(method='density_matrix')
result_ideal = sim_ideal.run(qc_ideal, shots=4096).result()

dm_ideal = DensityMatrix(result_ideal.data()['density_matrix'])

# 5 qubits: q0(1/Alice), q1(2/Alice), q2(3/Alice), q3(4/Bob), q4(5/Claire)
# Output EPR pair lives on q3 and q4 → trace out q0, q1, q2 (Alice)
rho_bc_ideal = partial_trace(dm_ideal, [0, 1, 2])

fidelity_ideal = state_fidelity(rho_bc_ideal, sv_input)
print(f"Ideal fidelity — (Bob q[3], Claire q[4]) vs |Ψ⟩_12 : {fidelity_ideal:.6f}")

# Individual qubit fidelities (marginal states)
rho_bob_ideal   = partial_trace(dm_ideal, [0, 1, 2, 4])   # Bob q3 alone
rho_claire_ideal = partial_trace(dm_ideal, [0, 1, 2, 3])  # Claire q4 alone
sv_0 = Statevector.from_label('0')
sv_plus = Statevector([1/np.sqrt(2), 1/np.sqrt(2)])
print(f"Bob q[3]   purity : {float(np.real(np.trace(rho_bob_ideal.data @ rho_bob_ideal.data))):.4f}")
print(f"Claire q[4] purity : {float(np.real(np.trace(rho_claire_ideal.data @ rho_claire_ideal.data))):.4f}")
print("(Individual qubits of a maximally entangled pair are maximally mixed — purity ≈ 0.5)")

counts_ideal = result_ideal.get_counts()
print("\nMeasurement counts (ideal):")
for outcome, cnt in sorted(counts_ideal.items()):
    print(f"  |{outcome}>  {cnt:5d}  ({100*cnt/4096:.1f}%)")


# ## 9 · Noisy Simulation (Density Matrix + Layered Noise)

# In[30]:


# Transpile to AerSimulator basis so noise model gates match
basis_aer  = ['u', 'cx', 'id', 'rz', 'sx', 'x', 'measure', 'if_else']
qc_noisy_t = transpile(qc, basis_gates=basis_aer, optimization_level=1)
qc_noisy_t.save_density_matrix()

sim_noisy    = AerSimulator(method='density_matrix', noise_model=noise_model)
result_noisy = sim_noisy.run(qc_noisy_t, shots=4096).result()

dm_noisy     = DensityMatrix(result_noisy.data()['density_matrix'])

# Trace out Alice — keep Bob (q3) and Claire (q4)
rho_bc_noisy = partial_trace(dm_noisy, [0, 1, 2])

fidelity_noisy = state_fidelity(rho_bc_noisy, sv_input)
print(f"Noisy fidelity — (Bob q[3], Claire q[4]) vs |Ψ⟩_12 : {fidelity_noisy:.6f}")
print(f"Ideal fidelity                                       : {fidelity_ideal:.6f}")
print(f"Fidelity degradation                                 : {fidelity_ideal - fidelity_noisy:.6f}")


# ## 10 · SamplerV2 on FakeKawasaki (IBM Runtime Primitive)

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
# Classical register cm: Alice's (i, j, qubit3_check)
counts_sampler = pub_result.data.cm.get_counts()

print("SamplerV2 — Alice measurement counts (FakeKawasaki hardware noise):")
total = sum(counts_sampler.values())
for outcome, cnt in sorted(counts_sampler.items()):
    print(f"  cm=|{outcome}>  {cnt:5d}  ({100*cnt/total:.1f}%)")

# Bit 2 (qubit 3) should always be 0 in ideal case
q3_one = sum(v for k, v in counts_sampler.items() if k[0] == '1')
print(f"\nQubit 3 measured as |1⟩ (noise indicator): {100*q3_one/total:.1f}%")
print("(Should be ≈0% ideal; non-zero fraction indicates noise/error)")


# ## 11 · Per-Channel Fidelity Comparison
# 
# Different input EPR states are tested. Each is a 2-qubit entangled state:
# - $|\Phi^+\rangle = \frac{1}{\sqrt{2}}(|00\rangle+|11\rangle)$ — standard Bell state
# - $|\Phi^-\rangle = \frac{1}{\sqrt{2}}(|00\rangle-|11\rangle)$
# - $|\Psi^+\rangle = \frac{1}{\sqrt{2}}(|01\rangle+|10\rangle)$
# - $|\Psi^-\rangle = \frac{1}{\sqrt{2}}(|01\rangle-|10\rangle)$
# - $|\Psi(\theta,\phi)\rangle$ — our parametrised input state
# 
# Fidelity measures how well the **joint** Bob+Claire state matches the original EPR pair.

# In[33]:


def make_bell(kind):
    """Return a 2-qubit QuantumCircuit preparing one of the four Bell states."""
    qc_ = QuantumCircuit(2)
    if kind in ('phi+', 'phi-'):
        qc_.h(0);  qc_.cx(0, 1)
        if kind == 'phi-': qc_.z(0)
    else:  # psi+, psi-
        qc_.x(1);  qc_.h(0);  qc_.cx(0, 1)
        if kind == 'psi-': qc_.z(0)
    return qc_

qc_custom = QuantumCircuit(2)
qc_custom.ry(theta, 0);  qc_custom.cx(0, 1);  qc_custom.rz(phi, 0)

test_states = {
    '|Φ+⟩': make_bell('phi+'),
    '|Φ-⟩': make_bell('phi-'),
    '|Ψ+⟩': make_bell('psi+'),
    '|Ψ-⟩': make_bell('psi-'),
    '|Ψ⟩' : qc_custom,
}


def run_multiparty_fidelity(input2_qc, noisy=False):
    """Build & run multi-party teleportation for a given 2-qubit input."""
    q_  = QuantumRegister(5, 'q')
    cm_ = ClassicalRegister(3, 'cm')
    qc_ = QuantumCircuit(q_, cm_)

    # Input EPR on q0, q1
    qc_.compose(input2_qc, qubits=[0, 1], inplace=True)
    qc_.barrier()

    # GHZ on q2, q3, q4
    qc_.h(q_[2]);  qc_.cx(q_[2], q_[3]);  qc_.cx(q_[2], q_[4])
    qc_.barrier()

    # CNOT(1→3), CNOT(2→3), H(1), H(2)
    qc_.cx(q_[0], q_[2]);  qc_.cx(q_[1], q_[2])
    qc_.h(q_[0]);  qc_.h(q_[1])
    qc_.barrier()

    # Alice measures
    qc_.measure(q_[0], cm_[0])
    qc_.measure(q_[1], cm_[1])
    qc_.measure(q_[2], cm_[2])
    qc_.barrier()

    # Bob & Claire corrections: Z^j X^i on both q3 and q4
    with qc_.if_test((cm_[0], 1)): qc_.x(q_[3])
    with qc_.if_test((cm_[1], 1)): qc_.z(q_[3])
    with qc_.if_test((cm_[0], 1)): qc_.x(q_[4])
    with qc_.if_test((cm_[1], 1)): qc_.z(q_[4])

    sv_ref = Statevector.from_instruction(input2_qc)
    nm  = noise_model if noisy else None
    sim = AerSimulator(method='density_matrix', noise_model=nm)
    if noisy:
        basis_aer = ['u', 'cx', 'id', 'rz', 'sx', 'x', 'measure', 'if_else']
        qc_ = transpile(qc_, basis_gates=basis_aer, optimization_level=1)
    qc_.save_density_matrix()
    res = sim.run(qc_, shots=2048).result()
    dm_ = DensityMatrix(res.data()['density_matrix'])
    # Keep Bob (q3) and Claire (q4) — trace out Alice q0,q1,q2
    rho = partial_trace(dm_, [0, 1, 2])
    return state_fidelity(rho, sv_ref)


fid_ideal_per = {k: run_multiparty_fidelity(v, noisy=False) for k, v in test_states.items()}
fid_noisy_per = {k: run_multiparty_fidelity(v, noisy=True)  for k, v in test_states.items()}

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
ax.set_ylabel("Joint State Fidelity F(ρ_{BC}, |Ψ⟩)");  ax.set_ylim(0.5, 1.1)
ax.set_title("Multi-Party Teleportation — Per-Input EPR Fidelity: Ideal vs Noisy")
ax.legend();  ax.axhline(1.0, color='grey', linestyle='--', lw=0.8)
plt.tight_layout();  plt.show()


# ## 12 · Fidelity vs Noise Strength Sweep

# In[34]:


p2q_sweep    = np.linspace(0.001, 0.15, 20)
fidelities_sweep = []
basis_aer    = ['u', 'cx', 'id', 'rz', 'sx', 'x', 'measure', 'if_else']

for p2q in p2q_sweep:
    nm_sweep = build_noise_model(p1q=p2q/10, p2q=p2q, p_meas=p2q/5)
    q_  = QuantumRegister(5, 'q')
    cm_ = ClassicalRegister(3, 'cm')
    qc_ = QuantumCircuit(q_, cm_)

    # Input EPR
    qc_.ry(theta, q_[0]);  qc_.cx(q_[0], q_[1]);  qc_.rz(phi, q_[0]);  qc_.barrier()

    # GHZ
    qc_.h(q_[2]);  qc_.cx(q_[2], q_[3]);  qc_.cx(q_[2], q_[4]);  qc_.barrier()

    # Bell basis ops
    qc_.cx(q_[0], q_[2]);  qc_.cx(q_[1], q_[2])
    qc_.h(q_[0]);  qc_.h(q_[1]);  qc_.barrier()

    # Measure
    qc_.measure(q_[0], cm_[0])
    qc_.measure(q_[1], cm_[1])
    qc_.measure(q_[2], cm_[2]);  qc_.barrier()

    # Corrections
    with qc_.if_test((cm_[0], 1)): qc_.x(q_[3])
    with qc_.if_test((cm_[1], 1)): qc_.z(q_[3])
    with qc_.if_test((cm_[0], 1)): qc_.x(q_[4])
    with qc_.if_test((cm_[1], 1)): qc_.z(q_[4])

    qc_t = transpile(qc_, basis_gates=basis_aer, optimization_level=1)
    qc_t.save_density_matrix()
    sim_ = AerSimulator(method='density_matrix', noise_model=nm_sweep)
    res_ = sim_.run(qc_t, shots=2048).result()
    dm_  = DensityMatrix(res_.data()['density_matrix'])
    rho_ = partial_trace(dm_, [0, 1, 2])   # keep Bob+Claire
    fidelities_sweep.append(state_fidelity(rho_, sv_input))

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(p2q_sweep, fidelities_sweep, 'o-', color='#5B9BD5', lw=2, ms=5,
        label='Multi-party teleportation fidelity')
ax.axhline(2/3, color='#ED7D31', linestyle='--', lw=1.5, label='Classical limit (2/3)')
ax.axhline(1.0, color='grey', linestyle=':', lw=1)
ax.fill_between(p2q_sweep, fidelities_sweep, 2/3,
                where=[f > 2/3 for f in fidelities_sweep],
                alpha=0.15, color='#5B9BD5', label='Quantum advantage region')
ax.set_xlabel("2-qubit Gate Depolarizing Error (p₂q)")
ax.set_ylabel("Joint State Fidelity F(ρ_{BC}, |Ψ⟩)")
ax.set_title("Multi-Party Teleportation — Fidelity vs Noise Strength")
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
    qc, sv_input = create_multiparty_teleportation_circuit(theta, phi)

    # 🔧 choose layout (manual OR automatic)
    layout = None
    if backend.num_qubits >= 20:
        layout = get_qubit_layout(data["coupling_map"], 5)   # 5 qubits

    # Use layout-enabled transpile
    qc_t = transpile_with_layout(qc, backend, layout)

    # ▶ Run simulation
    counts, probs = run_simulation(qc_t)

    print("\nBob & Claire outcomes:")
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
    rho_b  = partial_trace(dm_b, [0, 1, 2])   # keep Bob+Claire
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
    print(f"Fidelity : {fidelity:.4f}")
    print(f"Error    : {error:.4f}")


# ### Backend Sweep Outputs
# 
# After the helper definitions and backend loop run, the notebook finishes with three outputs:
# 
# - a bar chart comparing joint-state fidelity across backends,
# - a concise printed summary, and
# - a detailed dump of the sampled counts and fidelity values.
# 

# In[39]:


plt.figure(figsize=(8, 5))
plt.bar(fidelity_results.keys(), fidelity_results.values(),
        color='#5B9BD5', edgecolor='black')
plt.axhline(2/3, color='red', linestyle='--', lw=1.5, label='Classical limit (2/3)')
plt.title("Multi-Party Teleportation — Joint Fidelity across Backends")
plt.xlabel("Backend")
plt.ylabel("Joint State Fidelity F(ρ_{BC}, |Ψ⟩)")
plt.ylim(0, 1.1)
plt.legend()
plt.show()


# In[40]:


print("=" * 55)
print("  Multiple-Party Teleportation (EPR pair to Bob & Claire)")
print("  Results Summary")
print("=" * 55)

for name in fidelity_results:
    print(f"\nBackend : {name}")
    print(f"Fidelity : {fidelity_results[name]:.6f}")
    print(f"Error    : {1 - fidelity_results[name]:.6f}")

    if fidelity_results[name] > 2/3:
        print("✓ Beats classical limit")

print("=" * 55)


# In[41]:


print("\nDetailed Results:")
for res in all_results:
    print(f"\nBackend: {res['backend']}")
    print(f"Counts: {res['counts']}")
    print(f"Fidelity: {res['fidelity']:.4f}")

