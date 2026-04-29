from .standard_teleportation import build_standard_circuit, build_noise_model, run_circuit
from .controlled_teleportation import build_controlled_circuit
from .probabilistic_teleportation import build_probabilistic_circuit
from .bidirectional_teleportation import build_bidirectional_circuit
from .multiparty_teleportation import build_multiparty_circuit
from .broadcasting_telecloning import build_telecloning_circuit

PROTOCOL_MAP = {
    "standard": build_standard_circuit,
    "controlled": build_controlled_circuit,
    "probabilistic": build_probabilistic_circuit,
    "bidirectional": build_bidirectional_circuit,
    "multiparty": build_multiparty_circuit,
    "telecloning": build_telecloning_circuit,
}


def teleport(protocol, run=False, noise=False, noise_params=None, **kwargs):
    """
    Unified teleportation interface.

    Parameters
    ----------
    protocol : str
    run : bool → whether to execute circuit
    noise : bool → whether to include noise
    noise_params : dict → parameters for noise model
    kwargs : protocol-specific arguments
    """

    if protocol not in PROTOCOL_MAP:
        raise ValueError(f"Unknown protocol: {protocol}")

    # ── Step 1: Build circuit ──
    builder = PROTOCOL_MAP[protocol]
    result = builder(**kwargs)

    qc = result[0]   # extract circuit

    # ── Step 2: If only building → return circuit ──
    if not run:
        return qc

    # ── Step 3: Build noise (optional) ──
    noise_model = None
    if noise:
        noise_params = noise_params or {}
        noise_model = build_noise_model(**noise_params)

    # ── Step 4: Run circuit ──
    result, counts = run_circuit(qc, noise_model=noise_model)

    return {
        "circuit": qc,
        "result": result,
        "counts": counts
    }