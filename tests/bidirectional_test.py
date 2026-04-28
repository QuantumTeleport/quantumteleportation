# tests/bidirectional_test.py

from qteleport import (
    build_bidirectional_circuit,
    run_circuit,
    build_noise_model
)

qc, _, _ = build_bidirectional_circuit(0.5, 1.0, 1.0, 0.3)

noise = build_noise_model()

result, counts = run_circuit(qc, noise_model=noise)

print("Bidirectional teleportation counts:")
print(counts)