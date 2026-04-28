from qteleport import (
    build_probabilistic_circuit,
    run_circuit,
    build_noise_model
)

qc, _ = build_probabilistic_circuit(0.5, 1.0, 0.7)

noise = build_noise_model()

result, counts = run_circuit(qc, noise_model=noise)

print("Probabilistic teleportation counts:")
print(counts)