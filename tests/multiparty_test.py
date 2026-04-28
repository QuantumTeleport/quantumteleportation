from qteleport import (
    build_multiparty_circuit,
    run_circuit,
    build_noise_model
)

qc, _ = build_multiparty_circuit(1.57, 1.0)

noise = build_noise_model()

result, counts = run_circuit(qc, noise_model=noise)

print("Multiparty teleportation counts:")
print(counts)