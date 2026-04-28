from qteleport import (
    build_telecloning_circuit,
    run_circuit,
    build_noise_model
)

qc, _ = build_telecloning_circuit(0.5, 1.0)

noise = build_noise_model()

result, counts = run_circuit(qc, noise_model=noise)

print("Telecloning counts:")
print(counts)