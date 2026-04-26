from qteleport import build_standard_circuit

qc, _ = build_standard_circuit(0.5, 1.0)

print("Circuit depth:", qc.depth())
print("Gate counts:", qc.count_ops())
