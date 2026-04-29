# tests/teleport_test.py

from qteleport import teleport


# ── Test 1: Build only ──
qc = teleport(protocol="standard", theta=0.5, phi=1.0)

print("Standard circuit depth:", qc.depth())
print("Gate counts:", qc.count_ops())


# ── Test 2: Controlled + noise ──
result = teleport(
    protocol="controlled",
    theta=0.5,
    phi=1.0,
    run=True,
    noise=True
)

print("\nControlled teleportation counts:")
print(result["counts"])