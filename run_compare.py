import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")                                   # draw to a file (works without a display)
import matplotlib.pyplot as plt
from common import load, simulate, metrics
from controllers import PD, PDGravity, ComputedTorque

T_MOVE = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0     # seconds for the move: try 2.0, then 0.6
m, d = load()
q0 = np.zeros(6)
q1 = np.array([0.5, -0.4, 0.8, 0.3, -0.5, 0.4])         # target pose (rad)
KP = np.array([300, 500, 300, 40, 30, 15.])
KD = np.array([40, 60, 40, 4, 3, 1.5])

controllers = {"PD": PD(KP, KD), "PD + gravity comp": PDGravity(m, KP, KD), "CT (no damping model)": ComputedTorque(wn=20),
               "CT (full model)": ComputedTorque(wn=20, model_damping=True)}
results = {name: simulate(m, d, c, q0, q1, T_move=T_MOVE, T_total=T_MOVE + 2.0) for name, c in controllers.items()}

print(f"move time = {T_MOVE} s")
print(f"{'controller':22s} {'rms err':>9s} {'peak err':>9s} {'final err':>10s} {'settle (s)':>10s} {'effort':>9s}")
for name, log in results.items():
    s = metrics(log, T_move=T_MOVE)
    print(f"{name:22s} {s['rms_err']:9.4f} {s['peak_err']:9.4f} {s['final_err']:10.5f} {s['settle_s']:10.2f} {s['effort']:9.1f}")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for name, log in results.items():
    ax[0].plot(log["t"], np.linalg.norm(log["qref"] - log["q"], axis=1), label=name)
ax[0].set_yscale("log"); ax[0].set_xlabel("time (s)"); ax[0].set_ylabel("tracking error norm (rad)"); ax[0].legend()
for name, log in results.items():
    ax[1].plot(log["t"], log["tau"][:, 1], label=name)
ax[1].set_xlabel("time (s)"); ax[1].set_ylabel("shoulder torque (N m)"); ax[1].legend()
plt.tight_layout(); plt.savefig(f"compare_{T_MOVE}s.png", dpi=120); print(f"saved compare_{T_MOVE}s.png")
