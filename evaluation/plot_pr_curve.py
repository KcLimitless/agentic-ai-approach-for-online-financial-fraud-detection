import numpy as np
import matplotlib.pyplot as plt

# --- Synthetic curve generation ---
recall = np.linspace(0, 1, 400)
precision = 0.985 - 0.12 * (1 - recall)**4
precision = np.clip(precision, 0, 1)

# actual model point
p_point = 0.985
r_point = 0.955

# --- Plot ---
plt.figure(figsize=(7, 6))
plt.plot(recall, precision, linewidth=2, label="PR Curve (synthetic)")
plt.scatter([r_point], [p_point], s=90, label=f"Model Point (P={p_point}, R={r_point})")

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve (Synthetic, AUCPR ≈ 0.989)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()

# --- Save the figure ---
plt.savefig("pr_curve.png", dpi=300, bbox_inches="tight")

plt.show()
