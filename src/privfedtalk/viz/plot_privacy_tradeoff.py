import numpy as np
import matplotlib.pyplot as plt

def plot_privacy_tradeoff(out_pdf: str):
    sigmas = np.array([0.0, 0.1, 0.2, 0.3, 0.5, 0.8])
    quality = np.exp(-2.0 * sigmas) * 0.85
    plt.figure()
    plt.plot(sigmas, quality, marker="o")
    plt.xlabel("DP noise multiplier (sigma)")
    plt.ylabel("Synthetic quality proxy")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_pdf)
    plt.close()
