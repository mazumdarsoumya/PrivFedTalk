import pandas as pd
import matplotlib.pyplot as plt

def plot_convergence_from_csv(csv_path: str, out_pdf: str):
    df = pd.read_csv(csv_path)
    if "round" not in df.columns: return
    plt.figure()
    if "avg_score" in df.columns:
        plt.plot(df["round"], df["avg_score"])
        plt.xlabel("Round"); plt.ylabel("Avg client score")
    else:
        plt.plot(df["round"], df.index)
        plt.xlabel("Round"); plt.ylabel("Value")
    plt.grid(True); plt.tight_layout(); plt.savefig(out_pdf); plt.close()
