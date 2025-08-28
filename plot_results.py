import pandas as pd
import matplotlib.pyplot as plt

def plot_results():
    metrics = ["Accuracy", "Precision", "F1-Score", "Specificity", "Sensitivity", "NPV", "MCC", "FPR", "FNR"]
    df_70_30 = pd.read_csv("results/MediFloraNet_70_30.csv" )
    df_80_20 = pd.read_csv("results/MediFloraNet_80_20.csv" )
    for metric in metrics:
        plt.figure(figsize=(8, 5))
        plt.plot(df_70_30["Model"], df_70_30[metric], marker='o', color='gold', label='Split 70/30', linewidth=2)
        plt.plot(df_80_20["Model"], df_80_20[metric], marker='o', color='red', label='Split 80/20', linewidth=2)

        plt.title(f"{metric} Comparison", fontsize=14)
        plt.xlabel("Models")
        plt.ylabel(metric)
        plt.grid( )
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"results/{metric}_comparison.png", dpi=300)
        plt.close()

    df_70_30.to_csv("results/MediFloraNet_70_30.csv", index=False)
    df_80_20.to_csv("results/MediFloraNet_80_20.csv", index=False)

    # --- Plot each metric (Proposed first) ---
    metrics = ["Accuracy", "Precision", "F1-Score", "Specificity", "Sensitivity", "NPV", "MCC", "FPR", "FNR"]

    for metric in metrics:
        plt.figure(figsize=(8, 5))
        plt.plot(df_70_30["Model"], df_70_30[metric], marker='o', color='gold', label='Split 70/30', linewidth=2)
        plt.plot(df_80_20["Model"], df_80_20[metric], marker='o', color='red', label='Split 80/20', linewidth=2)

        plt.title(f"{metric} Comparison", fontsize=14)
        plt.xlabel("Models")
        plt.ylabel(metric)
        plt.grid( )
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"results/{metric}_comparison.png", dpi=300)
        plt.close()

    df_table70_30 =  pd.read_csv("results/Benchmark Methods_MediFloraNet_70_30.csv")
    df_table80_20 = pd.read_csv("results/Benchmark Methods_MediFloraNet_80_20.csv")

    # --- Line plots ---
    metrics = ["Accuracy", "Precision", "F1-Score", "Specificity", "Sensitivity", "NPV", "MCC", "FPR", "FNR"]

    for metric in metrics:
        plt.figure(figsize=(8, 5))

        # Plot both splits
        plt.plot(df_table70_30["Model"], df_table70_30[metric], marker='o', color='gold', linewidth=2, label="70:30 Split")
        plt.plot(df_table80_20["Model"], df_table80_20[metric], marker='o', color='red', linewidth=2, label="80:20 Split")

        # plt.scatter(df_table5["Model"].iloc[-1], df_table5[metric].iloc[-1],
        #             s=180, c="blue", marker="*", edgecolors="black", label="Proposed (70:30)")
        # plt.scatter(df_table6["Model"].iloc[-1], df_table6[metric].iloc[-1],
        #             s=180, c="green", marker="*", edgecolors="black", label="Proposed (80:20)")

        plt.title(f"{metric} Comparison", fontsize=14)
        plt.xlabel("Models")
        plt.ylabel(metric)
        plt.grid(True )
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"results/{metric}lineplot.png", dpi=300)
        plt.close()

