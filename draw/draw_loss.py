import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_loss_history(ax, filepath="results/loss_history.npy", color_index=0, label=None):
    """
    Load a loss history .npy file and plot the loss curve.

    Args:
        filepath: Path to the .npy file containing loss history.
        color_index: Index into the seaborn "deep" color palette.
        label: Legend label for this curve.
    """
    # Academic paper style
    loss = np.load(filepath)
    epochs = np.arange(1, len(loss) + 1)
    ax.plot(epochs, loss, linewidth=1.5, color=sns.color_palette("deep")[color_index], label=label)
    


if __name__ == "__main__":
    sns.set_theme(style="ticks", context="paper", font_scale=1.4)
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "stix",
        "axes.linewidth": 1.2,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
    })
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss Curve")
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.7, color="gray")
    sns.despine()
    plot_loss_history(ax, r"E:\PythonProjectsByYear\Year2025\MC-PIKAN_Official\results\data\1DPINNs[3,20][50,40]loss_history.npy", color_index=0, label="PINNs [3,20]")
    plot_loss_history(ax, r"E:\PythonProjectsByYear\Year2025\MC-PIKAN_Official\results\data\1DPIKAN[3,10,3][50,40]loss_history.npy", color_index=1, label="PIKAN [3,10,3]")
    ax.legend(frameon=True, fancybox=False, edgecolor="black", fontsize=12)
    fig.tight_layout()
    plt.show()
    fig.savefig(r"E:\PythonProjectsByYear\Year2025\MC-PIKAN_Official\results\1D_loss_history.eps")
    fig.savefig(r"E:\PythonProjectsByYear\Year2025\MC-PIKAN_Official\results\1D_loss_history.svg")
    fig.savefig(r"E:\PythonProjectsByYear\Year2025\MC-PIKAN_Official\results\1D_loss_history.pdf")
    

