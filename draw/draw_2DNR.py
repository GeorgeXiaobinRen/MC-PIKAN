import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

if __name__ == "__main__":
    sns.set_theme(style="white")
    plt.rcParams.update({
        "font.family": "serif",
		"font.serif": ["Times New Roman"],
		"mathtext.fontset": "stix",
        'font.size': 20,
        'axes.unicode_minus': False
    })

    file_path = r'e:\PythonProjectsByYear\Year2025\MC-PIKAN_Official\results\data\2DNR_PIKAN[3,10,3]S40results.npy'
    data = np.load(file_path, allow_pickle=True).item()

    X_pred = data['X_pred']
    u_pred = data['u_pred']
    u_solution = data['u_solution']

    n_eta, n_xi = 500, 500

    Eta = X_pred[:, 0].reshape(n_eta, n_xi)
    Xi = X_pred[:, 1].reshape(n_eta, n_xi)
    U_sol = u_solution.reshape(n_eta, n_xi)
    U_pre = u_pred.reshape(n_eta, n_xi)
    Residual = np.abs(U_sol - U_pre)

    fig, axes = plt.subplots(1, 3, figsize=(10, 5))
    fig.tight_layout(pad=1.5, w_pad=0.4)

    plot_data = [
        ('Exact Solution', U_sol),
        ('Predicted Solution', U_pre),
        ('Absolute Error', Residual)
    ]

    for ax, (title, data_plot) in zip(axes, plot_data):
        mesh = ax.pcolormesh(Eta, Xi, data_plot, cmap='mako', shading='gouraud', rasterized=True)
        ax.contour(Eta, Xi, data_plot, levels=20, colors='white', linewidths=0.5, alpha=0.5)
        fig.colorbar(mesh, ax=ax)
        ax.axis('scaled')
        ax.set_title(title)
        # ax.set_xlabel(r'$\eta$')
        # ax.set_ylabel(r'$\xi$')

    for format in ['pdf']:
        save_path = r'e:\PythonProjectsByYear\Year2025\MC-PIKAN_Official\results\Volterra2DNR_MC-CPIKAN.' + format
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")
