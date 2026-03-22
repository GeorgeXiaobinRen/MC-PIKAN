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

    file_path = r'e:\PythonProjectsByYear\Year2025\MC-PIKAN_Official\results\data\10D_PIKAN[3,10,7]S10results.npy'
    data_list = np.load(file_path, allow_pickle=True)

    dimname = ["t", "x_1", "x_2", "x_3", "x_4", "x_5", "x_6", "x_7", "x_8", "x_9"]

    fig, axes = plt.subplots(2, 3, figsize=(9, 6))
    fig.tight_layout(pad=1, w_pad=1, h_pad=1)

    for i, data in enumerate(data_list):
        X = data['X']
        Y = data['Y']
        pre_values = data['pre_values']
        solution_values = data['solution_values']
        dims = data['dims']
        pre_residual = np.abs(pre_values - solution_values)

        plot_data = [
            ('Exact Solution', solution_values),
            ('Predicted Solution', pre_values),
            ('Absolute Error', pre_residual)
        ]

        for j, (title, data_plot) in enumerate(plot_data):
            ax = axes[i, j]
            # Use pcolormesh with gouraud shading for a perfectly smooth continuous colormap
            mesh = ax.pcolormesh(X, Y, data_plot, cmap='crest', shading='gouraud', rasterized=True)
            ax.contour(X, Y, data_plot, levels=20, colors='white', linewidths=0.5, alpha=0.5)
            fig.colorbar(mesh, ax=ax)
            ax.axis('scaled')

            if i == 0:
                ax.set_title(title)
            
            ax.set_xlabel(f'${dimname[dims[0]]}$')
            ax.set_ylabel(f'${dimname[dims[1]]}$')

    for format in ['pdf']:
        save_path = r'e:\PythonProjectsByYear\Year2025\MC-PIKAN_Official\results\Volterra10D_MC-PIKAN.' + format
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")
