import numpy as np

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_data(data, use_log_scale=False, line_style='-', linewidth=1.5,
			   color_index=0, label=None, xlabel="X", ylabel="Y", title="Data1 Curve",
			   show_markers=False, marker_size=60, marker_style='o'):
	"""
	Plot the data1 curve with configurable y-axis scale and line style.

	Args:
		ax: Matplotlib axes object (if None, creates new figure and axes)
		use_log_scale: If True, set y-axis to logarithmic scale (default: False)
		line_style: Line style string (e.g., '-', '--', '-.', ':') (default: '-')
		linewidth: Line width (default: 1.5)
		color_index: Index into the seaborn "deep" color palette (default: 0)
		label: Legend label for this curve (default: None)
		xlabel: X-axis label (default: "X")
		ylabel: Y-axis label (default: "Y")
		title: Plot title (default: "Data1 Curve")
		show_markers: If True, overlay scatter markers on data points (default: False)
		marker_size: Size of scatter markers (default: 60)
		marker_style: Marker shape, e.g. 'o', 's', 'D', '^' (default: 'o')

	Returns:
		fig: Figure object (created if ax is None)
		ax: Axes object
	"""
	sns.set_theme(style="ticks", context="paper", font_scale=1.4)
	plt.rcParams.update({
			"font.family": "serif",
			"font.serif": ["Times New Roman"],
			"mathtext.fontset": "stix",
			"axes.linewidth": 1.2,
			"xtick.major.width": 1.0,
			"ytick.major.width": 1.0,
		})
	fig, ax = plt.subplots(figsize=(6, 5))

	# Extract data
	x_data = data[0, :]
	y_data = data[1, :]

	# Plot the curve
	curve_color = sns.color_palette("deep")[color_index]
	ax.plot(x_data, y_data, linestyle=line_style, linewidth=linewidth,
			color=curve_color, label=label)

	# Overlay scatter markers on data points
	if show_markers:
		ax.scatter(x_data, y_data, s=marker_size, marker=marker_style,
				   color=curve_color, edgecolors='white', linewidths=1.2, zorder=5)

	# Set y-axis scale
	if use_log_scale:
		ax.set_yscale("log")

	# Set labels and title
	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)
	ax.set_title(title)

	# Add grid
	ax.yaxis.grid(True, linestyle="--", linewidth=0.8, alpha=0.7, color="gray")
	sns.despine()

	# Add legend if label is provided
	if label is not None:
		ax.legend(frameon=True, fancybox=False, edgecolor="black", fontsize=12)

	return fig, ax


if __name__ == "__main__":
	data1 = np.array([[1, 2, 3, 4, 5, 6],
					  [0.0752912387251854, 0.004333922639489174, 0.002688765525817871, 0.003336397232487797,
					   0.001569639891386032, 0.0017865969566628337]])

	fig, ax = plot_data(
		data1,
		use_log_scale=False,
		line_style='-.',
		linewidth=1.5,
		color_index=2,
		xlabel="$d$",
		ylabel="$\mathscr{E}_{\\text{rel}}$",
		title="Degree of Chebyshev Polynomials",
		show_markers=True,
		marker_size=60,
		marker_style='o'
	)
	plt.show()
	for format in ["eps", "svg", "pdf"]:
		fig.savefig(f"degree_of_chebyshev_polynomials_curve.{format}")

	plt.close(fig)

	data2 = np.array([[5,10,20,30,40], [0.035479288548231125, 0.05607824772596359, 0.002119426615536213, 0.002294833306223154, 0.002688765525817871]])

	fig, ax = plot_data(
		data2,
		use_log_scale=False,
		line_style='--',
		linewidth=1.5,
		color_index=3,
		xlabel="$n_s$",
		ylabel="$\mathscr{E}_{\\text{rel}}$",
		title="Number of Samples",
		show_markers=True,
		marker_size=60,
		marker_style='o'
	)
	plt.show()
	for format in ["eps", "svg", "pdf"]:
		fig.savefig(f"number_of_samples.{format}")
	pass

