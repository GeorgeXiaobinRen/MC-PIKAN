import numpy as np

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_data(data, use_log_scale=False, line_style='-', linewidth=1.5,
			   color_index=0, label=None, xlabel="X", ylabel="Y", title="Data1 Curve",
			   show_markers=False, marker_size=60, marker_style='o'):
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
					  [0.08179549127817154, 0.005374509375542402, 0.0028772600926458836, 0.0035432027652859688,
					   0.001898447168059647, 0.0019332696683704853]])

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

	data2 = np.array([[5,10,20,30,40], [0.04323914647102356, 0.06738028675317764, 0.0024911907967180014, 0.002716680755838752, 0.0028772600926458836]])

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
	for format in ["pdf"]:
		fig.savefig(f"number_of_samples.{format}")
	pass

