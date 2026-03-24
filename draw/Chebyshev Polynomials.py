import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['legend.fontsize'] =20
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] =["Times New Roman"]
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False

def chebyshev_t(n, x):
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return x
    else:
        return 2 * x * chebyshev_t(n-1, x) - chebyshev_t(n-2, x)



x = np.linspace(-1, 1, 500)

plt.figure(figsize=(10, 6))
plt.title("Chebyshev Polynomials of the First Kind ($n\leq 6$)", fontsize=20)
plt.xlabel("$x$", fontsize=20)
plt.ylabel("$T_n(x)$", fontsize=20)
plt.grid(True, linestyle='--', alpha=0.6)
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)


colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'orange', 'purple']
linestyles = ['-', '--', '-.', ':', '-', '--', '-.']


for n in range(7):
    y = chebyshev_t(n, x)
    plt.plot(x, y,
             label=f'$T_{n}(x)$',
             color=colors[n],
             linestyle=linestyles[n],
             linewidth=1.5)

plt.subplots_adjust(right=0.85)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=20)
plt.tight_layout()
plt.savefig('Chebyshev Polynomials.pdf', dpi=300)
plt.show()