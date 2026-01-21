import time
from utils import *
from random_seed import setup_seed
from problems.Volterra import Volterra10D
from models import CPIKANModel, PINNModel
import matplotlib.pyplot as plt
from train import train

if __name__ == "__main__":
	plt.rcParams['font.family'] = 'Microsoft YaHei'
	plt.rcParams['axes.unicode_minus'] = False
	setup_seed(104)
	device = torch.device("cuda:0")
	dtype = torch.float32
	m = 1 # eval(input())
	n_ksi = 500
	n_in = 4000
	n_bound = 50
	epoches = 2000
	if m == 0:
		model = CPIKANModel(input_dim=10, hidden_dim=10, dtype=dtype).to(device)
	elif m == 1:
		model = PINNModel(input_dim=10, hidden_dim=20, dtype=dtype).to(device)
	else:
		raise ValueError("Invalid value: m should be 0 or 1.")
	# 生成张量
	x_b = sample_boundary_points(dim=10, num_samples_per_boundary=n_bound, device=device, dtype=dtype, bound=True).requires_grad_(True) # torch.Size([20000, 10])
	x_i = torch.rand(n_in, 10, device=device, dtype=dtype) # torch.Size([10000, 10])
	s = torch.rand(n_ksi, 10, device=device, dtype=dtype)
	problem = Volterra10D(x_i, x_b, s)
	time0 = time.time()
	train(model, problem, max_epochs=epoches, lr=5e-3)
	time1 = time.time()
	print("本次训练用时：" + str(time1 - time0) + "s")
	with torch.no_grad():
		x_pred = torch.rand(50000, 10, device=device, dtype=dtype)
		u_pred = model(x_pred).view(-1, )
		u_solution = problem.solution(x_pred)
		residual = u_solution - u_pred
		relative_loss = torch.norm(residual ** 2) / torch.norm(u_solution ** 2)
		print(relative_loss.item())


	dim_data = [
		[[1, 1, 1, 1, 1 ,1 ,1 ,1], [2, 4]],
	    [[1, 1, 0, 0, 1 ,0 ,0 ,1], [8, 9]]]
	dimname = ["t", "x_1", "x_2", "x_3", "x_4", "x_5", "x_6", "x_7", "x_8", "x_9"]

	# 创建2×3的子图
	fig, axes = plt.subplots(2, 3, figsize=(18, 11))
	fig.tight_layout(pad=4)  # 调整子图间距

	for i, data in enumerate(dim_data):
		fixed_values = data[0]
		dims = data[1]
		n = 500

		grid35 = generate_grid_with_specific_dims(fixed_values, dims, n, dtype=dtype).to(device)
		pre_values = model(grid35).view(n, n).detach().cpu()
		solution_values = problem.solution(grid35).view(n, n).detach().cpu()
		pre_residual = torch.abs(pre_values - solution_values)
		X = (grid35.view(n, n, -1))[:, :, dims[0]].view(n, n).detach().cpu()
		Y = (grid35.view(n, n, -1))[:, :, dims[1]].view(n, n).detach().cpu()

		# 绘制真实解
		ax = axes[i, 0]
		contour = ax.contourf(X, Y, solution_values, levels=20, cmap='hot')
		# 添加等高线（黑色细线）
		ax.contour(X, Y, solution_values, levels=20, colors='k', linewidths=0.5)
		cbar = fig.colorbar(contour, ax=ax)
		ax.axis('scaled')
		ax.set_title(f'准确解', fontsize=22)
		ax.set_xlabel(f'${dimname[dims[0]]}$', fontsize=22)
		ax.set_ylabel(f'${dimname[dims[1]]}$', fontsize=22)
		ax.tick_params(axis='x', labelsize=15)
		ax.tick_params(axis='y', labelsize=15)
		cbar.ax.tick_params(labelsize=15)

		# 绘制预测解
		ax = axes[i, 1]
		contour = ax.contourf(X, Y, pre_values, levels=20, cmap='hot')
		ax.contour(X, Y, pre_values, levels=20, colors='k', linewidths=0.5)
		cbar = fig.colorbar(contour, ax=ax)
		ax.axis('scaled')
		ax.set_title(f'预测解', fontsize=22)
		ax.set_xlabel(f'${dimname[dims[0]]}$', fontsize=22)
		ax.set_ylabel(f'${dimname[dims[1]]}$', fontsize=22)
		ax.tick_params(axis='x', labelsize=15)
		ax.tick_params(axis='y', labelsize=15)
		cbar.ax.tick_params(labelsize=15)

		# 绘制误差
		ax = axes[i, 2]
		contour = ax.contourf(X, Y, pre_residual, levels=20, cmap='hot')
		ax.contour(X, Y, pre_residual, levels=20, colors='k', linewidths=0.5)
		cbar = fig.colorbar(contour, ax=ax)
		ax.axis('scaled')
		ax.set_title(f'绝对误差', fontsize=22)
		ax.set_xlabel(f'${dimname[dims[0]]}$', fontsize=22)
		ax.set_ylabel(f'${dimname[dims[1]]}$', fontsize=22)
		ax.tick_params(axis='x', labelsize=15)
		ax.tick_params(axis='y', labelsize=15)
		cbar.ax.tick_params(labelsize=15)

	plt.savefig(f'Volterra10D_MC-{model.name}.png')
	plt.show()

