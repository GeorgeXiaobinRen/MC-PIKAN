import time
import matplotlib.pyplot as plt
from random_seed import setup_seed
from problems.Volterra_1D import solution, train
from models import *

if __name__ == "__main__":
	plt.rcParams['font.family'] = 'Microsoft YaHei'
	plt.rcParams['axes.unicode_minus'] = False
	# set up the random seed
	setup_seed(233)
	# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	dtype = torch.float32
	n_s = 200
	n_e = 1000
	# 定义域
	x = torch.linspace(0, 1, 200, device=device, dtype=dtype)
	s1 = torch.rand(n_s, device=device, dtype=dtype)
	s2 = torch.rand(n_s, device=device, dtype=dtype)
	s = [s1, s2]
	m = 0 # eval(input())
	# 构建模型
	if m==0:
		model = PINNModel(num_layers=2, hidden_dim=4, dtype=dtype).to(device)
	elif m==1:
		model = CPIKANModel(num_layers=2, hidden_dim=2, degree=2, dtype=dtype).to(device)
	else:
		raise ValueError("Invalid value: m should be 0 or 1.")
	print_modelsize(model)
	print("Model on device:", device)
	# 训练模型
	time0 = time.time()
	train(model, x, s, lr=0.01, epochs=n_e)
	time1 = time.time()
	print("本次训练用时：" + str(time1 - time0) + "s")
	# 预测结果
	with torch.no_grad():
		x_pred = torch.linspace(0, 1, 2000, device=device, dtype=dtype)
		u_pred = model(x_pred.view(-1, 1))
		u_solution = solution(x_pred).view(-1, 1)
		residual = torch.abs(u_solution - u_pred)


	# 创建画布和子图
	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
	plt.rcParams['legend.fontsize'] = 15

	# 左边子图：预测值和真实解
	ax1.plot(x_pred.detach().cpu(), u_pred.detach().cpu(), label="准确解", ls='--')
	ax1.plot(x_pred.detach().cpu(), u_solution.detach().cpu(), label="预测解")
	ax1.set_xlabel("$x$", fontsize=15)
	ax1.set_ylabel("$u(x)$", fontsize=15)
	ax1.set_title(f"使用MC-{model.name}求解1维Volterra积分方程的解", fontsize=20)
	ax1.legend()
	ax1.grid()

	# 右边子图：残差（对数刻度）
	ax2.plot(x_pred.detach().cpu(), residual.detach().cpu(), label="绝对误差", color="red")
	ax2.set_xlabel("$x$", fontsize=15)
	ax2.set_ylabel("绝对误差（对数刻度）", fontsize=15)
	ax2.set_yscale('log')  # 设置纵轴为对数刻度
	ax2.set_title("预测解与准确解之间的绝对误差", fontsize=20)
	ax2.legend()
	ax2.grid()

	# 添加残差的统计信息
	relative_loss = torch.sqrt((residual**2).mean())/torch.sqrt((u_solution**2).mean())
	print(relative_loss.item())
	# 显示参数
	print(p.size() for p in model.parameters())
	print(model.lys[0].weight)
	print(model.lys[0].bias)
	print(model.lys[1].weight)
	print(model.lys[1].bias)

	# 显示图像0
	plt.tight_layout()  # 自动调整子图间距
	plt.savefig(f"Volterra1D_MC-{model.name}.png")
	plt.show()
