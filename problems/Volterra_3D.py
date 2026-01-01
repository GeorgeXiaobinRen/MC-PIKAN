import torch
from itertools import product
import time

from torch.optim.lr_scheduler import StepLR


def f(X):
	t, x1, x2= X[:, 0], X[:, 1], X[:, 2]
	return (torch.sin(x1)*torch.cos(x2) + t*torch.cos(x1)*torch.cos(x2) - t*torch.sin(x1)*torch.sin(x2)).view(-1, )


def solution(X):
	t, x1, x2= X[:, 0], X[:, 1], X[:, 2]
	return (t*torch.sin(x1)*torch.cos(x2)).view(-1, )


def I(X):
	t, x1, x2= X[:, 0], X[:, 1], X[:, 2]
	I1 = t**3/3
	I2 = -torch.cos(x1+1)
	I3 = torch.sin(x2)
	return (I1*I2*I3).view(-1, )

def g(X):
	return (f(X) - solution(X) - I(X)).view(-1, )

def sample_boundary_points(dim=3, num_samples_per_boundary=1000, device="cpu", dtype=torch.float32, bound=False):
	"""
	生成 [0,1]^dim 每个边界上的随机边界点。
	返回形状为 (dim * 2, num_samples_per_boundary, dim) 的张量。如果bound=True，则把第一个维度舍去
	"""
	boundaries = []

	for i in range(dim): # 遍历每个维度
		for fixed_val in [0.0, 1.0]:
			# 生成随机点
			points = torch.rand(num_samples_per_boundary, dim, dtype=dtype)
			points[:, i] = fixed_val
			boundaries.append(points)

	# 组合所有边界点
	boundary_tensor = torch.stack(boundaries).to(device)

	# 改为2维的张量
	if bound:
		boundary_tensor = boundary_tensor.view(-1, dim)
	return boundary_tensor

def in_mean(model, x_i, ksi):
	N_x_i = x_i.size(0)
	N_ksi = ksi.size(0)
	x_i_expanded = x_i.unsqueeze(1).expand(-1, N_ksi, -1)
	ksi_expanded = ksi.unsqueeze(0).expand(N_x_i, -1, -1)
	x_i_ksi = x_i_expanded * ksi_expanded # torch.Size([10000, 10, 10])
	Imean = torch.mean(model(x_i_ksi.flatten(0, 1)).view(N_x_i, N_ksi, ), dim=1).view(-1, )
	product = (x_i[:, 0] * torch.prod(x_i, dim=-1)).view(-1, )
	Int = model(x_i).view(-1, ) + g(x_i) + product * Imean - f(x_i)
	return Int.view(-1, )

def loss_ph(model, x, ksi):
	# loss = torch.sum(torch.abs(in_mean(model, x_i, ksi1) * in_mean(model, x_i, ksi2)))/2
	loss = torch.mean(in_mean(model, x, ksi) ** 2)
	return loss

def loss_bc(model, x):
	x = x.detach().requires_grad_(True)
	u = model(x)
	gradients = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
	sum_of_partials = torch.sum(gradients, dim=1).view(-1, )  # 沿特征维度求和
	# loss = torch.sum((sum_of_partials - f(x))**2)/2
	loss = torch.mean((sum_of_partials - f(x)) ** 2)
	return loss

def loss_fn(model, x, ksi):
	loss1 = loss_ph(model, x, ksi)
	loss2 = loss_bc(model, x)
	minloss = torch.min(loss1, loss2) + 1e-16
	# print(loss1, loss2)
	return loss1 ** 2 /minloss + loss2 ** 2 /minloss
	# return loss1 + loss2


def train(model, x, ksi, epochs=1000, lr=1e-2, epsilon = 1e-20, show_iter=200):
	lr0 = lr
	optimizer = torch.optim.Adam(model.parameters(), lr=lr)
	time0 = time.time()
	for epoch in range(epochs):
		optimizer.zero_grad()
		loss = loss_fn(model, x, ksi)
		loss.backward()
		optimizer.step()

		if loss.item() < 1e5 * lr:
			lr *= 0.95
			for param_group in optimizer.param_groups:
				param_group['lr'] = lr
		elif loss.item() > 1e7 * lr:
			lr = lr0
			for param_group in optimizer.param_groups:
				param_group['lr'] = lr

		if epoch % show_iter == 0:
			time1 = time.time()
			print(f"Epoch {epoch}, Loss: {loss.item()}, Learning Rate:{lr}, Speed: {show_iter/(time1-time0)} iter/s")
			time0 = time1

		if loss.item() <= epsilon:
			break


def train_LBFGS(model, x, ksi, epochs=5000, lr=1e-3, epsilon = 1e-20, show_iter=200):
	time0 = time.time()
	optimizer = torch.optim.LBFGS(model.parameters(), lr=lr)

	def closure():
		optimizer.zero_grad()
		loss = loss_fn(model, x, ksi)
		loss.backward()
		return loss

	for epoch in range(epochs):
		optimizer.step(closure)
		if epoch % show_iter == 0:
			loss = closure()
			time1 = time.time()
			print(f"Epoch {epoch}, Loss: {loss.item()}, Speed: {1000*(time1-time0)/show_iter} ms/iter")
			time0 = time1
			if loss.item() < epsilon:
				break


def generate_full_grid_torch(dim=3, points_per_dim=2, device='cpu', dtype=torch.float32):
	"""
	生成 [0,1]^dim 的均匀网格（PyTorch实现）
	:param dtype: 数据类型
	:param dim: 维度
	:param points_per_dim: 每维的点数
	:param device: 设备 ('cpu' 或 'cuda')
	:return: 网格点张量，形状为 (points_per_dim^dim, dim)
	"""
	# 生成每维的坐标值（等间距）
	axis_values = torch.linspace(0, 1, points_per_dim, device=device, dtype=dtype)

	# 生成笛卡尔积
	grid = torch.stack([
		torch.stack(tensors, dim=-1).view(-1)
		for tensors in product(*[axis_values] * dim)
	]).reshape(-1, dim)

	return grid

def generate_grid_with_specific_dims(fixed_values, variable_dims_indices=None, n_points=100, dtype=torch.float32):
	"""
	生成 [0,1]^10 网格，指定某些维度为可变，其余固定

	参数:
		fixed_values: 长度为1的列表/张量，表示固定维度的值（按非可变维度的顺序）
		variable_dims_indices: 可变维度的索引（从0开始计数，例如第3和第5维是[2,4]）
		n_points: 每个可变维度的网格点数

	返回:
		grid: 形状为 (n_points^len(variable_dims_indices), 10) 的张量
	"""
	# 转换为张量
	if variable_dims_indices is None:
		variable_dims_indices = [1, 2]
	fixed = torch.tensor(fixed_values, dtype=dtype)
	n_variable = len(variable_dims_indices)

	# 生成可变维度的网格点
	grid_1d = torch.linspace(0, 1, n_points, dtype=dtype)
	grids = torch.meshgrid(*([grid_1d] * n_variable), indexing='ij')
	variable_values = torch.stack([g.flatten() for g in grids], dim=1)  # (n_points^n_variable, n_variable)

	# 构建完整网格
	n_total_points = variable_values.shape[0]
	grid = torch.zeros(n_total_points, 3)

	# 填充固定维度
	fixed_mask = torch.ones(3, dtype=bool)
	fixed_mask[variable_dims_indices] = False
	grid[:, fixed_mask] = fixed.repeat(n_total_points, 1)

	# 填充可变维度
	grid[:, variable_dims_indices] = variable_values

	return grid

if __name__ == '__main__':
	device = torch.device("cuda:0")
	dtype = torch.float64
	x_b = sample_boundary_points(dim=3, num_samples_per_boundary=100, device=device, dtype=dtype,
								 bound=True).requires_grad_(True)  # torch.Size([20000, 10])
	x_i = torch.rand(1000, 3, device=device, dtype=dtype)  # torch.Size([10000, 10])
	x = torch.cat((x_i, x_b), dim=0)
	ksi = torch.rand(100, 3, device=device, dtype=dtype)
	print(loss_ph(solution, x, ksi), loss_bc(solution, x))