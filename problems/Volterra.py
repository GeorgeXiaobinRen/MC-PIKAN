import torch

class Volterra1D:
	def __init__(self, X_grid, s):
		self.X_grid = X_grid
		self.s = s
		self.N_X = X_grid.size(0)
		self.N_s = s.size(0)
		self.x_e = X_grid.view(-1, 1).expand(-1, self.N_s)
		self.s_e = s.view(1, -1).expand(self.N_X, -1)

		def K(x, s):
			return -torch.sin(torch.pi * (x - s))
		self.K = K

		def f(x):
			return (1 + 1 / (2 * torch.pi)) * torch.sin(torch.pi * x) - x * torch.cos(torch.pi * x) / 2
		self.f = f

		def solution(x):
			return torch.sin(torch.pi * x)
		self.solution = solution


	def loss_fn(self, model):
		xs = self.x_e * self.s_e
		k_vals = self.K(self.x_e, xs)  # (N_x, N_s)
		u_vals = model(xs.view(-1, 1)).view(self.N_X, self.N_s)
		product = self.x_e * k_vals * u_vals  # (N_x, N_s)
		inner_mean = model(self.X_grid.view(-1, 1)).view(-1, ) - self.f(self.X_grid) - torch.mean(product, dim=1)  # (N_x,)
		loss = torch.mean(inner_mean ** 2)
		return loss


class Volterra2DNR:
	def __init__(self, X_grid, s):
		self.X_grid = X_grid
		self.s = s
		self.N_X = X_grid.size(0)
		self.N_s = s.size(0)
		X_e = X_grid.unsqueeze(1).expand(-1, self.N_s, -1)
		self.x = X_e[:, :, 0]
		self.t = X_e[:, :, 1]
		s_e = s.unsqueeze(0).expand(self.N_X, -1, -1)
		self.s1 = s_e[:, :, 0]
		self.s2 = s_e[:, :, 1]

		def f(X):
			x, t = X[:, 0], X[:, 1]
			y = -331 * t ** 7 / 720000 - 2153 * t ** 6 * x / 9000000 + 7 * t ** 4 * x ** 3 / 9 + 29 * t ** 3 * x ** 4 / 18 + 6 * t ** 2 * x ** 5 / 5 + 11 * t * x ** 6 / 30 + t * x * torch.exp(
				torch.tensor(3, dtype=X.dtype)) / 1125 + t * x * torch.exp(
				torch.tensor(2, dtype=X.dtype)) / 100 + 14447 * t * x / 7200 + x ** 2 + 12871 / 45360000 + torch.exp(
				torch.tensor(4, dtype=X.dtype)) / 16000 + 8 * torch.exp(torch.tensor(3, dtype=X.dtype)) / 10125
			return y
		self.f = f

		def solution(X):
			return X[:, 0] **2 + 2 * X[:, 0] * X[:, 1]
		self.solution = solution


	def loss_fn(self, u):
		eta1 = self.s2 * self.x + self.s1 * self.t / 10 * (1 - self.s2)
		xi1 = self.s1 * self.t
		X_stacked1 = torch.stack([eta1, xi1], dim=-1)
		F1 = (self.x + self.t + eta1 + xi1) * (u(X_stacked1.flatten(0, 1)).view(self.N_X, self.N_s, )) ** 2
		I1 = self.t * (self.x - self.s1 * self.t / 10) * F1
		eta2 = (1 - self.s1) * self.s2 / 10 + self.s1 * torch.exp(self.s2) / 5
		xi2 = self.s2
		X_stacked2 = torch.stack([eta2, xi2], dim=-1)
		F2 = (self.x * self.t + eta2 * xi2 ** 2) * (u(X_stacked2.flatten(0, 1)).view(self.N_X, self.N_s, )) ** 2
		I2 = (torch.exp(self.s2) / 5 - self.s2 / 10) * F2
		in_mean = u(self.X_grid).view(-1, ) + torch.mean(I1+I2, dim=1) - self.f(self.X_grid)
		loss = torch.mean(in_mean ** 2)
		return loss




if __name__ == "__main__":
	x = torch.linspace(0, 1, 10)
	X = torch.cartesian_prod(x, x)
	s = torch.rand(1000, 2)

	question = Volterra2DNR(X, s)

	# def f(X):
	# 	x, t = X[:, 0], X[:, 1]
	# 	return -17*t**3/200 + 29*t**2*x/20 + 3*t*x**2/2 - t*x/4 + torch.e*t*x/5 + torch.exp(torch.tensor(2, dtype=X.dtype, device=X.device))/200 + 497/500
	#
	# def solution(X):
	# 	return torch.ones_like(X[:, 0])
	#
	#
	# question.solution = solution
	# question.f = f

	result = question.loss_fn(question.solution)
	print(f"F函数返回值: {result}")

	X = torch.linspace(0, 1, 100)
	s = torch.rand(1000)

	question = Volterra1D(X, s)


	result = question.loss_fn(question.solution)
	print(f"F函数返回值: {result}")
