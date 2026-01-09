import torch

class Volterra_2D_NR:
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


	def f(self, X):
		x, t = X[:, 0], X[:, 1]
		y = -331 * t ** 7 / 720000 - 2153 * t ** 6 * x / 9000000 + 7 * t ** 4 * x ** 3 / 9 + 29 * t ** 3 * x ** 4 / 18 + 6 * t ** 2 * x ** 5 / 5 + 11 * t * x ** 6 / 30 + t * x * torch.exp(
			torch.tensor(3, dtype=X.dtype)) / 1125 + t * x * torch.exp(
			torch.tensor(2, dtype=X.dtype)) / 100 + 14447 * t * x / 7200 + x ** 2 + 12871 / 45360000 + torch.exp(
			torch.tensor(4, dtype=X.dtype)) / 16000 + 8 * torch.exp(torch.tensor(3, dtype=X.dtype)) / 10125
		return y


	def solution(self, X):
		pass

	def I1(self, u):
		eta = self.s2 * self.x + self.s1 * self.t / 10 * (1 - self.s2)
		xi = self.s1 * self.t
		X_stacked = torch.stack([eta, xi], dim=-1)
		F1 = (self.x + self.t + eta + xi) * (u(X_stacked.flatten(0, 1)).view(self.N_X, self.N_s, )) ** 2
		I_result = self.t * (self.x - self.s1 * self.t / 10) * F1
		return I_result

	def I2(self, u):
		eta = (1 - self.s1) * self.s2 / 10 + self.s1 * torch.exp(self.s2) / 5
		xi = self.s2
		X_stacked = torch.stack([eta, xi], dim=-1)
		F2 = (self.x * self.t + eta * xi ** 2) * (u(X_stacked.flatten(0, 1)).view(self.N_X, self.N_s, )) ** 2
		I_result = (torch.exp(self.s2) / 5 - self.s2 / 10) * F2
		return I_result


	def loss_ph(self, model):
		pass


if __name__ == "__main__":
	class SimpleModel(torch.nn.Module):
		def forward(self, x):
			return torch.sum(x, dim=-1, keepdim=True)


	def f(x):
		return x

	model = SimpleModel()

	X = torch.tensor([[0.5, 0.4], [0.2, 0.8], [0.2, 0.8], [0.2, 0.8], [0.2, 0.8]], dtype=torch.float32)
	s = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.3, 0.4]], dtype=torch.float32)

	Q = Volterra_2D_NR(X, s)

	result = Q.I2(model)
	print(f"F函数返回值: {result}")

	Q.f = f
	result = Q.f(X)
	print(result)