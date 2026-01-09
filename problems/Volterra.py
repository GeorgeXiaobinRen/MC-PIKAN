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
		loss = torch.mean(torch.abs(inner_mean ** 2))
		return loss
