from problems.functions import *


class Volterratype:
	def __init__(self, X_grid, s):
		self.X_grid = X_grid
		self.s = s
		self.N_X = X_grid.size(0)
		self.N_s = s.size(0)


class Volterra1D(Volterratype):
	def __init__(self, X_grid, s, K=None, f=None, solution=None):
		super().__init__(X_grid, s)
		self.x_e = self.X_grid.view(-1, 1).expand(-1, self.N_s)
		self.s_e = self.s.view(1, -1).expand(self.N_X, -1)

		self.K = K if K is not None else Volterra1D_K
		self.f = f if f is not None else Volterra1D_f
		self.solution = solution if solution is not None else Volterra1D_solution

	def loss_fn(self, u):
		xs = self.x_e * self.s_e
		k_vals = self.K(self.x_e, xs)  # (N_x, N_s)
		u_vals = u(xs.view(-1, 1)).view(self.N_X, self.N_s)
		product = self.x_e * k_vals * u_vals  # (N_x, N_s)
		inner_mean = u(self.X_grid.view(-1, 1)).view(-1, ) - self.f(self.X_grid) - torch.mean(product, dim=1)  # (N_x,)
		loss = torch.mean(inner_mean ** 2)
		return loss


class Volterra2DNR(Volterratype):
	def __init__(self, X_grid, s, f=None, solution=None):
		super().__init__(X_grid, s)
		X_e = X_grid.unsqueeze(1).expand(-1, self.N_s, -1)
		self.x = X_e[:, :, 0]
		self.t = X_e[:, :, 1]
		s_e = s.unsqueeze(0).expand(self.N_X, -1, -1)
		self.s1 = s_e[:, :, 0]
		self.s2 = s_e[:, :, 1]

		self.f = f if f is not None else Volterra2DNR_f
		self.solution = solution if solution is not None else Volterra2DNR_solution

	def loss_fn(self, u):
		eta1 = self.s1 * self.x + self.s2 * self.t / 10 * (1 - self.s1)
		xi1 = self.s2 * self.t
		X_stacked1 = torch.stack([eta1, xi1], dim=-1)
		F1 = (self.x + self.t + eta1 + xi1) * (u(X_stacked1.flatten(0, 1)).view(self.N_X, self.N_s, )) ** 2
		I1 = self.t * (self.x - self.s2 * self.t / 10) * F1
		eta2 = (1 - self.s2) * self.s1 / 10 + self.s2 * torch.exp(self.s1) / 5
		xi2 = self.s1
		X_stacked2 = torch.stack([eta2, xi2], dim=-1)
		F2 = (self.x * self.t + eta2 * xi2 ** 2) * (u(X_stacked2.flatten(0, 1)).view(self.N_X, self.N_s, ))
		I2 = (torch.exp(self.s1) / 5 - self.s1 / 10) * F2
		in_mean = u(self.X_grid).view(-1, ) + torch.mean(I1 + I2, dim=1) - self.f(self.X_grid)
		loss = torch.mean(in_mean ** 2)
		return loss


class Volterra10D(Volterratype):
	def __init__(self, X_grid, X_boundary, s, f=None, solution=None, integral=None):
		super().__init__(X_grid, s)
		self.X_boundary = X_boundary
		self.X_e = X_grid.unsqueeze(1).expand(-1, self.N_s, -1)
		self.s_e = s.unsqueeze(0).expand(self.N_X, -1, -1)
		self.f = f if f is not None else Volterra10D_f
		self.solution = solution if solution is not None else Volterra10D_solution
		self.integral = integral if integral is not None else Volterra10D_integral

		def g(X):
			return (self.f(X) - self.solution(X) - self.integral(X)).view(-1, )
		self.g = g

	def loss_fn(self, u, mode = "adaptive"):
		X_dot_s = self.X_e * self.s_e  # torch.Size([N_X, N_s, dim])
		I = self.s_e[:, :, 0] * self.X_e[:, :, 0] * torch.prod(self.X_e[:, :, :], dim=-1) * (u(X_dot_s.flatten(0, 1)).view(self.N_X, self.N_s, ))  # torch.Size([N_X, N_s])
		in_mean = self.f(self.X_grid) - u(self.X_grid).view(-1, ) - self.g(self.X_grid) - torch.mean(I, dim=1)
		loss_ph = torch.mean(in_mean ** 2)

		x_withgrad = self.X_boundary.detach().requires_grad_(True)
		gradients = torch.autograd.grad(u(x_withgrad).sum(), x_withgrad, create_graph=True)[0]
		sum_of_partials = torch.sum(gradients, dim=1).view(-1, )
		loss_bc = torch.mean((sum_of_partials - self.f(self.X_boundary).view(-1, )) ** 2)

		# """
		# The functionality for selecting weights in the loss function should be improved in the future
		# """
		if mode == "adaptive":
			# Use a more stable adaptive weighting (e.g., balancing based on log scale or normalized values)
			# Here we use a simpler but more robust normalization
			w_ph = loss_ph.detach() / (loss_ph.detach() + loss_bc.detach() + 1e-16)
			w_bc = loss_bc.detach() / (loss_ph.detach() + loss_bc.detach() + 1e-16)
			return (1.0 - w_ph) * loss_ph + (1.0 - w_bc) * loss_bc
		else:
			return loss_ph + loss_bc


if __name__ == "__main__":
	# x = torch.linspace(0, 1, 30)
	# X = torch.cartesian_prod(x, x)
	# s = torch.rand(10000, 2)
	#
	# question = Volterra2DNR(X, s)
	#
	# result = question.loss_fn(question.solution)
	# print(f"F函数返回值: {result}")
	#
	# X = torch.linspace(0, 1, 100)
	# s = torch.rand(500)
	#
	# question = Volterra1D(X, s)
	#
	#
	# result = question.loss_fn(question.solution)
	# print(f"F函数返回值: {result}")

	from utils import *
	device = torch.device("cuda:0")
	x_b = sample_boundary_points(dim=10, num_samples_per_boundary=100, device=device,
									 bound=True).requires_grad_(True)  # torch.Size([20000, 10])
	x_i = torch.rand(1000, 10, device=device)  # torch.Size([10000, 10])
	s = torch.rand(2000, 10, device=device)
	question = Volterra10D(x_i, x_b, s)

	print(question.loss_fn(question.solution))
