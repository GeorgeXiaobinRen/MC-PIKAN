from problems.functions import *


class Volterratype:
	def __init__(self, X_grid, s):
		self.X_grid = X_grid
		self.s = s
		self.N_X = X_grid.size(0)
		self.N_s = s.size(0)


class Volterra1D(Volterratype):
	"""
    One-dimensional Volterra integral equation solver class

    Implements numerical solution of the first kind Volterra integral equation:
    u(x) = f(x) + ∫₀ˣ K(x,ξ)u(ξ)dξ, where 0 ≤ x ≤ 1
    ($$u(x)=f(x)+\int_0^xK(x,\xi)u(\xi)\mathrm{d}\xi,\quad0\leq x\leq1.$$)

    This class handles one-dimensional Volterra integral equations with kernel function K(x,s),
    providing loss function calculation and replaceable core function components.

    Attributes:
        x_e (torch.Tensor): Extended grid point tensor, shape (N_X, N_s)
        s_e (torch.Tensor): Extended integration variable tensor, shape (N_X, N_s)
        K (callable): Kernel function K(x,s), can be customized externally
        f (callable): Free term function f(x), can be customized externally
        solution (callable): Exact solution function, can be customized externally

    Args:
        X_grid (torch.Tensor): Grid points of the computational domain
        s (torch.Tensor): Integration variable sampling points
        K (callable, optional): Kernel function, uses default if None
        f (callable, optional): Free term function, uses default if None
        solution (callable, optional): Exact solution function, uses default if None
    """
	def __init__(self, X_grid, s, K=None, f=None, solution=None):
		super().__init__(X_grid, s)
		self.x_e = self.X_grid.view(-1, 1).expand(-1, self.N_s)
		self.s_e = self.s.view(1, -1).expand(self.N_X, -1)

		self.K = K if K is not None else Volterra1D_K
		self.f = f if f is not None else Volterra1D_f
		self.solution = solution if solution is not None else Volterra1D_solution

	def loss_fn(self, model):
		xs = self.x_e * self.s_e
		k_vals = self.K(self.x_e, xs)  # (N_x, N_s)
		u_vals = model(xs.view(-1, 1)).view(self.N_X, self.N_s)
		product = self.x_e * k_vals * u_vals  # (N_x, N_s)
		inner_mean = model(self.X_grid.view(-1, 1)).view(-1, ) - self.f(self.X_grid) - torch.mean(product, dim=1)  # (N_x,)
		loss = torch.mean(inner_mean ** 2)
		return loss


class Volterra2DNR(Volterratype):
	"""
    Two-dimensional Nonlinear Volterra integral equation solver class

    Implements numerical solution of a two-dimensional nonlinear Volterra integral equation:
    The general form is
    $$u(x, t)+\int_0^t\int_{\xi/10}^x\left(x+t+\eta+\xi\right)u^2(\eta, \xi)\mathrm{d}\eta\mathrm{d}\xi+\int_\Omega\left(xt+\eta\xi^2\right)u(\eta,\xi)\mathrm{d}\eta\mathrm{d}\xi=f(x, t)$$
    on the region
	$$\Omega=\left\{(\eta,\xi)\in\mathbb{R}^2:0<\xi<1,\frac\xi{10}<\eta<\frac15e^\xi\right\}.$$

    This class handles two-dimensional nonlinear Volterra integral equations with specific kernel structures,
    providing loss function calculation for neural network training approaches.

    Attributes:
        x (torch.Tensor): First spatial dimension extended grid points, shape (N_X, N_s)
        t (torch.Tensor): Second spatial dimension (time) extended grid points, shape (N_X, N_s)
        s1 (torch.Tensor): First integration variable tensor, shape (N_X, N_s)
        s2 (torch.Tensor): Second integration variable tensor, shape (N_X, N_s)
        f (callable): Free term function f(x,t), can be customized externally
        solution (callable): Exact solution function, can be customized externally

    Args:
        X_grid (torch.Tensor): Grid points of the computational domain, shape (N_X, 2) where second dimension represents [x, t]
        s (torch.Tensor): Integration variable sampling points, shape (N_s, 2) where second dimension represents [s1, s2]
        f (callable, optional): Free term function, uses default if None
        solution (callable, optional): Exact solution function, uses default if None
    """
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
		"""
        Compute the loss function for the 2D nonlinear Volterra equation.
        
        The loss is calculated based on the residual of the integral equation:
        u(X_grid) + mean(I1 + I2) - f(X_grid), where I1 and I2 are double integrals
        with different kernel functions and variable transformations.
        
        Args:
            u (callable): Neural network approximation of the solution function
            
        Returns:
            torch.Tensor: Scalar value representing the mean squared residual
        """
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
	def __init__(self, X_grid, s):
		super().__init__(X_grid, s)
		self.X_e = X_grid.unsqueeze(1).expand(-1, self.N_s, -1)
		self.s_e = s.unsqueeze(0).expand(self.N_X, -1, -1)

		def f(X):
			t = X[:, 0]
			x123 = X[:, 1] + X[:, 2] + X[:, 3]
			x456 = X[:, 4] + X[:, 5] + X[:, 6]
			x789 = X[:, 7] + X[:, 8] + X[:, 9]
			return (x123 * torch.sin(x456) * torch.cos(x789) + 3 * t * torch.sin(x456) * torch.cos(
				x789) + 3 * t * x123 * torch.cos(x456) * torch.cos(x789) - 3 * t * x123 * torch.sin(x456) * torch.sin(
				x789)).view(-1, )
		self.f = f

		def solution(X):
			return (X[:, 0] * (X[:, 1] + X[:, 2] + X[:, 3]) * torch.sin(X[:, 4] + X[:, 5] + X[:, 6]) * torch.cos(
				X[:, 7] + X[:, 8] + X[:, 9])).view(-1, )
		self.solution = solution

		def integral(X):
			def cos(x):
				return torch.cos(x)
			def sin(x):
				return torch.sin(x)

			t, x1, x2, x3, x4, x5, x6, x7, x8, x9 = X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4], X[:, 5], X[:, 6], X[
				:, 7], X[:, 8], X[:, 9]
			I1 = t ** 3 / 3
			I2 = x1 * x2 * x3 * (x1 + x2 + x3) / 2
			I3 = cos(x4 + x5 + x6) - cos(x4 + x5) - cos(x4 + x6) - cos(x5 + x6) + cos(x4) + cos(x5) + cos(x6) - 1
			I4 = -sin(x7 + x8 + x9) + sin(x7 + x8) + sin(x7 + x9) + sin(x8 + x9) - sin(x7) - sin(x8) - sin(x9)
			return (I1 * I2 * I3 * I4).view(-1, )
		self.integral = integral

		def g(X):
			return (self.f(X) - self.solution(X) - self.integral(X)).view(-1, )
		self.g = g
		"""
        Notice that you don't have to redefine self.g(x) if you redefine self.f(x), self.solution(x) or self.g(x).
        """

	def in_mean(self, u):
		X_s = self.X_e * self.s_e  # torch.Size([N_X, N_s, dim])
		Imean = torch.mean(u(X_s.flatten(0, 1)).view(self.N_X, self.N_s, ), dim=1).view(-1, )
		product = (self.X_grid[:, 0] * torch.prod(self.X_grid, dim=-1)).view(-1, )
		Int = u(self.X_grid).view(-1, ) + self.g(self.X_grid) + product * Imean - self.f(self.X_grid)
		return Int.view(-1, )

	def loss_ph(self, u):
		loss = torch.mean(self.in_mean(u) ** 2)
		return loss

	def loss_bc(self, u):
		x = self.X_grid.detach().requires_grad_(True)
		gradients = torch.autograd.grad(u(x).sum(), x, create_graph=True)[0]
		sum_of_partials = torch.sum(gradients, dim=1).view(-1, )
		loss = torch.mean((sum_of_partials - self.f(x).view(-1, )) ** 2)
		return loss

	def loss_fn(self, model):
		loss1 = self.loss_ph(model)
		loss2 = self.loss_bc(model)
		minloss = torch.min(loss1, loss2) + 1e-16
		return loss1 ** 2 / minloss + loss2 ** 2 / minloss


if __name__ == "__main__":
	x = torch.linspace(0, 1, 30)
	X = torch.cartesian_prod(x, x)
	s = torch.rand(10000, 2)

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
	s = torch.rand(500)

	question = Volterra1D(X, s)


	result = question.loss_fn(question.solution)
	print(f"F函数返回值: {result}")
