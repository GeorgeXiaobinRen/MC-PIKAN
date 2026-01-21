import torch


# Volterra1D functions
def Volterra1D_solution(x):
	return torch.sin(torch.pi * x)


def Volterra1D_f(x):
	return (1 + 1 / (2 * torch.pi)) * torch.sin(torch.pi * x) - x * torch.cos(torch.pi * x) / 2


def Volterra1D_K(x, s):
	return -torch.sin(torch.pi * (x - s))


# Volterra2DNR functions
def Volterra2DNR_f(X):
	x, t = X[:, 0], X[:, 1]
	u_xt = x ** 2 + 2 * x * t
	I1 = -441 * t ** 7 / 2000000 + t ** 6 * (
			-2153 * t / 9000000 - 2153 * x / 9000000) + t ** 4 * x ** 3 / 3 + t ** 3 * (
				 4 * t * x ** 3 / 9 + 10 * x ** 4 / 9) + t ** 2 * (t * x ** 4 / 2 + x ** 5) + t * (
				 t * x ** 5 / 5 + 11 * x ** 6 / 30)
	I2 = 47 * t * x / 7200 + t * x * torch.e ** 2 / 100 + (
				72 * t * x + 64) * torch.e ** 3 / 81000 + 12871 / 45360000 + torch.e ** 4 / 16000
	y = u_xt + I1 + I2
	return y


def Volterra2DNR_solution(X):
	return X[:, 0] ** 2 + 2 * X[:, 0] * X[:, 1]


# Volterra10D functions
def Volterra10D_f(X):
	t = X[:, 0]
	x123 = X[:, 1] + X[:, 2] + X[:, 3]
	x456 = X[:, 4] + X[:, 5] + X[:, 6]
	x789 = X[:, 7] + X[:, 8] + X[:, 9]
	return (x123 * torch.sin(x456) * torch.cos(x789) + 3 * t * torch.sin(x456) * torch.cos(
		x789) + 3 * t * x123 * torch.cos(x456) * torch.cos(x789) - 3 * t * x123 * torch.sin(x456) * torch.sin(
		x789)).view(-1, )


def Volterra10D_solution(X):
	return (X[:, 0] * (X[:, 1] + X[:, 2] + X[:, 3]) * torch.sin(X[:, 4] + X[:, 5] + X[:, 6]) * torch.cos(
		X[:, 7] + X[:, 8] + X[:, 9])).view(-1, )


def Volterra10D_integral(X):
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