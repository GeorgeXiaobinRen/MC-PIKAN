import torch


def Volterra1D_solution(x):
	return torch.sin(torch.pi * x)


def Volterra1D_f(x):
	return (1 + 1 / (2 * torch.pi)) * torch.sin(torch.pi * x) - x * torch.cos(torch.pi * x) / 2


def Volterra1D_K(x, s):
	return -torch.sin(torch.pi * (x - s))


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