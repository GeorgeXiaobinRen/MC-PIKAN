import torch


def Volterra1D_solution(x):
	return torch.sin(torch.pi * x)


def Volterra1D_f(x):
	return (1 + 1 / (2 * torch.pi)) * torch.sin(torch.pi * x) - x * torch.cos(torch.pi * x) / 2


def Volterra1D_K(x, s):
	return -torch.sin(torch.pi * (x - s))
