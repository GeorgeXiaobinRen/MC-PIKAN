import time
import torch


def K(x, s):
	# return 1
	return -torch.sin(torch.pi * (x - s))

def f(x):
	# return 1
	return (1 + 1/(2*torch.pi)) * torch.sin(torch.pi*x) - x * torch.cos(torch.pi*x)/2

def solution(x):
	return torch.sin(torch.pi*x)

# 损失函数部分
def in_mean(model, x, s):
	N_x = x.size(0)
	N_s = s.size(0)

	x_expanded = x.view(-1, 1).expand(-1, N_s)  # (N_x, N_s)
	s_expanded = s.view(1, -1).expand(N_x, -1)  # (N_x, N_s)
	x_s = x_expanded * s_expanded  # (N_x, N_s)

	k_vals = K(x_expanded, x_s)  # (N_x, N_s)
	u_vals = model(x_s.view(-1, 1)).view(N_x, N_s)

	product = x_expanded * k_vals * u_vals  # (N_x, N_s)
	inner_mean = model(x.view(-1, 1)).view(-1, ) - f(x) - torch.mean(product, dim=1)  # (N_x,)
	return inner_mean

def loss_fn(model, x, s):
	s0 = s[0]
	loss_mc = torch.mean(torch.abs(in_mean(model, x, s0) **2))
	return loss_mc
