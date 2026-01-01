import torch

def f(x, tau):
	y = -331 * tau ** 7 / 720000 - 2153 * tau ** 6 * x / 9000000 + 7 * tau ** 4 * x ** 3 / 9 + 29 * tau ** 3 * x ** 4 / 18 + 6 * tau ** 2 * x ** 5 / 5 + 11 * tau * x ** 6 / 30 + tau * x * torch.exp(torch.tensor(3)) / 1125 + tau * x * torch.exp(2) / 100 + 14447 * tau * x / 7200 + x ** 2 + 12871 / 45360000 + torch.exp(4) / 16000 + 8 * torch.exp(3) / 10125
	return y

print(f(torch.tensor(0.0), torch.tensor(0.0)))
