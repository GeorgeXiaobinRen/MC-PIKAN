import torch


def f(X):
	x, tau = X[:, 0], X[:, 1]
	y = -331 * tau ** 7 / 720000 - 2153 * tau ** 6 * x / 9000000 + 7 * tau ** 4 * x ** 3 / 9 + 29 * tau ** 3 * x ** 4 / 18 + 6 * tau ** 2 * x ** 5 / 5 + 11 * tau * x ** 6 / 30 + tau * x * torch.exp(torch.tensor(3, dtype=x.dtype)) / 1125 + tau * x * torch.exp(torch.tensor(2, dtype=x.dtype)) / 100 + 14447 * tau * x / 7200 + x ** 2 + 12871 / 45360000 + torch.exp(torch.tensor(4, dtype=x.dtype)) / 16000 + 8 * torch.exp(torch.tensor(3, dtype=x.dtype)) / 10125
	return y

def solution(X):
	return X

def F(u, X, s):
	N_X = X.size(0)
	N_s = s.size(0)
	X_e = X.unsqueeze(1).expand(-1, N_s, -1)
	x = X_e[:, :, 0]
	tau = X_e[:, :, 1]
	s_e = s.unsqueeze(0).expand(N_X, -1, -1)
	s1 = s_e[:, :, 0]
	s2 = s_e[:, :, 1]
	eta = s2*x+s1*tau/10*(1-s2)
	xi = s1*tau
	X_stacked = torch.stack([eta, xi], dim=-1)
	F_result = (x+tau+eta+xi)*(u(X_stacked.flatten(0, 1)).view(N_X, N_s, ))**2
	return F_result


def I1(u, X, s):
	return 0

if __name__ == "__main__":
	class SimpleModel(torch.nn.Module):
		def forward(self, x):
			return torch.sum(x, dim=-1, keepdim=True)

	model = SimpleModel()

	X = torch.tensor([[0.5, 0.5], [0.2, 0.8], [0.2, 0.8], [0.2, 0.8], [0.2, 0.8]], dtype=torch.float32)
	s = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.3, 0.4]], dtype=torch.float32)


	result = F(model, X, s)
	print(f"F函数返回值: {result}")