import time
import matplotlib.pyplot as plt
import torch
from random_seed import setup_seed
from torchinfo import summary
from problems.Volterra import Volterra2DNR
from train import train
from models import *

if __name__ == "__main__":
	plt.rcParams['axes.unicode_minus'] = False
	setup_seed(233) # set up the random seed
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	dtype = torch.bfloat16
	n_s = 1000
	n_e = 10000

	x_grid = torch.linspace(0, 1, 30, device=device, dtype=dtype)
	X = torch.cartesian_prod(x_grid, x_grid)
	s = torch.rand(n_s, 2, device=device, dtype=dtype)
	question = Volterra2DNR(X_grid=X, s=s)
	# def f(X):
	# 	x, t = X[:, 0], X[:, 1]
	# 	return -17*t**3/200 + 29*t**2*x/20 + 3*t*x**2/2 - t*x/4 + torch.tensor(torch.e, dtype=X.dtype, device=X.device)*t*x/5 + torch.exp(torch.tensor(2, dtype=X.dtype, device=X.device))/200 + 497/500
	#
	#
	# def solution(X):
	# 	return torch.tensor(1, dtype=X.dtype, device=X.device)
	#
	#
	# question.solution = solution
	# question.f = f

	# build model
	m = 1
	if m==0:
		model = PINNModel(num_layers=3, input_dim=2, hidden_dim=10, dtype=dtype).to(device)
	elif m==1:
		model = CPIKANModel(num_layers=2, input_dim=2, hidden_dim=6, degree=3, dtype=dtype).to(device)
	else:
		raise ValueError("Invalid value: m should be 0 or 1.")
	print_modelsize(model)
	print("Model on device:", device)

	# train model
	time0 = time.time()
	train(model, question, lr=0.1, max_epochs=n_e, dynamic_lr=True)
	time1 = time.time()
	print("Training time: " + str(time1 - time0) + "s")
	# prediction result
	with torch.no_grad():
		x_pred = torch.linspace(0, 1, 30, device=device, dtype=dtype)
		X_pred = torch.cartesian_prod(x_grid, x_grid)
		u_pred = model(X_pred)
		u_solution = question.solution(X_pred).view(-1, 1)
		residual = torch.abs(u_solution - u_pred)
		print(torch.mean(residual))
