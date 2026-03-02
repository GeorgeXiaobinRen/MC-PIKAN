import time, os, sys
# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt
from random_seed import setup_seed
from torchinfo import summary
from problems.Volterra import Volterra2DNR
from train import train
from models import *

if __name__ == "__main__":
	plt.rcParams['axes.unicode_minus'] = False
	setup_seed(233) # set up the random seed
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	dtype = torch.float32
	n_s = 2000
	n_e = 10000

	x_grid = torch.linspace(0, 1, 30, device=device, dtype=dtype)
	X = torch.cartesian_prod(x_grid, x_grid)
	s = torch.rand(n_s, 2, device=device, dtype=dtype)
	question = Volterra2DNR(X_grid=X, s=s)
	result = question.loss_fn(question.solution)
	print(f"loss function return value: {result}")

	# build model
	m = 1
	if m==0:
		model = PINNModel(num_layers=2, input_dim=2, hidden_dim=8, dtype=dtype).to(device)
	elif m==1:
		model = CPIKANModel(num_layers=2, input_dim=2, hidden_dim=4, degree=3, dtype=dtype).to(device)
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
		x_pred = torch.linspace(0, 1, 1000, device=device, dtype=dtype)
		X_pred = torch.cartesian_prod(x_pred, x_pred)
		u_pred = model(X_pred)
		u_solution = question.solution(X_pred).view(-1, 1)
		residual = torch.abs(u_solution - u_pred)
		print(torch.mean(residual))
