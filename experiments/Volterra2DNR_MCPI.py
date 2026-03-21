import time
import numpy as np

import matplotlib.pyplot as plt
from random_seed import setup_seed
from torchinfo import summary
from problems.Volterra import Volterra2DNR
from train import train
from models import *

if __name__ == "__main__":
	plt.rcParams['axes.unicode_minus'] = False

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	dtype = torch.float32
	n_s = 40
	n_e = 4000

	n_xi = 10
	n_eta = 10
	def create_X(n_xi, n_eta):
		xi_grid = torch.linspace(0, 1, n_xi, device=device, dtype=dtype)
		eta_min = xi_grid / 10
		eta_max = torch.exp(xi_grid) / 5
		eta_normalized = torch.linspace(0, 1, n_eta, device=device, dtype=dtype).view(-1, 1)
		eta_coords = eta_min + eta_normalized * (eta_max - eta_min)
		X = torch.stack([
			eta_coords.reshape(-1),
			xi_grid.repeat(n_eta)
		], dim=-1)
		return X

	X = create_X(n_xi, n_eta)

	setup_seed(19)  # set up the random seed
	s = torch.rand(n_s, 2, device=device, dtype=dtype)
	question = Volterra2DNR(X_grid=X, s=s)
	# result = question.loss_fn(question.solution)
	# print(f"loss function return value: {result}")


	# build model
	m = 1
	if m==0:
		model = PINNModel(num_layers=3, input_dim=2, hidden_dim=20, dtype=dtype).to(device)
	elif m==1:
		model = CPIKANModel(num_layers=3, input_dim=2, hidden_dim=10, degree=3, dtype=dtype).to(device)
	else:
		raise ValueError("Invalid value: m should be 0 or 1.")
	print_modelsize(model)
	print("Model on device:", device)

	# train model
	time0 = time.time()
	train(model, question, lr=0.001, max_epochs=n_e, dynamic_lr=False)
	time1 = time.time()
	print("Training time: " + str(time1 - time0) + "s")
	# prediction result
	with torch.no_grad():
		X_pred=create_X(500, 500)
		u_pred = model(X_pred)
		u_solution = question.solution(X_pred).view(-1, 1)
		residual = torch.abs(u_solution - u_pred)
		# Save X_pred, u_pred, and u_solution to a .npy file
		data_to_save = {
			'X_pred': X_pred.cpu().numpy(),
			'u_pred': u_pred.cpu().numpy(),
			'u_solution': u_solution.cpu().numpy()
		}
		np.save('../results/data/2DNR_PIKAN[3,10,3]S40results.npy', data_to_save, allow_pickle=True)
		print(torch.mean(residual))
