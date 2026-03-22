import time, os
import numpy as np
from utils import *
from random_seed import setup_seed
from problems.Volterra import Volterra10D
from models import CPIKANModel, PINNModel
import matplotlib.pyplot as plt
from train import train

if __name__ == "__main__":
	plt.rcParams['axes.unicode_minus'] = False
	device = torch.device("cuda:0")
	dtype = torch.float32
	m = 1 # eval(input())
	n_ksi = 10
	n_in = 10000
	n_bound = 1000
	if m == 0:
		model = PINNModel(input_dim=10, hidden_dim=20, dtype=dtype).to(device)
	elif m == 1:
		model = CPIKANModel(input_dim=10, num_layers=3, hidden_dim=10, degree=7, dtype=dtype).to(device)
	else:
		raise ValueError("Invalid value: m should be 0 or 1.")

	x_b = sample_boundary_points(dim=10, num_samples_per_boundary=n_bound, device=device, dtype=dtype, bound=True).requires_grad_(True) # torch.Size([20000, 10])
	x_i = torch.rand(n_in, 10, device=device, dtype=dtype) # torch.Size([10000, 10])

	setup_seed(0)
	s = torch.rand(n_ksi, 10, device=device, dtype=dtype)
	problem = Volterra10D(x_i, x_b, s)
	ls = problem.loss_fn(problem.solution)

	time0 = time.time()
	train(model, problem, max_epochs=10000, lr=2e-3, opt_method="Adam", dynamic_lr=True)
	train(model, problem, max_epochs=1000, lr=1, opt_method="LBFGS", dynamic_lr=True)
	time1 = time.time()
	print("Training time: " + str(time1 - time0) + "s")
	with torch.no_grad():
		x_pred = torch.rand(500000, 10, device=device, dtype=dtype)
		u_pred = model(x_pred).view(-1, )
		u_solution = problem.solution(x_pred)
		residual = u_solution - u_pred
		relative_err = torch.linalg.norm(residual, ord=2) / torch.linalg.norm(u_solution, ord=2)
		print(relative_err.item())

	dim_data = [
		[[1, 1, 1, 1, 1 ,0 ,1 ,1], [8, 9]],
	    [[1, 1, 0, 0, 0 ,0 ,0 ,0], [4, 5]]]

	data_to_save = []
	for i, data in enumerate(dim_data):
		fixed_values = data[0]
		dims = data[1]
		n = 500

		grid35 = generate_grid_with_specific_dims(fixed_values, dims, n, dtype=dtype).to(device)
		pre_values = model(grid35).view(n, n).detach().cpu().numpy()
		solution_values = problem.solution(grid35).view(n, n).detach().cpu().numpy()
		X = (grid35.view(n, n, -1))[:, :, dims[0]].view(n, n).detach().cpu().numpy()
		Y = (grid35.view(n, n, -1))[:, :, dims[1]].view(n, n).detach().cpu().numpy()

		data_to_save.append({
			'X': X,
			'Y': Y,
			'pre_values': pre_values,
			'solution_values': solution_values,
			'dims': dims
		})

	os.makedirs(r'../results/data', exist_ok=True)
	np.save(f'../results/data/10D_PIKAN[{model.num_layers},{model.hidden_dim},{model.degree}]S{n_ksi}results.npy', data_to_save, allow_pickle=True)
	# np.save(r'../results/data/10D_results.npy', data_to_save, allow_pickle=True)
