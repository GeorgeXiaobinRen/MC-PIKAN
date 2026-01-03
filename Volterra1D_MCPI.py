import time
import matplotlib.pyplot as plt
from random_seed import setup_seed
from torchinfo import summary
from problems.Volterra_1D import solution, loss_fn
from train import train
from models import *

if __name__ == "__main__":
	plt.rcParams['axes.unicode_minus'] = False
	setup_seed(233) # set up the random seed
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	dtype = torch.float64
	n_s = 200
	n_e = 10000

	x = torch.linspace(0, 1, 80, device=device, dtype=dtype)
	s0 = torch.rand(n_s, device=device, dtype=dtype)
	print(s0.size())
	s = [s0]

	# build model
	m = 1
	if m==0:
		model = PINNModel(num_layers=2, hidden_dim=6, dtype=dtype).to(device)
	elif m==1:
		model = CPIKANModel(num_layers=2, hidden_dim=3, degree=3, dtype=dtype).to(device)
	else:
		raise ValueError("Invalid value: m should be 0 or 1.")
	print_modelsize(model)
	print("Model on device:", device)

	# train model
	time0 = time.time()
	train(model, loss_fn, x, s, lr=0.1, max_epochs=n_e, dynamic_lr=True)
	time1 = time.time()
	print("Training time: " + str(time1 - time0) + "s")
	# prediction result
	with torch.no_grad():
		x_pred = torch.linspace(0, 1, 2000, device=device, dtype=dtype)
		u_pred = model(x_pred.view(-1, 1))
		u_solution = solution(x_pred).view(-1, 1)
		residual = torch.abs(u_solution - u_pred)


	# create canvas and subplots
	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
	plt.rcParams['legend.fontsize'] = 15

	# left subplot: prediction vs ground truth
	ax1.plot(x_pred.detach().cpu(), u_pred.detach().cpu(), label="Exact solution", ls='--')
	ax1.plot(x_pred.detach().cpu(), u_solution.detach().cpu(), label="Predicted solution")
	ax1.set_xlabel("$x$", fontsize=15)
	ax1.set_ylabel("$u(x)$", fontsize=15)
	ax1.set_title(f"Solving 1D Volterra integral equation using MC-{model.name}", fontsize=20)
	ax1.legend()
	ax1.grid()

	# right subplot: residual (log scale)
	ax2.plot(x_pred.detach().cpu(), residual.detach().cpu(), label="Absolute error", color="red")
	ax2.set_xlabel("$x$", fontsize=15)
	ax2.set_ylabel("Absolute error (log scale)", fontsize=15)
	ax2.set_yscale('log')
	ax2.set_title("Absolute error between predicted and exact solution", fontsize=20)
	ax2.legend()
	ax2.grid()

	# add statistical information of residual
	relative_loss = torch.sqrt((residual**2).mean())/torch.sqrt((u_solution**2).mean())
	l2_loss = torch.sqrt((residual**2).mean())
	print(f"Relative loss: {relative_loss.item()}")
	print(f"L2 loss: {l2_loss.item()}")

	# display the parameters
	if m == 0:
		for i in range(len(model.lys)):
			print(model.lys[i].weight)
			print(model.lys[i].bias)
	elif m == 1:
		for i in range(len(model.lys)):
			print(model.lys[i].weight)
	summary(model, input_size=(1, 1), verbose=2, dtypes=[dtype, dtype])

	# display image
	plt.tight_layout() # automatically adjust subplot spacing
	plt.savefig(f"Volterra1D_MC-{model.name}.png")
	plt.show()
