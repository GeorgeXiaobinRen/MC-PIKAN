import torch
import torch.nn as nn
import layers as kal

class CPIKANModel(nn.Module):
	def __init__(self, input_dim=1, output_dim=1, num_layers=3, hidden_dim=10, degree=3, dtype=torch.float32):
		super(CPIKANModel, self).__init__()
		self.num_layers = num_layers
		self.lys = nn.ModuleList()
		self.lys.append(kal.ChebyKANLayer(input_dim, hidden_dim, degree, dtype)) # input-hidden
		if num_layers >= 3:
			for i in range(1, num_layers - 1):
				self.lys.append(kal.ChebyKANLayer(hidden_dim, hidden_dim, degree, dtype)) # hidden-hidden
		self.lys.append(kal.ChebyKANLayer(hidden_dim, output_dim, degree, dtype)) # hidden-output
		self.name = "PIKAN"
		self.detailed_name = "cPIKAN with explicit expressions"

	def forward(self, x):
		for ly in self.lys:
			x = ly(x)
		return x


class Unoptd_cPIKANModel(nn.Module):
	def __init__(self, input_dim=1, output_dim=1, num_layers=3, hidden_dim=10, degree=3, dtype=torch.float32):
		super(Unoptd_cPIKANModel, self).__init__()
		self.num_layers = num_layers
		self.lys = nn.ModuleList()
		self.lys.append(kal.unoptd_ChebyKANLayer(input_dim, hidden_dim, degree, dtype))  # input-hidden
		if num_layers >= 3:
			for i in range(1, num_layers - 1):
				self.lys.append(kal.unoptd_ChebyKANLayer(hidden_dim, hidden_dim, degree, dtype))  # hidden-hidden
		self.lys.append(kal.unoptd_ChebyKANLayer(hidden_dim, output_dim, degree, dtype))  # hidden-output
		self.name = "Unoptimized cPIKAN"

	def forward(self, x):
		for ly in self.lys:
			x = ly(x)
		return x


class PINNModel(nn.Module):
	def __init__(self, input_dim=1, output_dim=1, num_layers=3, hidden_dim=20, dtype=torch.float32):
		super(PINNModel, self).__init__()
		self.num_layers = num_layers
		self.lys = nn.ModuleList()
		self.lys.append(nn.Linear(input_dim, hidden_dim, dtype=dtype))  # input-hidden
		if num_layers >= 3:
			for i in range(1, num_layers-1):
				self.lys.append(nn.Linear(hidden_dim, hidden_dim, dtype=dtype)) # hidden-hidden
		self.lys.append(nn.Linear(hidden_dim, output_dim, dtype=dtype))  # hidden-output
		self.activation = nn.Tanh()
		self.name = "PINNs"

	def forward(self, x):
		for i in range(self.num_layers-1):
			x = self.activation(self.lys[i](x))
		x = self.lys[-1](x)
		return x


def print_modelsize(model):
	print("The network model is", model.name)
	print("Layers:", model.lys)
	try:
		print("Activation function:", model.activation)
	except AttributeError:
		pass

	
	total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	print(f"Total trainable parameters: {total_params}")


if __name__ == "__main__":
	model1 = CPIKANModel(input_dim=10, output_dim=1, num_layers=3, hidden_dim=10, degree=2)
	model2 = PINNModel(input_dim=10, output_dim=1, num_layers=3, hidden_dim=20)
	print_modelsize(model1)
	print_modelsize(model2)