import torch
from itertools import product

# functions for high-dimensional problems
def sample_boundary_points(dim=10, num_samples_per_boundary=1000, device="cpu", dtype=torch.float32, bound=False):
	"""
	Generate random boundary points on each boundary of [0,1]^dim.
	Returns a tensor of shape (dim * 2, num_samples_per_boundary, dim).
	"""
	boundaries = []

	for i in range(dim):  # iterate through each dimension
		for fixed_val in [0.0, 1.0]:
			points = torch.rand(num_samples_per_boundary, dim, dtype=dtype)
			points[:, i] = fixed_val
			boundaries.append(points)

	# combine all points
	boundary_tensor = torch.stack(boundaries).to(device)
	if bound:
		boundary_tensor = boundary_tensor.view(-1, dim)
	return boundary_tensor


def generate_full_grid_torch(dim=10, points_per_dim=2, device='cpu', dtype=torch.float32):
	"""
	Generate a uniform grid for [0,1]^dim (PyTorch implementation)
	"""
	# Generate coordinate values for each dimension (equally spaced)
	axis_values = torch.linspace(0, 1, points_per_dim, device=device, dtype=dtype)

	# Generate Cartesian product
	grid = torch.stack([
		torch.stack(tensors, dim=-1).view(-1)
		for tensors in product(*[axis_values] * dim)
	]).reshape(-1, dim)

	return grid


def generate_grid_with_specific_dims(fixed_values, variable_dims_indices=None, n_points=5, dtype=torch.float32):
	"""
	Generate a [0,1]^10 grid with specified dimensions as variables and the rest as fixed
	"""
	if variable_dims_indices is None:
		variable_dims_indices = [2, 4]

	# Convert fixed values to tensor
	fixed = torch.tensor(fixed_values, dtype=dtype)
	n_variable = len(variable_dims_indices)

	# Generate grid points for variable dimensions
	grid_1d = torch.linspace(0, 1, n_points, dtype=dtype)
	grids = torch.meshgrid(*([grid_1d] * n_variable), indexing='ij')
	variable_values = torch.stack([g.flatten() for g in grids], dim=1)  # (n_points^n_variable, n_variable)

	# Build complete grid
	n_total_points = variable_values.shape[0]
	grid = torch.zeros(n_total_points, 10, dtype=dtype)

	# Fill fixed dimensions
	fixed_mask = torch.ones(10, dtype=torch.bool)
	fixed_mask[variable_dims_indices] = False
	grid[:, fixed_mask] = fixed.repeat(n_total_points, 1)

	# Fill variable dimensions
	grid[:, variable_dims_indices] = variable_values

	return grid


if __name__ == '__main__':
	device = torch.device("cuda:0")
	dtype = torch.float64
	x_b = sample_boundary_points(dim=10, num_samples_per_boundary=100, device=device, dtype=dtype,
								 bound=True).requires_grad_(True)  # torch.Size([20000, 10])
	x_i = torch.rand(1000, 10, device=device, dtype=dtype)  # torch.Size([10000, 10])
	ksi = torch.rand(10, 10, device=device, dtype=dtype)
	# print(loss_ph(solution, x_i, ksi), loss_bc(solution, x_b))