
if __name__ == '__main__':
	device = torch.device("cuda:0")
	dtype = torch.float64
	x_b = sample_boundary_points(dim=10, num_samples_per_boundary=100, device=device, dtype=dtype,
								 bound=True).requires_grad_(True)  # torch.Size([20000, 10])
	x_i = torch.rand(1000, 10, device=device, dtype=dtype)  # torch.Size([10000, 10])
	ksi = torch.rand(10, 10, device=device, dtype=dtype)
	print(loss_ph(solution, x_i, ksi), loss_bc(solution, x_b))