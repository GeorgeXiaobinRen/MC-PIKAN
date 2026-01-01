import torch, time

def train(model, loss_fn, x, ksi, max_epochs=1000, lr=1e-3, epsilon=1e-20, show_iter=200, opt_method="Adam", dynamic_lr=True):
	print("Start training...")
	print(f"Max Epochs: {max_epochs}, Learning Rate: {lr}, Epsilon: {epsilon}, Interval of Display: {show_iter}, Dynamic LR: {dynamic_lr}")

	lr0 = lr
	if opt_method == "Adam":
		optimizer = torch.optim.Adam(model.parameters(), lr=lr)
	elif opt_method == "LBFGS":
		optimizer = torch.optim.LBFGS(model.parameters(), lr=lr)
	elif opt_method == "SGD":
		optimizer = torch.optim.SGD(model.parameters(), lr=lr)
	elif opt_method == "RMSprop":
		optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
	else:
		raise ValueError("Invalid optimizer method")
	time0 = time.time()

	for epoch in range(max_epochs):
		optimizer.zero_grad()
		loss = loss_fn(model, x, ksi)
		loss.backward()

		total_grad_norm = 0
		for p in model.parameters():
			if p.grad is not None:
				total_grad_norm += p.grad.norm().item()

		optimizer.step()

		if dynamic_lr:
			lr = lr * (1e-1 ** (epoch // 1000))
			for param_group in optimizer.param_groups:
				param_group['lr'] = lr

		if epoch % show_iter == 0:
			time1 = time.time()
			speed = 1000 * (time1 - time0) / show_iter
			if dynamic_lr:
				print(f"Epoch {epoch}, Learning Rate: {lr}, Loss: {loss.item()}, Speed: {speed} ms/iter")
			else:
				print(f"Epoch {epoch}, Loss: {loss.item()}, Speed: {speed} ms/iter")
			time0 = time1

		if loss.item() <= epsilon:
			break