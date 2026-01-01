import time
import torch


def K(x, s):
	# return 1
	return -torch.sin(torch.pi * (x - s))

def f(x):
	# return 1
	return (1 + 1/(2*torch.pi)) * torch.sin(torch.pi*x) - x * torch.cos(torch.pi*x)/2

def solution(x):
	return torch.sin(torch.pi*x)

# 损失函数部分
def in_mean(model, x, s):
	N_x = x.size(0)
	N_s = s.size(0)

	x_expanded = x.view(-1, 1).expand(-1, N_s)  # (N_x, N_s)
	s_expanded = s.view(1, -1).expand(N_x, -1)  # (N_x, N_s)
	x_s = x_expanded * s_expanded  # (N_x, N_s)

	k_vals = K(x_expanded, x_s)  # (N_x, N_s)
	u_flat = model(x_s.view(-1, 1))  # (N_x * N_s,  )
	u_vals = u_flat.view(N_x, N_s)  # (N_x, N_s)

	product = x_expanded * k_vals * u_vals  # (N_x, N_s)
	inner_mean = model(x.view(-1, 1)).view(-1, ) - f(x) - torch.mean(product, dim=1)  # (N_x,)
	return inner_mean

def loss_fn(model, x, s):
	s1 = s[0]
	s2 = s[1]
	loss_mc = torch.mean(torch.abs(in_mean(model, x, s1) * in_mean(model, x, s2)))
	return loss_mc

# 训练过程
def train(model, x, s, epochs=5000, lr=1e-3, show_iter=500):
	optimizer = torch.optim.Adam(model.parameters(), lr=lr)
	time0 = time.time()
	for epoch in range(epochs):
		optimizer.zero_grad()
		loss = loss_fn(model, x, s)
		loss.backward()

		# 监控梯度
		total_grad_norm = 0
		for p in model.parameters():
			if p.grad is not None:
				total_grad_norm += p.grad.norm().item()
		if epoch % show_iter == 0:
			time1 = time.time()
			speed = 1000*(time1-time0)/show_iter
			print(f"Epoch {epoch}, Loss: {loss.item()}, Grad Norm: {total_grad_norm}, Speed: {speed} ms/iter")
			time0 = time.time()
		optimizer.step()
	# optimizer = torch.optim.LBFGS(model.parameters(), lr=lr)
	# scheduler = StepLR(optimizer, step_size=100, gamma=0.95)
	#
	# def closure():
	# 	optimizer.zero_grad()
	# 	loss = loss_fn(model, x, s)
	# 	loss.backward()
	# 	return loss
	#
	# for epoch in range(epochs):
	# 	optimizer.step(closure)
	# 	scheduler.step()
	# 	if epoch % 500 == 0:
	# 		loss = closure()
	# 		print(f"Epoch {epoch + epochs}, Loss: {loss.item()}")
	# 		print(loss.dtype)