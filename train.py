import torch, os
import numpy as np
from tqdm import tqdm

def train(model, problem, max_epochs=1000, lr=1e-3, epsilon=1e-20, show_iter=20, opt_method="Adam", dynamic_lr=False, lr_iter=2000, lr_decay=0.5, save_path="results/best_model.pth", loss_history_path="results/loss_history.npy"):
	print("Start training...")
	print(f"Max Epochs: {max_epochs}, Learning Rate: {lr}, Epsilon: {epsilon}, Optimizer: {opt_method}")

	try:
		if opt_method == "LBFGS":
			optimizer = torch.optim.LBFGS(model.parameters(), lr=lr, max_iter=max_epochs, max_eval=None, history_size=50, line_search_fn=None)
		else:
			optimizer = torch.optim.__dict__[opt_method](model.parameters(), lr=lr)
	except:
		raise ValueError("Invalid optimizer method.")

	scheduler = None
	if dynamic_lr:
		# Use a more robust scheduler instead of manual decay
		scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=lr_decay, patience=lr_iter//show_iter)

	best_loss = float('inf')
	# Ensure results directory exists
	os.makedirs(os.path.dirname(save_path), exist_ok=True)
	loss_history = []

	pbar = tqdm(range(max_epochs), desc="Training", unit="epoch")
	for epoch in pbar:
		if opt_method == "LBFGS":
			def closure():
				optimizer.zero_grad()
				loss = problem.loss_fn(model)
				loss.backward()
				return loss
			loss = optimizer.step(closure)
		else:
			optimizer.zero_grad()
			loss = problem.loss_fn(model)
			loss.backward()
			optimizer.step()

		loss_val = loss.item()
		loss_history.append(loss_val)
		
		# Step scheduler if enabled
		if scheduler is not None:
			scheduler.step(loss_val)
			current_lr = optimizer.param_groups[0]['lr']
		else:
			current_lr = lr

		# Checkpointing best model
		if loss_val < best_loss:
			best_loss = loss_val
			torch.save(model.state_dict(), save_path)

		if epoch % show_iter == 0:
			pbar.set_postfix({"Loss": f"{loss_val:.2e}", "LR": f"{current_lr:.1e}", "Best": f"{best_loss:.2e}"})

		if loss_val <= epsilon:
			pbar.set_description(f"Converged at epoch {epoch}")
			break
	
	print(f"Training finished. Best loss: {best_loss:.2e}. Model saved to {save_path}")

	# Save loss history
	if loss_history_path:
		os.makedirs(os.path.dirname(loss_history_path), exist_ok=True)
		np.save(loss_history_path, np.array(loss_history))
		print(f"Loss history saved to {loss_history_path}")