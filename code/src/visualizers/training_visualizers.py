"""
this script contains various tools to help visualize training runs
"""

import os
from pathlib import Path

import json
import numpy as np
import matplotlib.pyplot as plt

class TrainingVisualizer():
	"""
	main class for interactive training visualization

	visualizes
	training loss
	validation loss
	validation perplexity
	learning rate (scaled)

	outputs json logs to text file
	creates iterative plot
	"""

	def __init__(
		self,
		output_dir,
		total_steps,
		num_epochs,
		lr_scaling_factor, # log(v) or max expected loss
	):

		# create the output dir if it doesn't exist
		Path(output_dir).mkdir(exist_ok=True, parents=True)

		# these are the text and plot files
		self.text_logs_path = os.path.join(output_dir, "training_logs.txt")
		self.plot_logs_path = os.path.join(output_dir, "training_logs.jpg")
		open(self.text_logs_path, "w+").close()

		# save useful values
		self.total_steps = total_steps
		self.num_epochs = num_epochs
		self.steps_per_epoch = total_steps // num_epochs
		self.lr_scaling_factor = lr_scaling_factor

		# save list of values
		self.training_loss_values = []
		self.validation_loss_values =[]
		self.lr_values = []
		self.perplexity_values = []
		self.steps = []

	def update(self, value_dict):
		"""
		value dict is dumped directly to text logs and then used for plotting
		"""

		# ez step first: dump values to txt
		with open(self.text_logs_path, "a+") as f:
			f.write(json.dumps(value_dict)+"\n")

		# update the lists
		self.steps.append(value_dict["step"])
		self.training_loss_values.append(value_dict["train_loss"])
		self.validation_loss_values.append(value_dict["val_loss"])
		self.perplexity_values.append(value_dict["val_pplx"])
		self.lr_values.append(value_dict["lr"] * self.lr_scaling_factor)

		# Find lowest loss points
		min_train_idx = np.argmin(self.training_loss_values)
		min_val_idx = np.argmin(self.validation_loss_values)

		min_train_step = self.steps[min_train_idx]
		min_train_loss = self.training_loss_values[min_train_idx]

		min_val_step = self.steps[min_val_idx]
		min_val_loss = self.validation_loss_values[min_val_idx]

		# Plot
		plt.figure()
		plt.plot(self.steps, self.training_loss_values, label=f"Training Loss (min={min_train_loss:.4f})")
		plt.plot(self.steps, self.validation_loss_values, label=f"Validation Loss (min={min_val_loss:.4f})")
		# plt.plot(self.steps, self.lr_values, label="Scaled LR")

		# Mark min points
		plt.scatter(min_train_step, min_train_loss)
		plt.scatter(min_val_step, min_val_loss)

		# Labels and legend
		plt.xlabel("Steps")
		plt.ylabel("Values")
		plt.title("Training Progress")
		plt.legend()
		plt.grid(True)

		plt.savefig(self.plot_logs_path)
		plt.close()
