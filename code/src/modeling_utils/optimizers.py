"""
this script contains various optimizers and lr schedulers
"""

import numpy as np
import torch

def LinearDecayWithWarmup_LRScheduler(lr, t, scheduler_params):
	"""
	api for scheduler = lr, t, scheduler_params
	lr = base learning rate
	t = current time step (range = [1, T])
	scheduler_params = necessary params
	
	necessary scheduler params for linear decay with warmup
	total_steps: total steps in training
	num_warmup_steps: number of steps to warmup for
	"""

	# make sure that the required hyper params are set
	assert "total_steps" in scheduler_params
	assert "num_warmup_steps" in scheduler_params
	assert t <= scheduler_params["total_steps"]

	# case 1: t <= num_warmup_steps
	if t <= scheduler_params["num_warmup_steps"]:
		lr_sched = lr * (t / scheduler_params["num_warmup_steps"])
		return lr_sched

	# case 2: t > num_warmup_steps
	total_steps_remain = scheduler_params["total_steps"] - scheduler_params["num_warmup_steps"]
	t_remain = t - scheduler_params["num_warmup_steps"]
	lr_sched = lr * ((total_steps_remain - t_remain) / total_steps_remain)

	return lr_sched

def CosnineAnnealingWithWarmup_LRScheduler(lr, t, scheduler_params):
	"""
	apply cosine annealed learning rate scheduler
	"""

	# make sure that the required hyper params are set
	assert "lr_min" in scheduler_params
	assert "num_warmup_steps" in scheduler_params
	assert "num_anneal_steps" in scheduler_params

	# case 1: t <= num_warmup_steps
	# warmup stage
	if t <= scheduler_params["num_warmup_steps"]:
		lr_sched = lr * (t / scheduler_params["num_warmup_steps"])
		return lr_sched

	# case 2: t <= num_anneal_steps
	# cosine annealing till lr_min
	if t <= scheduler_params["num_anneal_steps"]:
		lr_min = scheduler_params["lr_min"]
		t1 = scheduler_params["num_warmup_steps"]
		t2 = scheduler_params["num_anneal_steps"]
		lr_sched = lr_min + (((lr - lr_min)/2.) * (1 + np.cos(np.pi * ((t-t1)/(t2-t1)))) )
		return lr_sched

	# case 3: t > num_anneal_steps
	return scheduler_params["lr_min"]

def apply_max_norm_gradient_clipping(param_groups, max_norm, eps=1e-6):
	"""
	apply max norm gradient clipping in place
	"""

	# iteration 1 over all params to compute l2 gradient norm
	l2_norm = 0.

	for group in param_groups:
		for p in group["params"]:
			if p.grad is not None:
				l2_norm += torch.sum(p.grad.data ** 2)

	l2_norm = torch.sqrt(l2_norm)

	# early return if l2 norm is within limits
	if l2_norm <= max_norm:
		return

	# compute the scaling coefficient
	coef = max_norm / (l2_norm + eps)

	# iteration 2 over all params to scale gradient
	for group in param_groups:
		for p in group["params"]:
			if p.grad is not None:
				p.grad.data.mul_(coef)

	return

class StochasticGradientDescent_Optimizer(torch.optim.Optimizer):
	"""
	main class for stochastic gradient descent optimizer

	we pass parameter groups to be optimized in the constructor
	in case different parameter groups have different hyper-params
	like learning rate, weight decay, etc
	"""

	def __init__(
		self,
		params, # model parameter groups to be optimized
		lr = 1e-5, # learning rate
		lr_scheduler = None,
		lr_scheduler_params = None,
		max_gradient_norm = None,
	):

		"""
		init function for stochastic gradient descent
		"""

		# call the super constructor
		# apparently the init for optimizer class is a bit involved
		# we need to pass a few things to be kept track of

		defaults = {"lr": lr}

		super().__init__(
			params,
			defaults = defaults,
		)

		# you don't have to explicitly set member variables
		# just add it to "defaults"
		# and they shall be available for each param group, weird

		# set the scheduler here create a lambda function for it
		self.lr_scheduler = None
		if lr_scheduler is not None:
			self.lr_scheduler = lambda t: lr_scheduler(lr, t, lr_scheduler_params)
		self.current_lr = lr # for plotting

		# set the max gradient norm for clipping
		self.max_gradient_norm = max_gradient_norm

	def step(self):
		"""
		we don't need to pass anything to the step function
		everything is stored within the state of optmizer
		"""

		# apply gradient clipping if needed
		if self.max_gradient_norm is not None:
			apply_max_norm_gradient_clipping(self.param_groups, self.max_gradient_norm, eps=1e-6)

		# iterate over all param groups
		# the parent class initialization creates this
		for group in self.param_groups:

			# get the group learning rate
			# this too is assigned by parent's init
			lr = group["lr"]

			# iterate over all params in the group
			for p in group["params"]:

				# happens when requires_grad = False
				if p.grad is None:
					continue

				# get the state of p
				state = self.state[p]

				# get the time step count
				t = state.get("t", 1)

				# use lr_scheduler if needed
				if self.lr_scheduler is not None:
					lr = self.lr_scheduler(t)
				self.current_lr = lr # for plotting

				# get the gradient for p
				# this is populated by .backward()
				grad = p.grad.data

				# update the values of the param in place
				p.data -= (lr / np.sqrt(t)) * grad

				# update the state for that param
				state["t"] = t + 1

		return

class AdamW_Optimizer(torch.optim.Optimizer):
	"""
	implments the adamW optimizer
	combines momentum and weight decay
	more explanations of flow are in sgd ^
	"""

	def __init__(
		self,
		params, # model.parameters()
		lr = 1e-5,
		b1 = 0.9,
		b2 = 0.999,
		weight_decay = 0.01,
		eps = 1e-8,
		lr_scheduler = None,
		lr_scheduler_params = None,
		max_gradient_norm = None,
	):
		"""
		initialize all hyper parameters for adamW
		"""

		# set the defaults
		defaults = {
			"lr": lr,
			"b1": b1,
			"b2": b2,
			"weight_decay": weight_decay,
			"eps": eps,
		}

		# call the super constructor
		super().__init__(
			params,
			defaults = defaults,
		)

		# set the scheduler here create a lambda function for it
		self.lr_scheduler = None
		if lr_scheduler is not None:
			self.lr_scheduler = lambda t: lr_scheduler(lr, t, lr_scheduler_params)
		self.current_lr = lr # for plotting

		# set the max gradient norm for clipping
		self.max_gradient_norm = max_gradient_norm

	def step(self):
		"""
		now implement the adamW algorithm
		"""

		# apply gradient clipping if needed
		if self.max_gradient_norm is not None:
			apply_max_norm_gradient_clipping(self.param_groups, self.max_gradient_norm, eps=1e-6)

		# iterate over all param groups
		for group in self.param_groups:

			# get all hyper params
			lr = group["lr"]
			b1 = group["b1"]
			b2 = group["b2"]
			weight_decay = group["weight_decay"]
			eps = group["eps"]

			# iterate over all params in the group
			for p in group["params"]:

				# happens when requires_grad = False
				if p.grad is None:
					continue

				# get the state of p
				state = self.state[p]

				# get the state values
				t = state.get("t", 1)
				m = state.get("m", torch.zeros(p.data.shape, dtype=p.data.dtype, device=p.device))
				v = state.get("v", torch.zeros(p.data.shape, dtype=p.data.dtype, device=p.device))

				# use lr_scheduler if needed
				if self.lr_scheduler is not None:
					lr = self.lr_scheduler(t)
				self.current_lr = lr # for plotting

				# get the gradient for p
				grad = p.grad.data

				# get the updated values of m and v
				m = (b1 * m) + ((1-b1) * grad)
				v = (b2 * v) + ((1-b2) * (grad**2))

				# get the effective value of lr
				lr_t = lr * np.sqrt(1-(b2**t)) / (1-(b1**t))

				# do the weight update with moments
				p.data -= lr_t * (m / (torch.sqrt(v) + eps))
				# p.data -= lr_t * grad

				# do the weight update with weight decay
				p.data = p.data * (1 - (lr * weight_decay))

				# update the state variables
				state["t"] = t+1
				state["m"] = m
				state["v"] = v

		return
