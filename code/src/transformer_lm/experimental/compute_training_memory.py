"""
compute number of fp32 variables to be stored on gpu for training
this includes weights, activations, gradients, optimizer states
"""

def compute_peak_memory(b, l, d_m, d_h, d_k, h, n, v):

	params = (2 * v * d_m) + n * ((2 * d_m) + (4 * d_m * h * d_k) + (3 * d_m * d_h)) + d_m

	activations = n * ((2 * b * l * d_m) + (3 * b * l * d_k * h) + (b * l * d_m) + (2 * b * l * l * h) + (b * l * d_m))
	activations +=  n * ((2 * b * l * d_h) + (b * l * d_m))
	activations += ((b * l * d_m) + (b * l * v) + (b * l))

	# 4 x params because params, grad, m, v
	return activations + (4 * params)

def convert_bytes_to_gb(bts):
	return bts / (2**30)

gpt_mem = lambda bs: convert_bytes_to_gb(compute_peak_memory(
	b=bs,
	l=1024,
	n=48,
	d_m=1600,
	h=25,
	d_k=64,
	d_h=6400,
	v=50257))

print(gpt_mem(1))