import torch
import matplotlib.pyplot as plt

def get_sinusoidal_embeddings(wavelength, max_dim, dim, seq_len):

	embeddings = torch.arange(seq_len, dtype=torch.float32)
	const = float(wavelength ** (2 * dim / max_dim))
	embeddings /= const
	return torch.sin(embeddings)

def plot_sequence(sequence, title, start=0, end=None):

	if end is None:
		end = sequence.shape[0]

	y = sequence.tolist()[start:end]
	x = range(len(y))

	plt.figure(figsize=(16, 4))
	plt.plot(x, y, marker='.', linestyle='-', linewidth=1)
	plt.xlabel("Token Distance")
	plt.ylabel("Softmax Similarity")
	plt.title(title)

	# plt.ylim(-1, 1)
	plt.show()


wavelength = 10000
max_dim = 1024
dim = 64
seq_len = 100

token_embeddings = torch.randn((seq_len))

position_embeddings = get_sinusoidal_embeddings(wavelength, max_dim, dim, seq_len)

final_embeddings = token_embeddings + position_embeddings

plot_sequence(token_embeddings, "token_embeddings")
plot_sequence(position_embeddings, "position_embeddings")
plot_sequence(final_embeddings, "final_embeddings")