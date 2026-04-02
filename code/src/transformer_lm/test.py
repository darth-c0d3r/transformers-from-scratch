"""
use this script to test end to end functionalities of transformers
"""

# ================================================================================ #
# IMPORTS GO HERE

import sys
sys.path.insert(1, "../tokenizers/")

import torch
from bpe_tokenizer import *
from embeddings import TokenEmbeddings, RotaryPositionEmbeddings

from transformers_core import *
# from decoder_lm import Decoder_LM
# import matplotlib.pyplot as plt
# import json

# ================================================================================ #
# PARAMETERS GO HERE

# input size
batch_size = 8
seq_len = 64
max_seq_len = 1024

# tokenizer settings
vocab_size = 32

# model settings
model_dim = 16
hidden_dim = 32

# attn settings
num_heads = 12
head_dim = 8

# model depth
num_layers = 6

# device and data type
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.float16

# what tests to run
RUN_EMBEDDINGS_TESTS = True

print(f"\ndevice: {device} | dtype: {dtype}\n")

# ================================================================================ #
# HELPER FUNCTIONS GO HERE

def print_tensor_info(tensor, name):
	print(f">> {name} | shape = {tensor.shape} | dtype = {tensor.dtype}\n")

# ================================================================================ #
# TEST TOKEN, POSITION, ROTARY EMBEDDINGS HERE

if RUN_EMBEDDINGS_TESTS:

	print(f"{'=' * 100}")
	print("TESTING TOKEN EMBEDDINGS, SINUSOIDAL EMBEDDINGS, ROTARY POSITION EMBEDDINGS\n")

	# get 3 different modules

	token_embeddings_without_sinusoidal = TokenEmbeddings(
		vocab_size=vocab_size,
		model_dim=model_dim,
		posn_embeddings=False,
		max_seq_len=max_seq_len,
		device=device,
		dtype=dtype,
	)

	token_embeddings_with_sinusoidal = TokenEmbeddings(
		vocab_size=vocab_size,
		model_dim=model_dim,
		posn_embeddings=True,
		max_seq_len=max_seq_len,
		device=device,
		dtype=dtype,
	)

	rotary_position_embeddings = RotaryPositionEmbeddings(
		head_dim=head_dim,
		max_seq_len=max_seq_len,
		device=device,
		dtype=dtype,
	)

	# TESTING TOKEN EMBEDDINGS

	input_ids = torch.randint(low=0, high=vocab_size, size=(batch_size, seq_len))
	print_tensor_info(input_ids, "input_ids")

	# vanilla case
	token_embeddings_1_output = token_embeddings_without_sinusoidal(input_ids)
	token_embeddings_2_output = token_embeddings_with_sinusoidal(input_ids)

	print_tensor_info(token_embeddings_1_output, "token_embeddings_1_output")
	print_tensor_info(token_embeddings_2_output, "token_embeddings_2_output")

	# shifting case (this will also work for padding)
	shift = 10
	input_ids_shifted = input_ids[:, shift:]
	token_positions_shifted = torch.arange(input_ids_shifted.shape[1]).repeat((input_ids_shifted.shape[0], 1)) + shift

	print_tensor_info(input_ids_shifted, "input_ids_shifted")
	print_tensor_info(token_positions_shifted, "token_positions_shifted")

	token_embeddings_1_shifted_output = token_embeddings_without_sinusoidal(input_ids_shifted, token_positions_shifted)
	token_embeddings_2_shifted_output = token_embeddings_with_sinusoidal(input_ids_shifted, token_positions_shifted)

	print_tensor_info(token_embeddings_1_shifted_output, "token_embeddings_1_shifted_output")
	print_tensor_info(token_embeddings_2_shifted_output, "token_embeddings_2_shifted_output")

	assert torch.all(token_embeddings_1_shifted_output == token_embeddings_1_output[:, shift:, :])
	assert torch.all(token_embeddings_2_shifted_output == token_embeddings_2_output[:, shift:, :])

	# kv cache case
	slice_idx = 12
	input_ids_sliced = input_ids[:, slice_idx:slice_idx+1]
	token_positions_sliced = torch.tensor([slice_idx]).repeat((input_ids_sliced.shape[0], 1))

	print_tensor_info(input_ids_sliced, "input_ids_sliced")
	print_tensor_info(token_positions_sliced, "token_positions_sliced")

	token_embeddings_1_sliced_output = token_embeddings_without_sinusoidal(input_ids_sliced, token_positions_sliced)
	token_embeddings_2_sliced_output = token_embeddings_with_sinusoidal(input_ids_sliced, token_positions_sliced)

	print_tensor_info(token_embeddings_1_sliced_output, "token_embeddings_1_sliced_output")
	print_tensor_info(token_embeddings_2_sliced_output, "token_embeddings_2_sliced_output")

	assert torch.all(token_embeddings_1_sliced_output == token_embeddings_1_output[:, slice_idx:slice_idx+1, :])
	assert torch.all(token_embeddings_2_sliced_output == token_embeddings_2_output[:, slice_idx:slice_idx+1, :])

	# TESTING RoPE EMBEDDINGS
	print(f"{'='*20}\n")

	rotary_input = torch.randn((batch_size, num_heads, seq_len, head_dim), dtype=dtype)
	print_tensor_info(rotary_input, "rotary_input")

	# vanilla case
	rotary_output = rotary_position_embeddings(rotary_input)
	print_tensor_info(rotary_output, "rotary_output")

	# shifting case (this will also work for padding)
	shift = 10
	rotary_input_shifted = rotary_input[:, :, shift:, :]
	token_positions_shifted = torch.arange(rotary_input_shifted.shape[-2]).repeat(rotary_input_shifted.shape[:-2] + tuple([1])) + shift

	print_tensor_info(rotary_input_shifted, "rotary_input_shifted")
	print_tensor_info(token_positions_shifted, "token_positions_shifted")

	rotary_output_shifted = rotary_position_embeddings(rotary_input_shifted, token_positions_shifted)

	print_tensor_info(rotary_output_shifted, "rotary_output_shifted")

	assert torch.all(rotary_output_shifted == rotary_output[:, :, shift:, :])

	# kv cache case
	slice_idx = 12
	rotary_input_sliced = rotary_input[:, :, slice_idx:slice_idx+1, :]
	token_positions_sliced = torch.tensor([slice_idx]).repeat(rotary_input_sliced.shape[:-2] + tuple([1]))

	print_tensor_info(rotary_input_sliced, "rotary_input_sliced")
	print_tensor_info(token_positions_sliced, "token_positions_sliced")

	rotary_output_sliced = rotary_position_embeddings(rotary_input_sliced, token_positions_sliced)

	print_tensor_info(rotary_output_sliced, "rotary_output_sliced")

	assert torch.all(rotary_output_sliced == rotary_output[:, :, slice_idx:slice_idx+1, :])

	print(f"{'='*20}\n")
	# TESTING WITH PADDING MASK (Shifting Case)

	shift = 13

	input_ids = torch.randint(low=0, high=vocab_size, size=(batch_size, seq_len))
	print_tensor_info(input_ids, "input_ids")

	input_ids_shifted = input_ids[:, shift:]
	print_tensor_info(input_ids_shifted, "input_ids_shifted")

	padding_mask = torch.ones(input_ids.shape)
	padding_mask[:, :shift] = 0
	print_tensor_info(padding_mask, "padding_mask")

	token_positions_padding = get_token_positions_from_padding_mask(padding_mask)
	print_tensor_info(token_positions_padding, "token_positions_padding")

	token_embeddings_1_output = token_embeddings_with_sinusoidal(input_ids_shifted)
	print_tensor_info(token_embeddings_1_output, "token_embeddings_1_output")

	token_embeddings_2_output = token_embeddings_with_sinusoidal(input_ids, token_positions_padding)
	print_tensor_info(token_embeddings_2_output, "token_embeddings_2_output")

	assert torch.all(token_embeddings_1_output == token_embeddings_2_output[:, shift:])

	print(f"{'='*20}\n")
	
	rotary_input = torch.randn((batch_size, num_heads, seq_len, head_dim), dtype=dtype)
	print_tensor_info(rotary_input, "rotary_input")

	rotary_input_shifted = rotary_input[:, :, shift:, :]
	print_tensor_info(rotary_input_shifted, "rotary_input_shifted")

	token_positions_padding = token_positions_padding.unsqueeze(1).repeat((1, num_heads, 1))
	print_tensor_info(token_positions_padding, "token_positions_padding")

	rotary_output_shifted = rotary_position_embeddings(rotary_input_shifted)
	print_tensor_info(rotary_output_shifted, "rotary_output_shifted")

	rotary_output_padding = rotary_position_embeddings(rotary_input, token_positions_padding)
	print_tensor_info(rotary_output_padding, "rotary_output_padding")

	assert torch.all(rotary_output_shifted == rotary_output_padding[:, :, shift:, :])

	print(f"{'=' * 100}\n")

import sys; sys.exit()

# ========================= #

input_ids = torch.randint(low=0, high=vocab_size, size=(batch_size, seq_len))

print(f"input_ids.shape = {input_ids.shape} ({input_ids.dtype})")

# ========================= #

token_embeddings_module = TokenEmbeddings(
	vocab_size = vocab_size,
	model_dim = model_dim,
	device = device,
	dtype = dtype,
)

token_embeddings_output = token_embeddings_module(input_ids)

print(f"token_embeddings_output.shape = {token_embeddings_output.shape} ({token_embeddings_output.dtype})")

# ========================= #

linear_module_1 = Linear(model_dim, hidden_dim, False, device, dtype)
linear_module_2 = Linear(hidden_dim, model_dim, True, device, dtype)

linear_output_1 = linear_module_1(token_embeddings_output)
linear_output_2 = linear_module_2(linear_output_1)

print(f"linear_output_1.shape = {linear_output_1.shape} ({linear_output_1.dtype})")
print(f"linear_output_2.shape = {linear_output_2.shape} ({linear_output_2.dtype})")

# ========================= #

relu_output = activation_ReLU(linear_output_2)

print(f"relu_output.shape = {relu_output.shape} ({relu_output.dtype})")

# ========================= #

ffn_module = FFN_with_SwiGLU(model_dim, hidden_dim, device, dtype)

ffn_output = ffn_module(token_embeddings_output)

print(f"ffn_output.shape = {ffn_output.shape} ({ffn_output.dtype})")

# ========================= #

layer_norm_module = LayerNorm_RMS(model_dim, 1e-5, device, dtype)

layer_norm_output = layer_norm_module(ffn_output)

print(f"layer_norm_module.shape = {layer_norm_output.shape} ({layer_norm_output.dtype})")

# ========================= #

softmax_output = softmax(layer_norm_output, -1)

print(f"softmax_output.shape = {softmax_output.shape} ({softmax_output.dtype})")

# ========================= #

rope_module = RotaryPositionEmbeddings(model_dim, seq_len*16, 10000, device, dtype)

rope_output = rope_module(ffn_output)

print(f"rope_output.shape = {rope_output.shape} ({rope_output.dtype})")

rope_output = rope_module(ffn_output.repeat((4,3,2,1,1,1)))

print(f"rope_output.shape = {rope_output.shape} ({rope_output.dtype})")

# ========================= #

keys = torch.randn((12, 16, 32, 64)).to(dtype)
values = torch.randn((12, 16, 32, 128)).to(dtype)
queries = torch.randn((12, 16, 4, 64)).to(dtype)

attn_weights, attn_output = scaled_dot_product_attention(queries, keys, values)

print(f"attn_weights.shape = {attn_weights.shape} ({attn_weights.dtype})")
print(f"attn_output.shape = {attn_output.shape} ({attn_output.dtype})")

# ========================= #

rope_module_mha = RotaryPositionEmbeddings(head_dim, seq_len*16, 10000, device, dtype)
mha_module = MultiHeadAttention(model_dim, head_dim, num_heads, True, device, dtype)

_, mha_module_wo_rope = mha_module(ffn_output, rope_module=None)
print(f"mha_module_wo_rope.shape = {mha_module_wo_rope.shape} ({mha_module_wo_rope.dtype})")

_, mha_module_w_rope = mha_module(ffn_output, rope_module=rope_module_mha)
print(f"mha_module_w_rope.shape = {mha_module_w_rope.shape} ({mha_module_w_rope.dtype})")

# ========================= #

decoder_block = DecoderOnlyTransformerBlock(model_dim, hidden_dim, head_dim, num_heads, device, dtype)

decoder_attn_weights, decoder_block_output = decoder_block(token_embeddings_output, rope_module=rope_module_mha)

print(f"decoder_block_output.shape = {decoder_block_output.shape} ({decoder_block_output.dtype})")
print(f"decoder_attn_weights.shape = {decoder_attn_weights.shape} ({decoder_attn_weights.dtype})")

# ========================= #

decoder_lm = Decoder_LM(
	vocab_size = vocab_size,
	model_dim = model_dim,
	hidden_dim = hidden_dim,
	head_dim = hidden_dim,
	num_heads = num_heads,
	num_layers = num_layers,
	max_seq_len = seq_len * 16,
	device = device,
	dtype = dtype,
)

attn_weights_list, decoder_lm_output = decoder_lm(input_ids)

print(f"decoder_lm_output.shape = {decoder_lm_output.shape} ({decoder_lm_output.dtype})")
print(f"attn_weights_list.shape = {attn_weights_list.shape} ({attn_weights_list.dtype})")

padding_mask = (torch.randn((batch_size, seq_len)) > 0).to(torch.int)
attn_weights_list, decoder_lm_output = decoder_lm(input_ids, padding_mask)

print(f"decoder_lm_output.shape = {decoder_lm_output.shape} ({decoder_lm_output.dtype})")
print(f"attn_weights_list.shape = {attn_weights_list.shape} ({attn_weights_list.dtype})")

# ========================= #

print()
print(decoder_lm)
print()
param_counts = {n: p.numel() for (n, p) in decoder_lm.named_parameters()}
# print(json.dumps(param_counts, indent=4))
print(f"total parameters = {sum(param_counts.values())}")

# ========================= #
# test end to end generation

print()
tokenizer = BPE_Tokenizer()
tokenizer.load_tokenizer("../tokenizers/tokenizer_virat_kohli/")
vocab_size = tokenizer.vocab_size
print(f"tokenizer vocab_size = {vocab_size}")

decoder_lm = Decoder_LM(
	vocab_size = vocab_size,
	model_dim = model_dim,
	hidden_dim = hidden_dim,
	head_dim = hidden_dim,
	num_heads = num_heads,
	num_layers = num_layers,
	max_seq_len = seq_len * 16,
	device = device,
	dtype = dtype,
)

input_text = "I think Virat Kohli is the"
input_ids = torch.tensor(parallel_encode(tokenizer, input_text, cache=True))
input_ids = input_ids.unsqueeze(0).repeat((batch_size, 1)).to(device)
padding_mask = torch.ones(input_ids.shape).to(device)

attn_weights_list, output_ids = decoder_lm.generate(input_ids=input_ids, padding_mask=None, max_new_tokens=16)

# decode the first batch
decoded_text = tokenizer.decode(output_ids[0].cpu().tolist())
print(f"no padding: {decoded_text}")
print()

decoder_lm.save_model("temp_model")

decoder_lm_2 = Decoder_LM.load_model("temp_model", device=device)
attn_weights_list, output_ids = decoder_lm_2.generate(input_ids=input_ids, padding_mask=None, max_new_tokens=16)

# decode the first batch
decoded_text = tokenizer.decode(output_ids[0].cpu().tolist())
print(f"no padding on reload: {decoded_text}")
print()



if False:

	padding_lengths = [4]
	for p in padding_lengths:

		pad = torch.randint(vocab_size, (batch_size, p)).to(device)

		ones = torch.ones(pad.shape).to(device)
		zeros = torch.zeros(pad.shape).to(device)

		attn_weights_list, output_ids = decoder_lm.generate(
			input_ids=torch.concat([pad, input_ids], dim=1),
			padding_mask=torch.concat([ones, padding_mask], dim=1),
			max_new_tokens=16,
		)

		decoded_text = tokenizer.decode(output_ids[0].cpu().tolist())

		print(f"pad size {p}, w/o masking: {decoded_text}")
		print()

		attn_weights_list, output_ids = decoder_lm.generate(
			input_ids=torch.concat([pad, input_ids], dim=1),
			padding_mask=torch.concat([zeros, padding_mask], dim=1),
			max_new_tokens=16,
		)
		
		decoded_text = tokenizer.decode(output_ids[0].cpu().tolist())

		print(f"pad size {p}, w/ masking: {decoded_text}")
		print()


# ========================= #



if False:

	# test correctness of scaled dot product attn
	keys = torch.randn((16, 32, 64))
	values = torch.randn((16, 32, 128))
	queries = torch.randn((16, 4, 64))

	attn_weights, attn_output = scaled_dot_product_attention(queries, keys, values)

	print(f"attn_weights.shape = {attn_weights.shape} ({attn_weights.dtype})")
	print(f"attn_output.shape = {attn_output.shape} ({attn_output.dtype})")

	for i in range(keys.shape[0]):
		k = keys[i]
		q = queries[i]
		v = values[i]

		w = attn_weights[i]
		o = attn_output[i]

		for j in range(o.shape[0]):

			qj = q[j]
			sdpa = torch.tensor([torch.dot(qj, k[k_])/np.sqrt(k.shape[-1]) for k_ in range(k.shape[0])])
			sdpa = softmax(sdpa, -1)
			diff = torch.max(torch.abs(w[j] - sdpa))
			assert diff.item() < 1e-5
			# print(i, j, "w", diff.item())
			# assert torch.allclose(w[j], sdpa, 1e-4), f"{i}, {j}"

			c = torch.stack([w[j][k_]*v[k_] for k_ in range(v.shape[0])])
			c = torch.sum(c, 0)
			diff = torch.max(torch.abs(o[j] - c))
			assert diff.item() < 1e-5
			# print(i, j, "o", diff.item())
			# assert torch.allclose(o[j], c, 1e-4), f"{i}, {j}"

# ========================= #
# a nice test to visualize similarity scores

def get_similarity_scores(key, query):
	"""
	key.shape = batch_size x seq_len x model_dim
	query.shape = batch_size x model_dim
	output.shape = batch_size x seq_len
	"""

	cosine_similarity = torch.sum(key * query.unsqueeze(1), -1) / np.sqrt(key.shape[-1])
	softmax_output = softmax(cosine_similarity, -1)
	return softmax_output

def plot_sequence(sequence, title, start=0, end=None):

	if end is None:
		end = sequence.shape[0]

	y = sequence.tolist()[start:end]
	x = range(len(y))

	plt.plot(x, y, marker='.', linestyle='-', linewidth=1)
	plt.xlabel("Token Distance")
	plt.ylabel("Softmax Similarity")
	plt.title(title)

	# plt.ylim(-1, 1)
	plt.show()

if False:

	# case 1: key and queries are same at all positions = x
	rope_module_test = RotaryPositionEmbeddings(
			model_dim = 128,
			max_seq_len = 2048,
			wavelength = 10000,
			device = device,
			dtype = torch.float32,
		)

	x = torch.randn((128)).repeat((1, 2048, 1))
	x_rope = rope_module_test(x)
	similarity_scores = get_similarity_scores(key=x_rope, query=x_rope[:,-1,:]).squeeze()
	print(f"similarity_scores.shape = {similarity_scores.shape} ({similarity_scores.dtype})")

	plot_sequence(torch.flip(similarity_scores, dims=[0]), "Equal Query and Key at all Positions", 10, 200)

	x = torch.randn((1, 2048, 128))
	y = torch.randn((1, 2048, 128))
	x_rope = rope_module_test(x)
	y_rope = rope_module_test(y)

	similarity_scores = get_similarity_scores(key=x_rope, query=y_rope[:,-1,:]).squeeze()
	print(f"similarity_scores.shape = {similarity_scores.shape} ({similarity_scores.dtype})")

	plot_sequence(torch.flip(similarity_scores, dims=[0]), "Random Query and Key at all Positions", 100, 300)

# print(input_ids[0,:])
# print(token_embeddings_output[0,0,:])
# print(linear_output_1[0,0,:])
# print(linear_output_2[0,0,:])
# print(layer_norm_output[0,0,:])
# print(softmax_output[0,0,:])
