"""
this script contains the class for core transformers operations
"""

import torch
import numpy as np

def activation_ReLU(input_batch):
	"""
	implements the ReLU activation function
	"""

	activation_output = torch.max(input_batch, torch.zeros(input_batch.shape))
	return activation_output.to(input_batch.dtype).to(input_batch.device)

def activation_SiLU(input_batch):
	"""
	implements the SiLU activation function
	use torch.sigmoid for numerical stability
	"""

	return input_batch * torch.sigmoid(input_batch)

def softmax(input_batch, dim, temperature=1.0):
	"""
	implements the softmax function for the agg_dimension
	input: can be any shape
	output: same shape as input
	"""

	# dim should be less than number of dimensions in input_batch
	assert dim < len(input_batch.shape)
	assert -1 * dim <= len(input_batch.shape)

	# # temperature can't be 0
	# assert temperature != 0.

	# if temperature is 0
	# softmax behaves like argmax
	if temperature == 0:
		print(f"> warning: softmax @ temp 0; returning argmax")
		
		# argmax_map = input_batch == input_batch.max(dim).values.unsqueeze(dim)
		# return argmax_map.to(input_batch.dtype).to(input_batch.device)

		indices = torch.argmax(input_batch, dim=dim, keepdim=True)
		one_hot = torch.zeros_like(input_batch).scatter_(dim, indices, 1)
		return one_hot.to(input_batch.dtype).to(input_batch.device)


	# logic
	# subtraction invariance for softmax
	# the value of softmax remains same
	# even if we add a constant C to all entries
	# exp(x_i) will be INF for large x_i
	# so we will subtract max(x)
	# this makes new max(x) = 0
	# along agg_dimension

	# step 1: subtract max value along agg_dimension
	effective_batch = input_batch - input_batch.max(dim).values.unsqueeze(dim)

	# step 2: compute softmax with temperature
	effective_batch = torch.exp(effective_batch / temperature)
	output_batch = effective_batch / effective_batch.sum(dim).unsqueeze(dim)

	return output_batch

def get_causal_attention_mask(seq_len):
	"""
	return causal attention mask of seq_len
	input: seq_len, int
	output: seq_len x seq_len mask
	returns a lower triangular square matrix
	"""

	return torch.tril(torch.ones((seq_len, seq_len)))

def scaled_dot_product_attention(
	queries,
	keys,
	values,
	attn_mask = None,
):
	"""
	compute the scaled dot product attention
	queries.shape = ... x seq_len_queries x d_k
	keys.shape = ... x seq_len_data x d_k
	values.shape = ... x seq_len_data x d_v
	output.shape = ... x seq_len_queries x d_v
	attn_mask.shape = ... x seq_len_queries x seq_len_data
	attn_mask is binary, either 0 or 1 elements
	"""

	# get the dot product here and divide by sqrt(d_k)
	similarity = queries @ keys.transpose(-1, -2)
	similarity /= (keys.shape[-1] ** 0.5)

	# if attn mask is not None
	# set similarity scores of masked out values
	# to be effectively - inf
	if attn_mask is not None:
		# similarity += (attn_mask-1)*1e20
		similarity = similarity.masked_fill(attn_mask == 0, float('-inf'))

	# similarity.shape = ... x seq_len_queries x seq_len_data
	# we want to compute softmax over keys
	attn_weights = softmax(similarity, -1)

	# now we want to multiply by values
	# attn_weights.shape = ... x seq_len_queries x seq_len_data
	# values.shape = ... x seq_len_data x d_v
	output = attn_weights @ values
	# output.shape = ... x seq_len_queries x d_v

	return attn_weights, output

class MultiHeadAttention(torch.nn.Module):
	"""
	class for applying multi head attention
	rope embeddings is optional and provided in forward
	write this in a way that it works for
	both causal and cross attention
	keep things generalized
	we assume d_k = d_v = head_dim
	"""

	def __init__(
		self,
		model_dim,
		head_dim,
		num_heads = 1,
		causal = True,
		device = None, # either None or a torch.device object
		dtype = torch.float32, # either None or a torch.dtype object
		init = True,
	):

		"""
		initialize a multi head attention block
		which works for causal as well as self attention
		"""

		# call the super constructor
		super().__init__()

		# initialize basic parameters
		self.model_dim = model_dim
		self.head_dim = head_dim
		self.num_heads = num_heads
		self.causal = causal
		self.device = device
		self.dtype = dtype

		# initialize empty projection matrices
		self.W_Q = Linear(
			input_dim = self.model_dim,
			output_dim = self.head_dim * self.num_heads,
			use_bias = False,
			device = device,
			dtype = dtype,
			init = init,
		)

		self.W_K = Linear(
			input_dim = self.model_dim,
			output_dim = self.head_dim * self.num_heads,
			use_bias = False,
			device = device,
			dtype = dtype,
			init = init,
		)

		self.W_V = Linear(
			input_dim = self.model_dim,
			output_dim = self.head_dim * self.num_heads,
			use_bias = False,
			device = device,
			dtype = dtype,
			init = init,
		)

		self.W_O = Linear(
			input_dim = self.head_dim * self.num_heads,
			output_dim = self.model_dim,
			use_bias = False,
			device = device,
			dtype = dtype,
			init = init,
		)

		return

	def forward(
		self,
		queries_batch, # torch.tensor
		data_batch = None, # optional, fall back to queries batch
		rope_module = None, # optional
		padding_mask = None, # optional, can be provided, default = causal
	):

		"""
		forward pass for attenion module
		queries_batch.shape = batch_size x seq_len_queries x model_dim
		data_batch.shape = batch_size x seq_len_data x model_dim
		"""

		# handle the self attention set up here
		if data_batch is None:
			data_batch = queries_batch

		# get the basic parameters in
		batch_size = queries_batch.shape[0]
		seq_len_queries = queries_batch.shape[1]
		seq_len_data = data_batch.shape[1]

		# get the first set of projections in
		projection_q = self.W_Q(queries_batch)
		projection_k = self.W_K(data_batch)
		projection_v = self.W_V(data_batch)

		# carve up the q k v projections to num heads
		projection_q = projection_q.contiguous().view((batch_size, seq_len_queries, self.num_heads, self.head_dim)).transpose(1, 2)
		projection_k = projection_k.contiguous().view((batch_size, seq_len_data, self.num_heads, self.head_dim)).transpose(1, 2)
		projection_v = projection_v.contiguous().view((batch_size, seq_len_data, self.num_heads, self.head_dim)).transpose(1, 2)
		# the projections shape is now batch_size x num_heads x seq_len x head_dim

		# apply rope embeddings if needed now
		# this is applied only to query and key vectors
		if rope_module is not None:
			projection_q = rope_module(projection_q)
			projection_k = rope_module(projection_k)

		# build the perfect attn mask which is
		# a combination of causal mask and padding mask
		# causal mask shape = seq_len_queries x seq_len_queries (only applicable for self attn)
		# padding mask shape = batch_size x seq_len_data
		# attn mask shape = batch_size x 1 x seq_len_queries x seq_len_data

		attn_mask = None

		# if padding mask isn't None
		# convert it into a good-er shape
		# padding_mask.shape = batch_size x 1 x 1 x seq_len_data
		# this as is would also work
		if padding_mask is not None:
			attn_mask = padding_mask.unsqueeze(1).unsqueeze(1)

		if self.causal is True:
			causal_mask = get_causal_attention_mask(seq_len_queries).to(queries_batch.device)

			# merge causal mask and padding mask here
			if padding_mask is not None:
				causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
				attn_mask = attn_mask * causal_mask

				# very subtle:
				# ensure that the padding tokens can still attend to themselves
				self_mask = torch.eye(seq_len_queries).to(queries_batch.device)
				attn_mask = torch.max(attn_mask, self_mask)

			else: # causal mask to be used as attn mask
				attn_mask = causal_mask

		# get the output after applying sdpa function
		attn_weights, output_batch = scaled_dot_product_attention(
			queries = projection_q,
			keys = projection_k,
			values = projection_v,
			attn_mask = attn_mask,
		)

		# now aggregate all attention heads
		# flip around num_heads and seq_len dimensions
		output_batch = output_batch.transpose(1, 2).flatten(start_dim=2)
		# output_batch.shape = batch_size x seq_len_queries x num_heads * head_dim

		# apply output projection now
		output_batch = self.W_O(output_batch)

		# return attn_weights and output_batch
		return attn_weights, output_batch

class LayerNorm_RMS(torch.nn.Module):
	"""
	class for layer normalization of a batched input
	this uses the root mean square normalization
	"""

	def __init__(
		self,
		model_dim,
		eps = 1e-5,
		device = None, # either None or a torch.device object
		dtype = torch.float32, # either None or a torch.dtype object
		init = True, # just here for consistency, not actually used
	):

		"""
		initialize a default rms layer norm layer
		"""

		# call the super constructor
		super().__init__()

		# initialize basic parameters
		self.model_dim = model_dim
		self.eps = eps
		self.device = device
		self.dtype = dtype

		# initialize gains parameter as ones
		self.G = torch.ones((self.model_dim), dtype=self.dtype, device=self.device)

		# case the gains to nn.Parameter (for training)
		self.G = torch.nn.Parameter(self.G)

		return

	def forward(
		self,
		input_batch, # torch.tensor
	):

		"""
		input: batch_size x sequence_length x model_dim
		output: batch_size x sequence_length x model_dim
		"""

		# each vector needs to be rms normed 
		# and then scaled by self.G

		# step 1: upcast input_batch to float32
		# this is to avoid overflows in rms calculation
		input_batch = input_batch.to(torch.float32)

		# step 2: compute rms
		# rms_batch.shape = batch_size x sequence_length
		rms_batch = torch.sqrt((torch.sum(input_batch**2, -1) / self.model_dim) + self.eps)

		# step 3: cast back to original data-type
		input_batch = input_batch.to(self.dtype)
		rms_batch = rms_batch.to(self.dtype).unsqueeze(-1)
		# need to add the unsqueeze for the divide to work

		# step 4: apply the normalization and scaling up operations
		output_batch = (input_batch / rms_batch) * self.G

		return output_batch

class Linear(torch.nn.Module):
	"""
	class for a simple linear layer
	this is described by input dim, output dim, bias
	"""

	def __init__(
		self,
		input_dim,
		output_dim,
		use_bias = False,
		device = None, # either None or a torch.device object
		dtype = None, # either None or a torch.dtype object
		init = True,
	):

		"""
		initialize a default linear layer
		"""

		# call the super constructor
		super().__init__()

		# initialize basic parameters
		self.input_dim = input_dim
		self.output_dim = output_dim
		self.use_bias = use_bias
		self.device = device
		self.dtype = dtype

		# set default data type as float32
		if self.dtype is None:
			self.dtype = torch.float32

		# initialize empty weights and biases
		self.W = torch.empty((self.input_dim, self.output_dim), dtype=self.dtype, device=self.device)
		if self.use_bias is True:
			self.B = torch.zeros((self.output_dim), dtype=self.dtype, device=self.device)

		# do a truncated normal distribution initialization if needed
		if init is True:

			sigma = np.sqrt(2 / (self.input_dim + self.output_dim))

			self.W = torch.nn.init.trunc_normal_(
				self.W,
				mean = 0,
				std = sigma ** 2,
				a = -3. * sigma,
				b = 3. * sigma,
			)

		# cast the weights and biases to nn.Parameter (for training)
		self.W = torch.nn.Parameter(self.W)
		if self.use_bias is True:
			self.B = torch.nn.Parameter(self.B)

		return

	def forward(
		self,
		input_batch, # torch.tensor
	):

		"""
		input: batch_size x sequence_length x input_dim
		output: batch_size x sequence_length x output_dim
		"""

		# step 1: projection
		output_batch = input_batch @ self.W

		# step 2: biasing (if needed)
		if self.use_bias is True:
			output_batch += self.B

		return output_batch

class FFN_with_SwiGLU(torch.nn.Module):
	"""
	class for the feed forward network in the decoder block
	decoder block consists of ffn and sel-attention
	this is one part of the decoder block
	the non linearity used here is SwiGLU
	which has its own gated weights as well
	"""

	def __init__(
		self,
		model_dim,
		hidden_dim,
		device = None, # either None or a torch.device object
		dtype = None, # either None or a torch.dtype object
		init = True,
	):

		"""
		initialize all the component layers for the FFN here
		"""

		# call the super constructor
		super().__init__()

		# initialize basic parameters
		self.model_dim = model_dim
		self.hidden_dim = hidden_dim
		self.device = device
		self.dtype = dtype

		# set default data type as float32
		if self.dtype is None:
			self.dtype = torch.float32

		# initialize the 3 linear layers here
		
		self.l1 = Linear(
			input_dim = self.model_dim,
			output_dim = self.hidden_dim,
			use_bias = False,
			device = device,
			dtype = dtype,
			init = init,
		)

		self.l2 = Linear(
			input_dim = self.hidden_dim,
			output_dim = self.model_dim,
			use_bias = False,
			device = device,
			dtype = dtype,
			init = init,
		)

		self.l3 = Linear(
			input_dim = self.model_dim,
			output_dim = self.hidden_dim,
			use_bias = False,
			device = device,
			dtype = dtype,
			init = init,
		)

	def forward(
		self,
		input_batch, # torch.tensor
	):

		"""
		input: batch_size x seq_len x model_dim
		output: batch_size x seq_len x model_dim
		"""

		projection_gated = self.l3(input_batch)
		projection_inner = activation_SiLU(self.l1(input_batch))

		gated_output = projection_gated * projection_inner
		output_batch = self.l2(gated_output)

		return output_batch

class DecoderOnlyTransformerBlock(torch.nn.Module):
	"""
	main class for a single block of a decoder only transformer
	this class can be simply expanded to support multiple attention types
	like mha, mqa, gqa, mhla, etc
	and also for multiple position embeddings
	like rope, sinusoidal, absolute, learned, etc
	"""

	def __init__(
		self,
		model_dim,
		hidden_dim,
		head_dim,
		num_heads,
		device = None, # either None or a torch.device object
		dtype = None, # either None or a torch.dtype object
		init = True,
	):

		"""
		initialize all the component blocks for a decoder only block
		"""

		# call the super constructor
		super().__init__()

		# initialize basic parameters
		self.model_dim = model_dim
		self.hidden_dim = hidden_dim
		self.head_dim = head_dim
		self.num_heads = num_heads
		self.device = device
		self.dtype = dtype

		# set default data type as float32
		if self.dtype is None:
			self.dtype = torch.float32

		# initialize all layers here

		# step 1: 2 x layer norms
		self.layer_norm_1 = LayerNorm_RMS(
			model_dim = self.model_dim,
			device = self.device,
			dtype = self.dtype,
			init = init,
		)

		self.layer_norm_2 = LayerNorm_RMS(
			model_dim = self.model_dim,
			device = self.device,
			dtype = self.dtype,
			init = init,
		)

		# step 2: 1 x multi head attention layer
		self.self_attn = MultiHeadAttention(
			model_dim = self.model_dim,
			head_dim = self.head_dim,
			num_heads = self.num_heads,
			causal = True,
			device = self.device,
			dtype = self.dtype,
			init = init,
		)

		# step 3: 1 x feed forward network
		self.ffn = FFN_with_SwiGLU(
			model_dim = self.model_dim,
			hidden_dim = self.hidden_dim,
			device = self.device,
			dtype = self.dtype,
			init = init,
		)

		return

	def forward(
		self,
		input_batch, # torch.tensor
		rope_module = None,
		padding_mask = None,
	):

		"""
		implement a forward pass with residual connections
		input_batch.shape = batch_size x seq_len x model_dim
		"""

		# step 1: apply layer norm 1 and self attention
		layer_norm_1_output = self.layer_norm_1(input_batch)
		self_attn_weights, self_attn_output = self.self_attn(
			layer_norm_1_output,
			rope_module = rope_module,
			padding_mask = padding_mask,
		)

		# step 2: apply the residual connection
		self_attn_output += input_batch

		# step 3: apply the layer norm 2 and ffn
		layer_norm_2_output = self.layer_norm_2(self_attn_output)
		ffn_output = self.ffn(layer_norm_2_output)

		# step 4: apply the residual connection
		ffn_output += self_attn_output

		# return self attn weights and ffn output
		return self_attn_weights, ffn_output

