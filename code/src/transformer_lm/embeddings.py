"""
this script contains class for computing embeddings
includes both token and position embeddings
- token embeddings
- sinusoidal position embeddings (inside token embeddings)
- rotary position embeddings
- support for custom position ids for kv caching or left padding

IMPORTANT TO NOTE
the feature option to squeeze later is not ideal
ideally, you should call the function with the right shapes
in test.py, we will build it with correct behaviour

"""

import torch

class RotaryPositionEmbeddings(torch.nn.Module):
	"""
	this class implements the RoPE embeddings
	creating this as a sub-class of nn.Module
	even though this has no trainable parameters

	rotary embeddings are applied to each heads separately
	so the dimension is head_dim as opposed to model_dim
	"""

	def __init__(
		self,
		head_dim,
		max_seq_len,
		wavelength = 10000, # as used in the rope paper
		device = None, # either None or a torch.device object
		dtype = torch.float32, # either None or a torch.dtype object
		init = True, # just here for consistency, not actually used
	):

		"""
		initialize the rotary position embeddings module
		"""

		# call the super constructor
		super().__init__()

		# initialize basic parameters
		self.head_dim = head_dim
		self.max_seq_len = max_seq_len
		self.wavelength = wavelength
		self.device = device
		self.dtype = dtype

		# just for rotary embeddings layer, this is needed
		assert self.head_dim % 2 == 0

		# we need to construct a 2D matrix here
		# max_seq_len x head_dim
		# p / exp(wavelength, 2i/d)
		# p --> max_seq_len
		# i --> head_dim / 2

		# max_seq_len x 1
		position_index = torch.arange(self.max_seq_len, dtype=torch.float32).view((-1, 1))

		# 1 x head_dim
		dimension_index = torch.arange(self.head_dim // 2, dtype=torch.float32).repeat_interleave(2).view((1, -1))
		dimension_index = -1. * dimension_index / (self.head_dim / 2.)
		dimension_index = torch.pow(self.wavelength, dimension_index)

		# temporarily put position index and dimension index to device
		if self.device is not None:
			position_index = position_index.to(self.device)
			dimension_index = dimension_index.to(self.device)

		# get the 2d mesh matrix
		position_dimension_matrix = position_index @ dimension_index

		# get the cosine matrix mesh
		cosine_mesh = torch.cos(position_dimension_matrix).to(self.dtype)

		# get the sine matrix mesh
		sine_mesh = torch.sin(position_dimension_matrix).to(self.dtype)

		# register the meshes as buffers
		# registering as buffer moves them around with .to(device) or .to(dtype)
		# persistent = False means not included in state dictionary
		self.register_buffer("cosine_mesh", cosine_mesh, persistent=False)
		self.register_buffer("sine_mesh", sine_mesh, persistent=False)

		return

	def forward(
		self,
		input_batch, # torch.tensor
		token_positions = None,
	):
		"""
		we may not always want to assume that input_batch starts from 0
		if the sequence is left padded, then there'd be a bunch of 0s
		at the start before position 1 comes in
		similarly, if we want to apply rope only to the final token
		and rest all is already kv cached
		then also we'll need token_positions
		if token_positions is not supplied
		then assume it is 0 to L-1
		"""

		# input batch should ideally be batch_size x seq_len x head_dim
		# but there could be cases where it is just batch_size x head_dim
		# especially when applying only to last token
		# so we'll build our function assuming that it is ... x head_dim
		# and we'll also assume that token_positions will be of the shape ...
		# token_positions won't have the dimension head_dim

		# handle the case when there's only 2 dimensions provided
		squeeze_later = False
		if len(input_batch.shape) == 2: # only 2 dimensions, batch_size x head_dim
			print(f"> warning: squeeze later functionality is unstable; ideally don't use it")
			input_batch = input_batch.unsqueeze(1)
			if token_positions is not None:
				token_positions = token_positions.unsqueeze(1)
			squeeze_later = True

		# handle the case if token positions is None
		# -2 is seq_len
		if token_positions is None:
			token_positions = torch.arange(input_batch.shape[-2]).repeat(input_batch.shape[:-2] + tuple([1]))
		token_positions = token_positions.to(input_batch.device).long()
		
		# at this point
		# input_batch.shape = ... x batch_size x seq_len x head_dim
		# token_positions.shape = ... x batch_size x seq_len

		# get the cosine and sine meshes
		selected_cosine_mesh = self.cosine_mesh[token_positions]
		selected_sine_mesh = self.sine_mesh[token_positions]

		# at this point
		# mesh.shape = ... x batch_size x seq_len x head_dim

		# the final formula for computing embeddings is
		# cosine mesh x input_batch + sine_mesh x swapped(input_batch)
		# where swapping means we swap every pair of points around
		# in the final dimension and negate every even index

		# the swapping part and neg
		# swapped_batch[:,:,2i] =  -1 * input_batch[:,:,2i+1]
		# swapped_batch[:,:,2i+1] = input_batch[:,:,2i]

		swapped_batch = torch.empty(input_batch.shape, dtype=self.dtype, device=input_batch.device)
		swapped_batch[..., 0::2] = -1 * input_batch[..., 1::2]
		swapped_batch[..., 1::2] = input_batch[..., 0::2]

		# now finally get the output batch
		output_batch = (selected_cosine_mesh * input_batch) + (selected_sine_mesh * swapped_batch)

		# if squeezing later is needed
		if squeeze_later is True:
			output_batch = output_batch.squeeze(1)

		return output_batch


class SinusoidalPositionEmbeddings(torch.nn.Module):
	"""
	this class implements sinusoidal position embeddings

	sinusoidal position embeddings are used as part of token embeddings
	and we'll need to take care of starting position due to padding
	"""

	def __init__(
		self,
		model_dim,
		max_seq_len,
		wavelength = 10000, # a decent default value, same as RoPE
		device = None, # either None or a torch.device object
		dtype = torch.float32, # either None or a torch.dtype object
		init = True, # just here for consistency, not actually used
	):

		"""
		initialize the sinusoidal position embeddings module
		"""

		# call the super constructor
		super().__init__()

		# initialize basic parameters
		self.model_dim = model_dim
		self.max_seq_len = max_seq_len
		self.wavelength = wavelength
		self.device = device
		self.dtype = dtype

		# just for position embeddings layer, this is needed
		# actually this is not really needed, just for simplicity
		assert self.model_dim % 2 == 0

		# we need to construct a 2D matrix here
		# max_seq_len x head_dim
		# p / exp(wavelength, 2i/d)
		# p --> max_seq_len
		# i --> head_dim / 2
		# on this mesh apply cosine and sine functions

		# max_seq_len x 1
		position_index = torch.arange(self.max_seq_len, dtype=torch.float32).view((-1, 1))

		# 1 x model_dim // 2
		dimension_index = torch.arange(self.model_dim // 2, dtype=torch.float32).view((1, -1))
		dimension_index = -1. * dimension_index / (self.model_dim / 2.)
		dimension_index = torch.pow(self.wavelength, dimension_index)

		# temporarily put position index and dimension index to device
		if self.device is not None:
			position_index = position_index.to(self.device)
			dimension_index = dimension_index.to(self.device)

		# get the 2d mesh matrix
		position_dimension_matrix = position_index @ dimension_index

		# get the cosine matrix mesh
		cosine_mesh = torch.cos(position_dimension_matrix).to(self.dtype)

		# get the sine matrix mesh
		sine_mesh = torch.sin(position_dimension_matrix).to(self.dtype)

		# now we leave to interweave the two
		# one from cosine and other from sine
		embeddings_mesh = torch.empty((self.max_seq_len, self.model_dim), device=self.device, dtype=self.dtype)
		embeddings_mesh[:, 0::2] = sine_mesh
		embeddings_mesh[:, 1::2] = cosine_mesh
		
		# register the meshes as buffers
		# registering as buffer moves them around with .to(device) or .to(dtype)
		# persistent = False means not included in state dictionary
		self.register_buffer("embeddings_mesh", embeddings_mesh, persistent=False)

		return

	def forward(
		self,
		input_batch, # torch.tensor
		token_positions = None,
	):
		"""
		we may not always want to assume that input_batch starts from 0
		if the sequence is left padded, then there'd be a bunch of 0s
		at the start before position 1 comes in
		similarly, if we want to apply posn embeddings only to the final token
		and rest all is already kv cached
		then also we'll need token_positions
		if token_positions is not supplied
		then assume it is 0 to L-1
		"""

		# input batch should ideally be batch_size x seq_len x model_dim
		# but there could be cases where it is just batch_size x model_dim
		# especially when applying only to last token
		# so we'll build our function assuming that it is ... x model_dim
		# and we'll also assume that token_positions will be of the shape ...
		# token_positions won't have the dimension model_dim

		# handle the case when there's only 2 dimensions provided
		squeeze_later = False
		if len(input_batch.shape) == 2: # only 2 dimensions, batch_size x model_dim
			print(f"> warning: squeeze later functionality is unstable; ideally don't use it")
			input_batch = input_batch.unsqueeze(1)
			if token_positions is not None:
				token_positions = token_positions.unsqueeze(1)
			squeeze_later = True

		# handle the case if token positions is None
		# -2 is seq_len
		if token_positions is None:
			token_positions = torch.arange(input_batch.shape[-2]).repeat(input_batch.shape[:-2] + tuple([1]))
		token_positions = token_positions.to(input_batch.device).long()
		
		# at this point
		# input_batch.shape = batch_size x seq_len x model_dim
		# token_positions.shape = batch_size x seq_len

		# get the selected embeddings mesh
		selected_embeddings_mesh = self.embeddings_mesh[token_positions]

		# at this point
		# mesh.shape = batch_size x seq_len x model_dim
		# this is same as input_batch

		# the final formula for computing embeddings is
		# simply add it to input batch
		output_batch = input_batch + selected_embeddings_mesh

		# if squeezing later is needed
		if squeeze_later is True:
			output_batch = output_batch.squeeze(1)

		return output_batch


class TokenEmbeddings(torch.nn.Module):
	"""
	main class for computing input token embeddings
	"""

	def __init__(
		self,
		vocab_size,
		model_dim,
		posn_embeddings = False, # do we want to add sinusoidal embeddings
		max_seq_len = None, # needed if using posn embeddings
		device = None, # either None or a torch.device object
		dtype = torch.float32, # either None or a torch.dtype object
		init = True,
	):

		"""
		initialize a default embeddings module
		"""

		# call the super constructor
		super().__init__()

		# initialize basic parameters
		self.vocab_size = vocab_size
		self.model_dim = model_dim
		self.posn_embeddings = posn_embeddings
		self.max_seq_len = max_seq_len
		self.device = device
		self.dtype = dtype

		# initialize an empty embeddings matrix
		self.embeddings_matrix = torch.empty((self.vocab_size, self.model_dim),
			dtype=self.dtype, device=self.device)

		# do a truncated normal distribution initialization if needed
		if init is True:
			self.embeddings_matrix = torch.nn.init.trunc_normal_(
				self.embeddings_matrix,
				mean = 0.,
				std = 1.,
				a = -3.,
				b = 3.,
			)

		# cast the embeddings matrix to nn.Parameter (for training)
		self.embeddings_matrix = torch.nn.Parameter(self.embeddings_matrix)

		# if we want to add posn embeddings, add module
		self.sinusoidal_position_embeddings = None

		if self.posn_embeddings:
			
			assert self.max_seq_len is not None, "max_seq_len not provided for position embeddings"

			self.sinusoidal_position_embeddings = SinusoidalPositionEmbeddings(
				model_dim = self.model_dim,
				max_seq_len = self.max_seq_len,
				device = self.device,
				dtype = self.dtype,
				init = init,
			)

		return

	def forward(
		self,
		token_ids, # torch.tensor
		token_positions = None, # needed for position embeddings, optional
	):

		"""
		input: batch_size x sequence_length
		output: batch_size x sequence_length x model_dim

		in case the input is just batch size (say, when only using for final token in kv cache enabled)
		input: batch_size
		output: batch_size x model_dim
		"""

		# handle the case when there's only 1 dimension provided
		squeeze_later = False
		if len(token_ids.shape) == 1: # only 1 dimensions, batch_size
			token_ids = token_ids.unsqueeze(1)
			if token_positions is not None:
				token_positions = token_positions.unsqueeze(1)
			squeeze_later = True

		# flatten the token ids to get a list
		# use this to index into embeddings matrix
		# use the original token ids shape to remould

		batch_size = token_ids.shape[0]
		seq_len = token_ids.shape[1]

		# get the token embeddings
		token_embeddings = self.embeddings_matrix[token_ids.contiguous().view(-1)].view((batch_size, seq_len, self.model_dim))

		# add position embeddings if needed
		if self.posn_embeddings:
			token_embeddings = self.sinusoidal_position_embeddings(token_embeddings, token_positions)

		# if squeezing later is needed
		if squeeze_later is True:
			token_embeddings = token_embeddings.squeeze(1)

		return token_embeddings
