"""
this script contains gpt style decoder-only transformer setup
"""

from pathlib import Path
import os
import json

from tqdm import tqdm

import torch

from transformer_lm.embeddings import TokenEmbeddings, RotaryPositionEmbeddings
from transformer_lm.transformers_core import DecoderOnlyTransformerBlock, LayerNorm_RMS, Linear
from transformer_lm.transformers_core import get_token_positions_from_padding_mask
from transformer_lm.generation import GenerationMixin

class Decoder_LM(GenerationMixin, torch.nn.Module):
	"""
	main class for decoder-only transformer based language models
	
	components:
	- initialize from scratch
	- load from checkpoint
	- save to a checkpoint

	"""

	def __init__(
		self,
		vocab_size,
		model_dim,
		hidden_dim,
		head_dim,
		latent_attn_dim = None,
		num_heads = 1,
		num_attn_groups = 1,
		num_layers = 4,
		num_loops = 1, # for looped transformer implementation
		max_seq_len = 4092,
		posn_embeddings = "rope",
		attn_type = "sdpa",
		device = None, # either None or a torch.device object
		dtype = None, # either None or a torch.dtype object
		init = True,
	):

		"""
		initialize all components needed for a decoder only language model
		"""

		# call the super constructor
		super().__init__()

		# initialize basic parameters
		self.vocab_size = vocab_size
		self.model_dim = model_dim
		self.hidden_dim = hidden_dim
		self.head_dim = head_dim
		self.latent_attn_dim = latent_attn_dim
		self.num_heads = num_heads
		self.num_attn_groups = num_attn_groups
		self.num_layers = num_layers
		self.num_loops = num_loops
		self.max_seq_len = max_seq_len
		self.posn_embeddings = posn_embeddings
		self.attn_type = attn_type
		self.device = device
		self.dtype = dtype

		# set default data type as float32
		if self.dtype is None:
			self.dtype = torch.float32

		# make sure posn embeddings are supported
		assert self.posn_embeddings in ["sinusoidal", "rope", "nope"]

		# initialize the token embeddings module here
		self.token_embeddings_module = TokenEmbeddings(
			vocab_size = self.vocab_size,
			model_dim = self.model_dim,
			posn_embeddings = (self.posn_embeddings == "sinusoidal"),
			max_seq_len = self.max_seq_len,
			device = self.device,
			dtype = self.dtype,
			init = init,
		)

		# add rope embeddings
		self.rope_module = None
		if self.posn_embeddings == "rope":
			self.rope_module = RotaryPositionEmbeddings(
				head_dim = self.head_dim,
				max_seq_len = self.max_seq_len,
				device = self.device,
				dtype = self.dtype,
				init = init,
			)

		# add n decoder only layers here
		self.model = torch.nn.ModuleList([
			DecoderOnlyTransformerBlock(
				model_dim = self.model_dim,
				hidden_dim = self.hidden_dim,
				head_dim = self.head_dim,
				latent_attn_dim = self.latent_attn_dim,
				num_heads = self.num_heads,
				num_attn_groups = self.num_attn_groups,
				attn_type = self.attn_type,
				device = self.device,
				dtype = self.dtype,
				init = init,
			)
			for _ in range(num_layers)
		])

		# add a layer norm at the end
		self.layer_norm = LayerNorm_RMS(
			model_dim = self.model_dim,
			device = self.device,
			dtype = self.dtype,
			init = init,
		)

		# add a language model head
		self.lm_head = Linear(
			input_dim = self.model_dim,
			output_dim = self.vocab_size,
			use_bias = True,
			device = self.device,
			dtype = self.dtype,
			init = init,
		)

		return

	def forward(
		self,
		input_ids,
		padding_mask = None,
		token_positions = None, # only needed for KV cache, otherwise computed here
	):

		"""
		the forward pass takes as input
		input_ids : batch_size x sequence_length
		padding_mask : batch_size x sequence_length (optional)
		output : batch_size x sequence_length x vocab_size
		attn_weights.shape = num_layers x batch_size x num_heads x sequence_length x sequence_length
		"""

		# let's track the current size of output tensor
		# input_ids : batch_size x sequence_length

		batch_size = input_ids.shape[0]
		seq_len = input_ids.shape[1]

		# if token_positions is None (which unless KV cache it should be)
		# then generate token positions based on padding mask
		if token_positions is None and padding_mask is not None:
			token_positions = get_token_positions_from_padding_mask(padding_mask)

		# step 1: token embeddings module
		output_tensor = self.token_embeddings_module(input_ids, token_positions)
		# output_tensor : batch_size x sequence_length x model_dim

		# step 2: iterate over all decoder only layers
		attn_weights_list = []
		for _ in range(self.num_loops):
			for layer in self.model:
				attn_weights, output_tensor = layer(
					output_tensor,
					rope_module = self.rope_module,
					padding_mask = padding_mask,
					token_positions = token_positions,
				)
				attn_weights_list.append(attn_weights)

		# step 3: final layer norm
		output_tensor = self.layer_norm(output_tensor)

		# step 4: apply lm head
		output_tensor = self.lm_head(output_tensor)

		return torch.stack(attn_weights_list), output_tensor

	@classmethod
	def load_model(
		cls,
		dir_path,
		device = None,
	):
		"""
		class method to load the model before running __init__
		"""

		# step 1: make sure path exists
		assert Path(dir_path).is_dir()

		# step 2: load config file
		with open(os.path.join(dir_path, "model_config.json"), "r") as f:
			model_config = json.loads("".join(f.readlines()))

		# print the model config being loaded
		print("\n" + "="*50)
		print(f"loading model from config")
		print(json.dumps(model_config, indent=4))		
		print("="*50 + "\n")

		# using eval is slightly unsafe as it can execute arbitrary code
		model_config["dtype"] = eval(model_config["dtype"])

		# step 3: load a model using the model_config
		model = cls(**model_config, device=device, init=False)

		# step 4: load state dict
		model.load_state_dict(
			torch.load(
				os.path.join(dir_path, "model_state_dict.pt"),
				map_location = device if device is not None else "cpu",
			)
		)

		print(f"\n> model loaded from {dir_path}\n")

		return model

	def save_model(
		self,
		dir_path,
	):
		"""
		save the current state of the model as it is
		saves the config file
		saves the model state dictionary
		"""

		# step 1: create the directory if it doesn't exist
		Path(dir_path).mkdir(exist_ok=True, parents=True)

		# step 2: now create the dict for config and save
		model_config = {
			"vocab_size": self.vocab_size,
			"model_dim": self.model_dim,
			"hidden_dim": self.hidden_dim,
			"head_dim": self.head_dim,
			"latent_attn_dim": self.latent_attn_dim,
			"num_heads": self.num_heads,
			"num_attn_groups": self.num_attn_groups,
			"num_layers": self.num_layers,
			"num_loops": self.num_loops,
			"max_seq_len": self.max_seq_len,
			"posn_embeddings": self.posn_embeddings,
			"attn_type": self.attn_type,
			"dtype": str(self.dtype),
		}

		# now save the config
		with open(os.path.join(dir_path, "model_config.json"), "w+") as f:
			f.write(json.dumps(model_config, indent=4))

		# step 3: save the state dictionary
		torch.save(self.state_dict(), os.path.join(dir_path, "model_state_dict.pt"))

		print(f"\n> model saved to {dir_path}\n")

		return
