"""
this script contains all the generation related functions
"""

from tqdm import tqdm

import torch

from transformer_lm.transformers_core import softmax

class GenerationMixin:
	"""
	the generation mix-in class will be inherited by
	the Decoder LM class
	this is to keep generation functions separate
	"""

	def get_total_prob_score(self, probs):
		"""
		given probabilities of the sampled tokens
		return the overall probability of the sequence
		which is essentially just the geometric mean

		input can be ... x seq_len
		"""

		# keep batch size and sequence length handy
		seq_len = probs.shape[-1]

		# each element becomes a power of seq len
		probs = torch.pow(probs, 1./seq_len)

		# now multiple along dimension -1
		return torch.prod(probs, -1)

	@torch.no_grad
	def generate_beam_search(
		self,
		input_ids,
		padding_mask = None,
		max_new_tokens = None,
		sampling = False,
		temperature = 1.0,
		top_p = None,
		top_k = None,
		num_beams = None,
		return_full_seq = False,
	):
		"""
		another very elegant implementation
		use the generate_beam_search as merely a wrapper
		and for each token, reuse the generate function for sampling
		always return num_beams amount of samples
		"""

		# base case for sanity
		if (num_beams is None) or (num_beams <= 1):
			return self.generate(
					input_ids=input_ids, padding_mask=padding_mask, max_new_tokens=max_new_tokens, 
					sampling=sampling, temperature=temperature, top_p=top_p, top_k=top_k,
					num_beams=None, num_samples=1, return_full_seq=return_full_seq
				)

		# now we will do the beam search

		# keep track of batch_size and input seq len
		batch_size = input_ids.shape[0]
		input_seq_len = input_ids.shape[1]

		# calculate the max number of tokens to be generated
		if max_new_tokens is None:
			max_new_tokens = self.max_seq_len - input_seq_len

		# upfront, we will repeat input_ids k times
		input_ids = torch.repeat_interleave(input_ids, num_beams, dim=0)
		if padding_mask is not None:
			padding_mask = torch.repeat_interleave(padding_mask, num_beams, dim=0)

		# input_ids.shape = batch_size * k x seq_len
		# padding_mask.shape = batch_size * k x seq_len

		# we keep a track of sampled probabilities as a list
		# here, instead of keeping it as a list of probabilities
		# at each token output, we'll directly maintain a tensor
		# sampled_probs.shape = batch_size x num_beams x seq_len
		# we'll start with a tensor of seq len 1 and value = 1
		
		sampled_probs = torch.ones((batch_size, num_beams, 1), dtype=torch.float32, device=input_ids.device)
		# sampled_probs.shape = batch_size x num_beams x seq_len 

		# start generating and appending to input_ids
		for _ in tqdm(range(max_new_tokens), desc="beaming..."):

			# at each step, we'll call the generate function
			# with num_beams = None, max_new_tokens = 1
			# num_samples = num_beams

			generated_data = self.generate(
					input_ids=input_ids, padding_mask=padding_mask, max_new_tokens=1,
					sampling=sampling, temperature=temperature, top_p=top_p, top_k=top_k,
					num_beams=None, num_samples=num_beams, return_full_seq=True, disable_tqdm=True
				)

			# generated data returns 3 things - output ids, attention scores, probabilities
			# we only care about output ids and probabilities
			# to return attention scores properly, we'll still reshape them correctly

			# output_ids.shape = batch_size * k x k x seq_len
			# attn_weight_list.shape = num_layers x batch_size * k x k x ...
			# sampled_probs.shape = batch_size * k x k x 1

			# we'll convert all of these to batch_size x k*k x ...
			# this will enable us to directly compare and sample

			# ======================================== #
			# reshape to help with sampling

			input_ids = generated_data["output_ids"].reshape(batch_size, num_beams * num_beams, -1)
			sampled_probs_current = generated_data["sampled_probs"].reshape(batch_size, num_beams * num_beams, 1)

			awl = generated_data["attn_weights_list"]
			awl = awl.reshape(awl.shape[0], batch_size, num_beams * num_beams, *awl.shape[3:])

			# now is the time to do crazy stuff by picking up top samples
			sampled_probs = torch.concat([torch.repeat_interleave(sampled_probs, num_beams, dim=1), sampled_probs_current], dim=-1)
			# sampled_probs.shape = batch_size x k*k x seq_len

			# ======================================== #
			# actual sampling process itself

			# now we compute overall sequence scores

			sampled_probs_scores = self.get_total_prob_score(sampled_probs)
			# sampled_probs_scores.shape = batch_size x k*k

			# now we sample top indices
			_, indices = torch.topk(sampled_probs_scores, dim=-1, k=num_beams)

			# now that we know topk indices, we bring down k*k back to k
			input_ids = torch.gather(input_ids, dim=1, index=indices.unsqueeze(-1).expand(-1, -1, input_ids.shape[-1]))
			sampled_probs = torch.gather(sampled_probs, dim=1, index=indices.unsqueeze(-1).expand(-1, -1, sampled_probs.shape[-1]))

			# disclosure: the syntax and the fxns to be used to do this selection was found online
			# now we deal with the attention weights list
			dim = 2
			idx = indices.unsqueeze(0)
			idx = idx.reshape(*idx.shape[:dim], *idx.shape[dim:], *([1] * (awl.ndim - idx.ndim)))
			idx = idx.expand(*awl.shape[:dim], indices.shape[-1], *awl.shape[dim + 1:])
			awl = torch.gather(awl, dim=dim, index=idx)

			# ======================================== #
			# move input_ids back to actual shape
			input_ids = input_ids.reshape(batch_size * num_beams, -1)

			# append 1s to padding mask
			if padding_mask is not None:
				padding_mask = torch.concat(
					[
						padding_mask,
						torch.ones((batch_size * num_beams, 1)).to(padding_mask.device)
					],
					dim=1,
				)

			# ======================================== #
			# ======================================== #

		# trim input ids if needed
		if return_full_seq is False:
			input_ids = input_ids[:, input_seq_len:]

		# get input_ids to correct returnable shape
		input_ids = input_ids.reshape((batch_size, num_beams, -1))
		sampled_probs = sampled_probs[:, :, 1:] # drop the extra 1s added for ease

		return {
			"attn_weights_list": awl,
			"sampled_probs": sampled_probs,
			"output_ids": input_ids,
		}

	@torch.no_grad
	def sampling_top_p_top_k(
		self,
		probs,
		top_p = None,
		top_k = None,
	):
		"""
		given probabilities from the model
		apply top-p and top-k sampling
		and return the sampled token ids
		probs.shape = batch_size x vocab_size
		"""

		# base case if both top-p and top-k are None
		if top_p is None and top_k is None:
			sampled_ids = torch.multinomial(probs, num_samples=1).squeeze(-1)
			return sampled_ids

		# at this point at least one of top-p and top-k is not None
		
		# if top k is None, then set it to vocab size
		if top_k is None:
			top_k = probs.shape[1]

		# get top-k probabilities and their indices
		topk_probs, topk_indices = torch.topk(probs, top_k, dim=-1, sorted=True)

		# if top-p is not None, then apply masking shenanigans
		if top_p is not None:

			# get the cumulative probability sum for top-k probabilities
			cumulative_probability_sum = torch.cumsum(topk_probs, dim=-1)

			# apply mask to the point where it crosses threshold
			top_p_mask = cumulative_probability_sum > top_p

			# actually we want to apply the mask AFTER the crossing point
			# so we will shuffle the mask right by 1
			top_p_mask[:, 1:] = top_p_mask[:, :-1].clone()

			# and we want at least one token to have weight
			top_p_mask[:, 0] = False

			# set the probabilities for the mask as 0
			topk_probs[top_p_mask] = 0.

		# re-normalize the topk probabilities
		topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)

		# now sample indices from top k probs
		sampled_idxs = torch.multinomial(topk_probs, num_samples=1)

		# now we will gather values along the -1 dimension on top k indices
		# top k indices = batch_size x k
		# we have index into top k, which we will gather for each batch
		sampled_ids = torch.gather(topk_indices, -1, sampled_idxs).squeeze(-1)

		return sampled_ids

	@torch.no_grad
	def generate(
		self,
		input_ids,
		padding_mask = None,
		max_new_tokens = None,
		sampling = False,
		temperature = 1.0,
		top_p = None,
		top_k = None,
		num_beams = None,
		num_samples = 1,
		return_full_seq = False,
		disable_tqdm = False,
	):
		"""
		write an elegantly beautiful generate function
		support
		- max_new_tokens
		- temperature
		- top_p
		- top_k
		- num_samples
		- stop_token [pending]
		- num_beams [pending]
		"""

		# base case for beam search handled upfront
		if (num_beams is not None) and (num_beams > 1):
			return self.generate_beam_search(
					input_ids=input_ids, padding_mask=padding_mask, max_new_tokens=max_new_tokens, 
					sampling=sampling, temperature=temperature, top_p=top_p, top_k=top_k,
					num_beams=num_beams, return_full_seq=return_full_seq
				)


		# input_ids=input_ids.shape = batch_size x seq_len
		# padding_mask.shape = batch_size x seq_len

		# keep track of batch_size and input seq len
		batch_size = input_ids.shape[0]
		input_seq_len = input_ids.shape[1]

		# calculate the max number of tokens to be generated
		if max_new_tokens is None:
			max_new_tokens = self.max_seq_len - input_seq_len

		# since we want to support num samples,
		# we'll repeat interleaved along batch size
		input_ids = torch.repeat_interleave(input_ids, num_samples, dim=0)
		if padding_mask is not None:
			padding_mask = torch.repeat_interleave(padding_mask, num_samples, dim=0)

		# keep track of sampled probabilities
		sampled_probs = []

		# start generating and appending to input_ids
		for _ in tqdm(range(max_new_tokens), desc="generating...", disable=disable_tqdm):

			# do a forward pass to get output logits
			attn_weights_list, logits = self.forward(input_ids, padding_mask)
			# logits.shape = batch_size x seq_len_current x vocab_size

			logits = logits[:, -1, :]
			# logits.shape = batch_size x vocab_size

			# compute softmax here (after projection to fp32)
			probs = softmax(logits.to(torch.float32), -1, temperature)

			# sample token ids here
			sampled_ids = None
			if sampling is False:
				sampled_ids = torch.argmax(probs, -1)
			else:
				sampled_ids = self.sampling_top_p_top_k(probs, top_p, top_k)

			# sampled_ids.shape = batch_size
		
			# append sampled_ids to input_ids
			input_ids = torch.concat(
				[
					input_ids,
					sampled_ids.unsqueeze(1)
				],
				dim=1,
			)

			# append 1s to padding mask
			if padding_mask is not None:
				padding_mask = torch.concat(
					[
						padding_mask,
						torch.ones((batch_size * num_samples, 1)).to(padding_mask.device)
					],
					dim=1,
				)

			# append probs to sampled_probs
			sampled_probs.append(torch.gather(probs, -1, sampled_ids.unsqueeze(1)))

		# trim input ids if needed
		if return_full_seq is False:
			input_ids = input_ids[:, input_seq_len:]

		# now we rearrange the data back into num_samples zone

		# first, the output data
		input_ids = input_ids.reshape(batch_size, num_samples, -1)

		# then, the probabilities
		sampled_probs = torch.concat(sampled_probs, -1).reshape(batch_size, num_samples, -1)

		# finally the attn_weight_list
		# current shape is num_layers x batch_size * num_samples x num_heads x seq_len x seq_len
		# we want it to be num_layers x batch_size x num_samples x num_heads x seq_len x seq_len
		attn_weights_list = attn_weights_list.reshape(attn_weights_list.shape[0], batch_size, num_samples, *attn_weights_list.shape[2:])

		return {
			"attn_weights_list": attn_weights_list,
			"sampled_probs": sampled_probs,
			"output_ids": input_ids,
		}
