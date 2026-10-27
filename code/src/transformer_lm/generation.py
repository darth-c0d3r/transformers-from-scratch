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

	@torch.no_grad()
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
		diverse_sampling = False,
		return_full_seq = False,
	):
		"""
		returns a dict with 3 keys (same as .generate)
		
		return {
			"attn_weights_list": attn_weights_list,
			"sampled_probs": sampled_probs,
			"output_ids": input_ids,
		}

		attn_weights_list.shape = num_layers x batch_size x num_samples x num_heads x seq_len x seq_len
		sampled_probs.shape = batch_size x num_samples x seq_len
		output_ids = batch_size x num_samples x seq_len

		one of the problems with beam search could be
		that while we're sampling, the k highest scoring sequences might only differ in final position
		and that too because we're forcing k different samples at each step with top-k

		so for now, our diverse implementation is to have k branches and then expand each
		but rather than taking a global top-k, take top-1 per branch (if diverse is set to true)

		in general, there are other ways to implement diversity
		but since this is a learning exercise, we'll take this approach

		"""

		# base case - if num beams is 0 or None
		if (num_beams is None) or (num_beams <= 1):
			return self.generate(
					input_ids=input_ids, padding_mask=padding_mask, max_new_tokens=max_new_tokens,
					sampling=sampling, temperature=temperature, top_p=top_p, top_k=top_k, num_beams=None,
					num_samples=1, return_full_seq=return_full_seq
				)

		# if sampling is true, just print a warning
		if sampling is True:
			print(f"> sampling not supported for beam search; proceeding with greedy beam search;")

		# input_ids.shape = batch_size x seq_len
		# padding_mask.shape = batch_size x seq_len

		# keep track of batch_size and input seq len
		batch_size = input_ids.shape[0]
		input_seq_len = input_ids.shape[1]

		# calculate the max number of tokens to be generated
		if max_new_tokens is None:
			max_new_tokens = self.max_seq_len - input_seq_len

		# initially, sampled probabilities is None
		# at any other point, this is batch_size * k x seq_len
		sampled_probs = None

		# start generating and appending to input_ids
		for step_idx in tqdm(range(max_new_tokens), desc="beaming...", disable=False):

			# ============================================================ #
			# BASIC FWD PASS AND SAMPLING

			# do a forward pass to get output logits
			attn_weights_list, logits = self.forward(input_ids, padding_mask)

			# attn_weights_list.shape = num_layers x batch_size * k x num_heads x seq_len x seq_len
			# logits.shape = batch_size * k x seq_len_current x vocab_size

			logits = logits[:, -1, :]
			# logits.shape = batch_size * k x vocab_size

			# compute softmax here (after projection to fp32)
			probs = softmax(logits.to(torch.float32), -1, temperature)

			# now, for each input element, we'll select top k elements
			# get top-k probabilities and their indices
			topk_probs, topk_indices = torch.topk(probs, num_beams, dim=-1, sorted=True)
			# topk.shape = batch_size * k x k

			# ============================================================ #
			# SPECIAL PROCESSING FOR STEP 0

			# do the special case processing for step 0
			if step_idx == 0:

				# update input_ids and padding mask

				# batch_size x seq_len --> batch_size * k x seq_len
				input_ids = torch.repeat_interleave(input_ids, num_beams, dim=0)
				input_ids = torch.cat([input_ids, topk_indices.reshape((batch_size * num_beams, 1))], dim=-1)

				# update padding mask
				if padding_mask is not None:
					padding_mask = torch.repeat_interleave(padding_mask, num_beams, dim=0)
					padding_mask = torch.cat([padding_mask, torch.ones((batch_size * num_beams, 1)).to(padding_mask.device)], dim=-1)

				# set up sampled_probs
				sampled_probs = topk_probs.reshape((batch_size * num_beams, 1))

				# update attn_weights_list but that's only needed in case we want to return
				attn_weights_list = torch.repeat_interleave(attn_weights_list.unsqueeze(2), num_beams, dim=2)

				continue

			# ============================================================ #
			# ADDING SAMPLED DATA BACK TO INPUT DATA

			# we'll need to update input_ids, sampled_probs, attn_weights_list
			# and we'll add 1s to padding mask as well

			# input_ids --> batch_size * k x seq_len
			# topk_indices --> batch_size * k x k
			# we want --> batch_size x k * k x (seq_len + 1)

			input_ids = torch.repeat_interleave(input_ids.reshape((batch_size, num_beams, -1)), num_beams, dim=1)
			input_ids = torch.cat([input_ids, topk_indices.reshape((batch_size, num_beams * num_beams, 1))], dim=-1)

			# sampled_probs --> batch_size * k x seq_len
			# topk_probs --> batch_size * k x k
			# we want --> batch_size x k * k x (seq_len + 1)

			sampled_probs = torch.repeat_interleave(sampled_probs.reshape((batch_size, num_beams, -1)), num_beams, dim=1)
			sampled_probs = torch.cat([sampled_probs, topk_probs.reshape((batch_size, num_beams * num_beams, 1))], dim=-1)

			# attn_weights_list --> num_layers x batch_size * k x num_heads x seq_len x seq_len
			# we want --> num_layers x batch_size x batch_size x k * k x num_heads x seq_len x seq_len
			awl_shape = attn_weights_list.shape
			attn_weights_list = attn_weights_list.reshape((awl_shape[0], batch_size, num_beams, *awl_shape[2:]))
			attn_weights_list = torch.repeat_interleave(attn_weights_list, num_beams, dim=2)

			# ============================================================ #
			# PICKING ONLY TOP K SAMPLES AND RESTORING SANITY

			# we'll get sequence probabilities
			total_prob_scores = self.get_total_prob_score(sampled_probs)
			# total_prob_scores.shape = batch_size x k * k

			# take top k indices here (add diversity option here)

			# diversity sampling push
			if diverse_sampling is True:

				# if diverse sampling is true,
				# we want to pick the top token from each sequence
				# we have k * k probability scores
				# sets of k correspond to each input (sorted)
				# we want to set evry other score as 0
				diverse_mask = torch.zeros_like(total_prob_scores)
				diverse_mask[:,::num_beams] = 1.

				total_prob_scores *= diverse_mask


			# get top-k probabilities and their indices
			_, topk_indices = torch.topk(total_prob_scores, num_beams, dim=-1, sorted=True)
			# topk.shape = batch_size x k

			# now we'll pick top k indices for
			# input_ids, sampled_probs, attn_weights_list

			# input_ids --> batch_size x k * k x (seq_len + 1)
			# we want --> batch_size x k x (seq_len + 1)
			# and then --> batch_size * k x (seq_len + 1)
			index = torch.repeat_interleave(topk_indices.unsqueeze(-1), input_ids.shape[-1], dim=-1)
			input_ids = torch.gather(input_ids, index=index, dim=1)
			input_ids = input_ids.reshape((batch_size * num_beams, -1))

			# sampled_probs --> batch_size x k * k x (seq_len + 1)
			# we want --> batch_size x k x (seq_len + 1)
			# and then --> batch_size * k x (seq_len + 1)
			index = torch.repeat_interleave(topk_indices.unsqueeze(-1), sampled_probs.shape[-1], dim=-1)
			sampled_probs = torch.gather(sampled_probs, index=index, dim=1)
			sampled_probs = sampled_probs.reshape((batch_size * num_beams, -1))

			# attn_weights_list --> num_layers x batch_size x k * k x num_heads x seq_len x seq_len
			# we want --> num_layers x batch_size x k x num_heads x seq_len x seq_len
			awl_shape = attn_weights_list.shape
			index = topk_indices[None, :, :, None, None, None]
			index = index.expand(awl_shape[0], -1, -1, *awl_shape[3:])
			attn_weights_list = torch.gather(attn_weights_list, index=index, dim=2)

			# padding mask --> simply add 1s
			if padding_mask is not None:
				padding_mask = torch.cat([padding_mask, torch.ones((batch_size * num_beams, 1)).to(padding_mask.device)], dim=-1)

			# now we're ready for the next round again

		# ============================================================ #
		# after finishing, convert data to required output format

		# trim input ids if needed
		if return_full_seq is False:
			input_ids = input_ids[:, input_seq_len:]

		# now we rearrange the data back into num_samples zone

		# first, the output data
		input_ids = input_ids.reshape(batch_size, num_beams, -1)

		# then, the probabilities
		sampled_probs = sampled_probs.reshape(batch_size, num_beams, -1)

		return {
			"attn_weights_list": attn_weights_list,
			"sampled_probs": sampled_probs,
			"output_ids": input_ids,
		}

	@torch.no_grad()
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

	@torch.no_grad()
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
		- num_beams
		"""

		# base case for beam search handled upfront
		if (num_beams is not None) and (num_beams > 1):
			return self.generate_beam_search(
					input_ids=input_ids, padding_mask=padding_mask, max_new_tokens=max_new_tokens, 
					sampling=sampling, temperature=temperature, top_p=top_p, top_k=top_k,
					num_beams=num_beams, diverse_sampling=sampling, return_full_seq=return_full_seq
				)


		# input_ids.shape = batch_size x seq_len
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
		attn_weights_list = None

		# start generating and appending to input_ids
		for _ in tqdm(range(max_new_tokens), desc="generating...", disable=False):

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
