"""
this script contains various loss/eval functions

CrossEntropyLoss
Perplexity
Probability

"""

import sys
sys.path.insert(1, "../")

import torch
from transformer_lm.transformers_core import softmax

def CrossEntropyLoss(logits, targets, loss_mask=None):
	"""
	calculates the avg cross entropy loss
	
	logits.shape = ... x seq_len x vocab_size
	targets.shape = ... x seq_len
	loss_mask.shape = ... x seq_len

	"""

	# if loss mask is None, it is equivalent to ones
	if loss_mask is None:
		loss_mask = torch.ones(targets.shape, dtype=logits.dtype).to(targets.device)

	# for stability of the exp operation
	# subtract the highest element along the vocab_size dimension
	# from all logits, ensuring highest logit = 0
	logits_subtracted = logits - torch.max(logits, -1).values.unsqueeze(-1)

	# get the exp component, shape = ... x seq_len
	# exp_component = torch.logsumexp(logits, dim=-1)
	exp_component = torch.log(torch.sum(torch.exp(logits_subtracted), -1))

	# get the indexed logits component, shape = ... x seq_len
	logits_component = torch.gather(logits_subtracted, -1, targets.unsqueeze(-1)).squeeze(-1)

	# get the loss per token, shape = ... x seq_len
	loss_per_token = exp_component - logits_component

	# now some of these tokens would be masked out
	loss_per_token = loss_per_token * loss_mask

	return torch.sum(loss_per_token) / torch.sum(loss_mask)

def Perplexity(logits, targets, loss_mask=None):
	"""
	calculates the avg perplexity

	logits.shape = ... x seq_len x vocab_size
	targets.shape = ... x seq_len
	loss_mask.shape = ... x seq_len

	"""

	# if loss mask is None, it is equivalent to ones
	if loss_mask is None:
		loss_mask = torch.ones(targets.shape, dtype=logits.dtype).to(targets.device)

	# we can now apply softmax on logits
	# although this is bloated, we only need value for targets[i][j]
	# but it is okay to have a neat and clean solution
	logits_softmax = softmax(logits, -1)

	# now apply gather and only keep the target indices, shape = ... x seq_len
	logits_softmax = torch.gather(logits_softmax, -1, targets.unsqueeze(-1)).squeeze(-1)

	# now is the time to apply loss mask
	logits_softmax = logits_softmax.masked_fill(loss_mask == 0, 1)

	# raise to the power of -1/m
	sequence_lengths = torch.sum(loss_mask, -1).unsqueeze(-1)
	perplexity_per_token = torch.pow(logits_softmax, -1.0 / sequence_lengths)
	perplexity_per_sequence = torch.prod(perplexity_per_token, -1)

	return torch.mean(perplexity_per_sequence)

def PerplexityStable(logits, targets, loss_mask=None):
	"""
	calculates the avg perplexity
	this uses a more stable implementation
	by operating in log space and then doin exp

	logits.shape = ... x seq_len x vocab_size
	targets.shape = ... x seq_len
	loss_mask.shape = ... x seq_len

	"""

	# if loss mask is None, it is equivalent to ones
	if loss_mask is None:
		loss_mask = torch.ones(targets.shape, dtype=logits.dtype).to(targets.device)

	# we can now apply softmax on logits
	# although this is bloated, we only need value for targets[i][j]
	# but it is okay to have a neat and clean solution
	logits_log_softmax = torch.nn.functional.log_softmax(logits, -1)

	# now apply gather and only keep the target indices, shape = ... x seq_len
	logits_log_softmax = torch.gather(logits_log_softmax, -1, targets.unsqueeze(-1)).squeeze(-1)

	# now is the time to apply loss mask
	logits_log_softmax = logits_log_softmax * loss_mask

	# since we took log_softmax we need to negate it
	logits_log_softmax = -1. * logits_log_softmax

	# now, take a mean along sentence
	sequence_lengths = torch.sum(loss_mask, -1)
	perplexity_per_sequence = torch.exp(torch.sum(logits_log_softmax, -1) / sequence_lengths)

	return torch.mean(perplexity_per_sequence)
