import sys
sys.path.insert(1, "../")

import random
from tqdm import tqdm

import torch

from transformer_lm.decoder_lm import Decoder_LM
from modeling_utils.eval_functions import CrossEntropyLoss, Perplexity, PerplexityStable
from optimizers import StochasticGradientDescent_Optimizer, AdamW_Optimizer
from optimizers import LinearDecayWithWarmup_LRScheduler, CosnineAnnealingWithWarmup_LRScheduler

batch_size = 8
seq_len = 64
vocab_size = 64

logits_randn = torch.randn((batch_size, seq_len, vocab_size))
logits_uniform = torch.distributions.uniform.Uniform(-1, 1).sample([batch_size, seq_len, vocab_size]) 
logits_equal = torch.zeros((batch_size, seq_len, vocab_size), dtype=torch.float32) + 4.2
indices = torch.randint(0, vocab_size, (batch_size, seq_len))

print(CrossEntropyLoss(logits_randn, indices), torch.log(torch.tensor(vocab_size)))
print(CrossEntropyLoss(logits_uniform, indices), torch.log(torch.tensor(vocab_size)))
print(CrossEntropyLoss(logits_equal, indices), torch.log(torch.tensor(vocab_size)))
print()

print(Perplexity(logits_randn, indices))
print(Perplexity(logits_uniform, indices))
print(Perplexity(logits_equal, indices))
print()

print(PerplexityStable(logits_randn, indices))
print(PerplexityStable(logits_uniform, indices))
print(PerplexityStable(logits_equal, indices))
print()

# train a dummy model to always output the same token

model = Decoder_LM(
	vocab_size = vocab_size,
	model_dim = 128,
	hidden_dim = 256,
	head_dim = 64,
	num_heads = 8,
	num_layers = 4,
	max_seq_len = 1024,
	device = None,
	dtype = None,
	init = True,
)

num_params = sum([p.numel() for p in model.parameters()])
print(f"\nmodel params = {num_params}\n")

num_epochs = 100
min_seq_len = 16
max_seq_len = 64
target_token_id = 13

# lr_scheduler = LinearDecayWithWarmup_LRScheduler
# lr_scheduler_params = {"num_warmup_steps": 20, "total_steps": num_epochs}

lr_scheduler = CosnineAnnealingWithWarmup_LRScheduler
lr_scheduler_params = {"num_warmup_steps": 20, "num_anneal_steps": 90, "lr_min": 1e-7}

# optim = StochasticGradientDescent_Optimizer(model.parameters(), lr=1e-1)
optim = AdamW_Optimizer(
	model.parameters(),
	lr=1e-3,
	weight_decay=0.0,
	lr_scheduler=lr_scheduler,
	lr_scheduler_params=lr_scheduler_params,
	max_gradient_norm=10.,
)

for _ in range(num_epochs):

	optim.zero_grad()
	seq_len = random.choice(range(min_seq_len, max_seq_len))

	token_ids = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.int)
	targets = torch.zeros((batch_size, seq_len), dtype=token_ids.dtype) + target_token_id

	_, logits = model(token_ids)

	loss = CrossEntropyLoss(logits, targets)
	print(f"loss = {loss.cpu().item():.4f}")
	loss.backward()

	optim.step()

	print(f"lr = {optim.current_lr}\n")

seq_len = random.choice(range(min_seq_len, max_seq_len))
token_ids = torch.randint(0, vocab_size, (1, seq_len), dtype=torch.int)
print(model.generate(token_ids, max_new_tokens=8)[1])
