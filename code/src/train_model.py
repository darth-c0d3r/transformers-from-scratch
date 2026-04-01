"""
use this script to train a model on a tsv
"""

from argparse import ArgumentParser
from pathlib import Path
import shutil
import json5

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset

from dataset_loaders.dataset import CSV_Dataset
from tokenizers.bpe_tokenizer import *
from transformer_lm.decoder_lm import Decoder_LM
from modeling_utils.eval_functions import CrossEntropyLoss, Perplexity, PerplexityStable
from modeling_utils.optimizers import AdamW_Optimizer
from modeling_utils.optimizers import CosnineAnnealingWithWarmup_LRScheduler
from visualizers.training_visualizers import TrainingVisualizer

def get_device(use_gpu=True):
	return torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")

@torch.no_grad
def evaluate(model, tokenizer, val_dataloader, loss_fxn, eval_fxn):
	"""
	calculate the loss fxn and evaluation fxn for validation set
	"""

	device = model.device

	# turn on the evaluation flag
	model.eval()

	# instantiate loss and score as 0
	total_loss = 0.
	total_score = 0.

	# iterate over mini-batches of validation dataset
	for idx, batch in tqdm(enumerate(val_dataloader), desc="running evals", total=len(val_dataloader), disable=True):

		# apply chat template to all training documents
		documents = tokenizer.apply_chat_template(batch['Document'])

		# get input token ids and padding mask
		input_token_ids, padding_mask = batch_encode(tokenizer, documents, cache=True)
		input_token_ids = torch.tensor(input_token_ids).to(device)
		padding_mask = torch.tensor(padding_mask).to(device)

		# target token ids are one skipped and input token ids are one left
		target_token_ids = input_token_ids[:, 1:]
		input_token_ids = input_token_ids[:, :-1]
		padding_mask = padding_mask[:, :-1]

		# we should also pass token_positions for posn embeddings

		# now, do a fwd pass from the model
		_, logits = model(input_token_ids, padding_mask)

		# get the loss value and evaluation value
		loss = loss_fxn(logits, target_token_ids, loss_mask=padding_mask)
		score = eval_fxn(logits, target_token_ids, loss_mask=padding_mask)

		# add to total values
		total_loss += loss.cpu().item()
		total_score += score.cpu().item()

	# return mean loss and eval score
	return total_loss/len(val_dataloader), total_score/len(val_dataloader)

def train_model(config):

	# step 1: get the device to be used
	device = get_device(use_gpu=True)
	print(f"> using device {device}\n")

	# step 2: get train and validation datasets
	if config["dataset_type"] == "tsv":
		train_dataset = CSV_Dataset(config["train_dataset"], sep="\t")
		val_dataset = CSV_Dataset(config["val_dataset"], sep="\t")
	else:
		dataset = load_dataset(config["train_dataset"]).rename_column("text", "Document").shuffle(seed=42)
		train_dataset = dataset["train"].select(range(config["train_dataset_size"]))
		val_dataset = dataset["validation"].select(range(config["val_dataset_size"]))
	print(f"> train dataset = {len(train_dataset)} || val dataset = {len(val_dataset)}\n")

	# step 3: convert dataset to dataloaders
	train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
	val_dataloader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)

	# step 4: load the tokenizer
	tokenizer = BPE_Tokenizer()
	tokenizer.load_tokenizer(config["tokenizer"])

	# step 5: build or load the model
	if config["model_dir"] is not None: # case 1: load from model dir
		model = Decoder_LM.load_model(config["model_dir"], device=device).to(config["dtype"])
		assert model.vocab_size == tokenizer.vocab_size
		print(f"> model loaded from {config['model_dir']}\n")

	else: # case 2: build model from scratch
		model = Decoder_LM(
			vocab_size = tokenizer.vocab_size,
			** config["model_config"],
			device = device,
			dtype = config["dtype"],
		)
		print(f"> model initialized from scratch\n")

	print(f"> model params = {sum([p.numel() for p in model.parameters()])}\n")

	# step 6: do some basic napkin math book-keeping

	print("="*10 + "\n")

	print(f"> vocab size = {model.vocab_size} (log(v) = {torch.log(torch.tensor(model.vocab_size)).item():.4f})")
	print(f"> num epochs = {config['num_epochs']}")
	print(f"> batch size = {config['batch_size']}")
	print(f"> gradient accumulation steps = {config['gradient_accumulation_steps']}")
	print(f"> effective batch size = {config['batch_size'] * config['gradient_accumulation_steps']}")
	print(f"> trainining data size = {len(train_dataset)}")
	print(f"> num training data batches = {len(train_dataloader)}")
	print(f"> total steps in training = {len(train_dataloader) * config['num_epochs']}")
	total_weight_updates = int(len(train_dataloader) * config['num_epochs'] / config['gradient_accumulation_steps'])
	print(f"> total weight updates = {total_weight_updates}")

	print("\n" + "="*10 + "\n")

	# step 6: initialize the loss function and optimizer
	loss_fxn = CrossEntropyLoss
	eval_fxn = PerplexityStable

	lr_scheduler = CosnineAnnealingWithWarmup_LRScheduler
	num_warmup_steps = int(total_weight_updates * config["ratio_warmup_steps"])
	num_anneal_steps = int(total_weight_updates * config["ratio_anneal_steps"])
	lr_scheduler_params = {"num_warmup_steps": num_warmup_steps, "num_anneal_steps": num_anneal_steps, "lr_min": config["min_learning_rate"]}

	# optim = StochasticGradientDescent_Optimizer(model.parameters(), lr=1e-1)
	optim = AdamW_Optimizer(
		model.parameters(),
		lr = config["learning_rate"],
		weight_decay = config["weight_decay"],
		lr_scheduler = lr_scheduler,
		lr_scheduler_params = lr_scheduler_params,
		max_gradient_norm = config["max_gradient_norm"],
	)

	# step 7: finally start the training
	print(f"starting training...\n")

	# track this to perform periodic weight updates
	total_steps_so_far = 0

	# instantiate high value of best perplexity
	best_pplx = 1e8
	loss = torch.log(torch.tensor(model.vocab_size))

	# create the training visualizer
	visualizer = TrainingVisualizer(
		os.path.join(config["output_dir"], "progress"),
		total_steps = len(train_dataloader) * config["num_epochs"],
		num_epochs = config["num_epochs"],
		lr_scaling_factor = torch.log(torch.tensor(model.vocab_size)).item()/config["learning_rate"],
	)


	# iterate for n epochs and over each mini-batch
	for epoch in range(config["num_epochs"]):
		pbar = tqdm(
			enumerate(train_dataloader), 
			# desc=f"epoch {epoch+1}/{config['num_epochs']} (loss={loss.cpu().item():.4f})",
			desc=f"epoch {epoch+1}/{config['num_epochs']}",
			total=len(train_dataloader)
		)

		for idx, batch in pbar:

			# ensure the model has train flag enabled
			model.train()

			# increment total steps so far at the start
			total_steps_so_far += 1

			# apply chat template to all training documents
			documents = tokenizer.apply_chat_template(batch['Document'])

			# get input token ids and padding mask
			input_token_ids, padding_mask = batch_encode(tokenizer, documents, cache=True)
			input_token_ids = torch.tensor(input_token_ids).to(device)
			padding_mask = torch.tensor(padding_mask).to(device)

			# target token ids are one skipped and input token ids are one left
			target_token_ids = input_token_ids[:, 1:]
			input_token_ids = input_token_ids[:, :-1]
			padding_mask = padding_mask[:, :-1]

			# we should also pass token_positions for posn embeddings

			# now, do a fwd pass from the model
			_, logits = model(input_token_ids, padding_mask)

			# get the loss value
			loss = loss_fxn(logits, target_token_ids, loss_mask=padding_mask)

			# do a backward pass
			loss.backward()

			# make it reflect in progress bar
			pbar.set_postfix(loss=f"{loss.cpu().item():.4f}")

			# see if we're ready for weight update
			if total_steps_so_far % config["gradient_accumulation_steps"] == 0:

				# update the weights and reset the gradients
				optim.step()
				optim.zero_grad()

			# see if we're ready to report validation loss
			if total_steps_so_far % config["eval_frequency"] == 0:

				# get the validation loss and perplexity
				val_loss, val_pplx = evaluate(model, tokenizer, val_dataloader, loss_fxn, eval_fxn)

				# see if this is the best perplexity score
				if val_pplx < best_pplx:

					# save the best model so far
					best_pplx = val_pplx
					model.save_model(os.path.join(config["output_dir"], "best_pplx_ckpt"))

				# report values to visualizer
				visualizer.update({
						"epoch": epoch,
						"step": total_steps_so_far,
						"train_loss": loss.cpu().item(),
						"val_loss": val_loss,
						"val_pplx": val_pplx,
						"lr": optim.current_lr,
					})

	# do an evaluation at the end of training
	# get the validation loss and perplexity
	val_loss, val_pplx = evaluate(model, tokenizer, val_dataloader, loss_fxn, eval_fxn)

	# see if this is the best perplexity score
	if val_pplx < best_pplx:

		# save the best model so far
		best_pplx = val_pplx
		model.save_model(os.path.join(config["output_dir"], "best_pplx_ckpt"))

	# save the final checkpoint
	model.save_model(os.path.join(config["output_dir"], "final_ckpt"))

	# final signal to visualizer
	visualizer.update({
		"epoch": config["num_epochs"],
		"step": total_steps_so_far,
		"train_loss": loss.cpu().item(),
		"val_loss": val_loss,
		"val_pplx": val_pplx,
		"lr": optim.current_lr,
	})

	print(f"\n{'='*10} FIN {'='*10}\n")


if __name__ == '__main__':

	# set up argument parser
	parser = ArgumentParser(description="train and save model on a corpus")
	parser.add_argument("--config_filename", type=str, required=True, help="config filename")
	args = parser.parse_args()

	# read the config
	assert Path(args.config_filename).is_file()
	
	with open(args.config_filename, "r") as f:
		config = json5.load(f)

	print("="*10 + "\n")
	print(json5.dumps(config, indent=4))
	print("\n" + "="*10 + "\n")

	# make sure corpus and tokenizer
	if config["dataset_type"] != "hgfc":
		assert Path(config["train_dataset"]).is_file()
		assert Path(config["val_dataset"]).is_file()
	assert Path(config["tokenizer"]).is_dir()

	# create the output dir
	Path(config["output_dir"]).mkdir(exist_ok=True, parents=True)

	# copy the config file to output dir
	shutil.copy(args.config_filename, config["output_dir"])

	# do a literal eval of dtype
	config["dtype"] = eval(config["dtype"])

	# call the training routine
	train_model(config)
