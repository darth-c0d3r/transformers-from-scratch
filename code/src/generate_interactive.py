"""
use this script to run .generate function in an interactive mode
"""

from argparse import ArgumentParser
from pathlib import Path
import shutil
import json5
import time

import torch

from tokenizers.bpe_tokenizer import *
from transformer_lm.decoder_lm import Decoder_LM

def get_device(use_gpu=True):
	return torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")

@torch.no_grad
def generate_interactive(config):

	# step 1: get the device to be used
	device = get_device(use_gpu=True)
	print(f"> using device {device}\n")

	# step 2: load the tokenizer
	tokenizer = BPE_Tokenizer()
	tokenizer.load_tokenizer(config["tokenizer"])

	# step 3: load the model
	model = Decoder_LM.load_model(config["model_dir"], device=device).to(config["dtype"])
	assert model.vocab_size == tokenizer.vocab_size
	print(f"> model params = {sum([p.numel() for p in model.parameters()])}\n")
	model.eval()

	# step 4: do we want to apply stop token?
	# true for sft model, false for completion model
	apply_stop_token = False

	# do a while true loop
	while True:
		print(f"\n{'='*10}\n")

		# get the raw input prompt
		prompt = input("> prompt: ")

		# start the timer
		s = time.time()

		# apply chat template to the prompt
		prompt = tokenizer.apply_chat_template([prompt], apply_stop_token=apply_stop_token)
		print(f"> prompt formatted: {prompt[0]}")

		# get the input token ids
		input_ids, padding_mask = batch_encode(tokenizer, prompt, cache=True)

		# put the data on the device
		input_ids = torch.tensor(input_ids).to(device)
		padding_mask = torch.tensor(padding_mask).to(device)

		# do the generation step
		num_beams = 2
		num_samples = 2
		generated_data = model.generate(
				input_ids=input_ids,
				padding_mask=padding_mask,
				max_new_tokens=16,
				sampling=True,
				top_k=16,
				top_p=0.6,
				num_beams=num_beams,
				num_samples=num_samples,
			)

		# get the output ids
		output_ids = generated_data["output_ids"]

		# get the model confidence score
		sampled_probs = generated_data["sampled_probs"]

		total_confidence_list = model.get_total_prob_score(sampled_probs)[0].cpu().tolist()

		# do the decoding
		decoded_text_list = [tokenizer.decode(output_ids[0][idx].cpu().tolist()) for idx in range(num_samples)]

		# stop the timer
		e = time.time()

		for decoded_text, confidence in zip(decoded_text_list, total_confidence_list):
			print()
			print(f"> output: {decoded_text}\t|\t(score = {confidence:.3f})")
			# print(f"> probabilities: {sampled_probs[0].cpu().tolist()}")
		
		print(f"\n> generated {output_ids.numel()} tokens in {e-s:.2f}s ({output_ids.numel()/(e-s):.2f} tok/s)\n")



if __name__ == '__main__':

	# set up argument parser
	parser = ArgumentParser(description="sample generations from a trained model")
	parser.add_argument("--config_filename", type=str, required=True, help="config filename")
	args = parser.parse_args()

	# read the config
	assert Path(args.config_filename).is_file()
	
	with open(args.config_filename, "r") as f:
		config = json5.load(f)

	print("="*10 + "\n")
	print(json5.dumps(config, indent=4))
	print("\n" + "="*10 + "\n")

	# make sure model and tokenizer exist
	assert Path(config["model_dir"]).is_dir()
	assert Path(config["tokenizer"]).is_dir()

	# do a literal eval of dtype
	config["dtype"] = eval(config["dtype"])

	# call the generation routine
	generate_interactive(config)
