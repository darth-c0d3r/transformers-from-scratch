"""
use this script to train a tokenizer on a corpus
a corpus is a raw txt file used in its entirety to train
"""

from argparse import ArgumentParser
from pathlib import Path
import json5
import pandas as pd

from datasets import load_dataset

from tokenizers.bpe_tokenizer import *

def train_tokenizer(config):
	"""
	main routine to train tokenizer
	"""

	# save the config to output path
	with open(os.path.join(config["output_dir"], "config.json"), "w+") as f:
		f.write(json5.dumps(config, indent=4))

	# get the raw init tokenizer
	tokenizer = BPE_Tokenizer()

	# set its pre tokenizer
	if config["pretokenizer"] is not None:
		tokenizer.set_pretokenizer(PRETOKENIZER_PATTERNS[config["pretokenizer"]])
		print(f"pretokenizer set to {PRETOKENIZER_PATTERNS[config['pretokenizer']]}\n")

	# get the corpora
	if config["dataset_type"] == "tsv":
		df_train = pd.read_csv(config["train_dataset"], sep="\t")
		train_corpus = df_train["Document"].tolist()[:config["train_dataset_size"]]

		df_val = pd.read_csv(config["val_dataset"], sep="\t")
		val_corpus = df_val["Document"].tolist()[:config["val_dataset_size"]]

	else:

		dataset = load_dataset(config["train_dataset"]).shuffle(seed=42)

		train_corpus = dataset['train'][:config["train_dataset_size"]]['text']
		val_corpus = dataset['validation'][:config["val_dataset_size"]]['text']

	# run the training
	tokenizer.train_bpe(
		train_corpus,
		num_merges=config["num_merges"],
		early_stop=False,
		verbose=True,
		use_multipliers=True,
		single_pass=True
	)

	# add pad token id
	if config["pad_token"] is not None:
		pad_token_id = tokenizer.insert_special_token(config["pad_token"])
		tokenizer.pad_token_id = pad_token_id
		print(f"inserted pad token {config['pad_token']} @ {pad_token_id}\n")

	# add start token id
	if config["start_token"] is not None:
		start_token_id = tokenizer.insert_special_token(config["start_token"])
		tokenizer.start_token_id = start_token_id
		print(f"inserted start token {config['start_token']} @ {start_token_id}\n")

	# add all other speacial tokens
	if config["other_special_tokens"] is not None:
		for spcl_tok in config["other_special_tokens"]:
			tok_id = tokenizer.insert_special_token(spcl_tok)
			print(f"inserted special token {spcl_tok} @ {tok_id}\n")

	# print vocab size
	print(f"final vocab size = {tokenizer.vocab_size}\n")

	# print compression ratios
	print("train compression ratio\t", tokenizer.get_compression_ratio(config["pad_token"].join(train_corpus), algorithm="base"))
	print("val compression ratio\t", tokenizer.get_compression_ratio(config["pad_token"].join(val_corpus), algorithm="base"))
	print()

	# get sample tokenization
	sample_input = tokenizer.apply_chat_template([val_corpus[0]])[0]
	encoded_batch, _ = batch_encode(tokenizer, [sample_input], cache=True, padding="left")
	print("=" * 10)
	print("SAMPLE ENCODING\n")
	print(sample_input)
	print()
	print(encoded_batch)
	print()
	print(tokenizer.encode(sample_input))
	print()
	print(tokenizer.convert_token_ids_to_bytes(encoded_batch[0]))
	print("\n" + "=" * 10 + "\n")

	assert tokenizer.decode(tokenizer.encode(sample_input)) == sample_input

	# save the tokenizer
	tokenizer.save_tokenizer(config["output_dir"])
	print(f"tokenizer saved to {config['output_dir']}")

if __name__ == '__main__':

	# set up argument parser
	parser = ArgumentParser(description="train and save tokenizer on a corpus")
	parser.add_argument("--config_filename", type=str, required=True, help="config filename")
	args = parser.parse_args()

	# read the config
	assert Path(args.config_filename).is_file()
	
	with open(args.config_filename, "r") as f:
		config = json5.load(f)

	print("="*10 + "\n")
	print(json5.dumps(config, indent=4))
	print("\n" + "="*10 + "\n")

	# make sure corpus exists
	if config["dataset_type"] != "hgfc":
		assert Path(config["train_dataset"]).is_file()
		assert Path(config["val_dataset"]).is_file()

	# create the output dir
	Path(config["output_dir"]).mkdir(exist_ok=True, parents=True)

	# call the training routine
	train_tokenizer(config)
