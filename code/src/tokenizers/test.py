"""
use this script to test all functions and features
of the byte pair encoding setup
"""

# ================================================================================ #
# ALL THE IMPORTS GO HERE

from bpe_tokenizer import *
import pandas as pd
import os

# ================================================================================ #

# ================================================================================ #
# HELPER FUNCTIONS

def get_tsv_dataset(filename, column, verbose=False):
	"""
	read the tsv file and return the column
	"""

	print(f"\n{'='*60}\n")
	print(f"loading data from {filename}")
	documents = pd.read_csv(filename, sep='\t')[column].tolist()
	print(f"loaded {len(documents)} documents\n")

	if verbose:
		print("\n".join(documents[:5]))
		print()
	
	print(f"{'='*60}\n")
	
	return documents

# ================================================================================ #

# ================================================================================ #
# THE MAIN ROUTINE

def run_tests():
	"""
	run the script to test bpe_tokenizer end to end
	"""

	# ================================================================================ #
	# LOAD DATASETS, NO SPECIAL TOKENS YET, FIRST TRAIN A TOKENIZER

	data_dir_1 = "../../../data/movie_subs/dune_part_1/"
	data_dir_2 = "../../../data/movie_subs/call_me_by_your_name/"

	dataset_1 = {}
	dataset_2 = {}

	dataset_1["train"] = get_tsv_dataset(os.path.join(data_dir_1, "train.tsv"), "Document")
	dataset_1["val"] = get_tsv_dataset(os.path.join(data_dir_1, "val.tsv"), "Document")

	dataset_2["train"] = get_tsv_dataset(os.path.join(data_dir_2, "train.tsv"), "Document")
	dataset_2["val"] = get_tsv_dataset(os.path.join(data_dir_2, "val.tsv"), "Document")
	
	# ================================================================================ #
	# TOKENIZER TRAINING PART

	tokenizer = BPE_Tokenizer()
	tokenizer.set_pretokenizer(PRETOKENIZER_PATTERNS["TIKTOKEN"])
	
	tokenizer.train_bpe(
		document_list = dataset_1["train"],
		num_merges = 8192,
		early_stop = True,
		verbose = True,
		use_multipliers = True,
		single_pass = True,
	)

	# STATS
	# using single pass --> 1782 vocab size in 3.47s
	# w/o using sing pass --> 1782 vocab size in 6.18s

	print(f"""\n{"="*60}\n""")

	# ================================================================================ #
	# INSERT SPECIAL TOKENS AND PLANT TREES FOR FAST (not really) ENCODE

	tokenizer.insert_special_token("[||UNK||]")
	tokenizer.insert_special_token("[||SEP||]")

	tokenizer.start_token_id = tokenizer.insert_special_token("[||START||]")
	tokenizer.pad_token_id = tokenizer.insert_special_token("[||PAD||]")

	total_trees = tokenizer.build_merge_forest()
	print(f"\nplanted {total_trees} trees in the merge forest\n")

	prefix_tree_size = tokenizer.build_prefix_tree()
	print(f"\ncreated a prefix tree of vocab size {prefix_tree_size}")

	print(f"""\n{"="*60}\n""")

	# ================================================================================ #
	# EVALUATE CORRECTNESS OF ALL ALGORITHMS
	
	data_joined_1 = " ".join(dataset_1["val"])
	data_joined_2 = " ".join(dataset_2["val"])

	data_joined_3 = f"""[||START||][||UNK|]]{data_joined_1}[||SEP||]{data_joined_2}[||UNK||][||PAD||]"""

	assert tokenizer.decode(tokenizer.encode(data_joined_3, algorithm="base")) == data_joined_3
	assert tokenizer.decode(tokenizer.encode(data_joined_3, algorithm="greedy")) == data_joined_3
	assert tokenizer.decode(tokenizer.encode(data_joined_3, algorithm="fast")) == data_joined_3
	assert tokenizer.decode(parallel_encode(tokenizer, data_joined_3)) == data_joined_3
	assert tokenizer.decode(batch_encode(tokenizer, [data_joined_3], algorithm="base")[0][0]) == data_joined_3
	assert tokenizer.decode(tokenizer.encode(data_joined_3, algorithm="fast", concurrent=True)) == data_joined_3

	print(f"""\n{"="*60}\n""")

	# ================================================================================ #
	# COMPRESSION RATIO TESTS
	
	if True:

		print("mixed data; base\t", tokenizer.get_compression_ratio(data_joined_3, algorithm="base"))
		print("mixed data; greedy\t", tokenizer.get_compression_ratio(data_joined_3, algorithm="greedy"))
		print("mixed data; fast\t", tokenizer.get_compression_ratio(data_joined_3, algorithm="fast"))
		print("mixed data; prefix\t", tokenizer.get_compression_ratio(data_joined_3, algorithm="prefix"))

		print()

		print("same dist data; base\t", tokenizer.get_compression_ratio(data_joined_1, algorithm="base"))
		print("same dist data; greedy\t", tokenizer.get_compression_ratio(data_joined_1, algorithm="greedy"))
		print("same dist data; fast\t", tokenizer.get_compression_ratio(data_joined_1, algorithm="fast"))
		print("same dist data; prefix\t", tokenizer.get_compression_ratio(data_joined_1, algorithm="prefix"))

	print(f"""\n{"="*60}\n""")

	# ================================================================================ #
	# THROUGHPUT TESTS

	if True:

		time_s = time.time()
		
		for _ in range(10):
			for doc in dataset_1["train"]:
				tokenizer.encode(
					doc,
					apply_pretokenizer=True,
					skip_special_tokens=False,
					concurrent=False,
					algorithm="base",
					cache=False,
				)

		time_e = time.time()
		print(f"base   | sequential | no cache   | tokenizer time = {time_e-time_s:.4f}s")

		# ================================================== #

		time_s = time.time()
		
		for _ in range(10):
			for doc in dataset_1["train"]:
				tokenizer.encode(
					doc,
					apply_pretokenizer=True,
					skip_special_tokens=False,
					concurrent=False,
					algorithm="fast",
					cache=False,
				)

		time_e = time.time()
		print(f"fast   | sequential | no cache   | tokenizer time = {time_e-time_s:.4f}s")

		# ================================================== #

		time_s = time.time()
		
		for _ in range(10):
			for doc in dataset_1["train"]:
				tokenizer.encode(
					doc,
					apply_pretokenizer=True,
					skip_special_tokens=False,
					concurrent=False,
					algorithm="greedy",
					cache=False,
				)

		time_e = time.time()
		print(f"greedy | sequential | no cache   | tokenizer time = {time_e-time_s:.4f}s")

		# ================================================== #

		time_s = time.time()
		
		for _ in range(10):
			for doc in dataset_1["train"]:
				tokenizer.encode(
					doc,
					apply_pretokenizer=True,
					skip_special_tokens=False,
					concurrent=False,
					algorithm="prefix",
					cache=False,
				)

		time_e = time.time()
		print(f"prefix | sequential | no cache   | tokenizer time = {time_e-time_s:.4f}s")

		# ================================================== #

		time_s = time.time()
		
		for _ in range(10):		
			for doc in dataset_1["train"]:
				tokenizer.encode(
					doc,
					apply_pretokenizer=True,
					skip_special_tokens=False,
					concurrent=False,
					algorithm="base",
					cache=True,
				)

		time_e = time.time()
		print(f"base   | sequential | with cache | tokenizer time = {time_e-time_s:.4f}s")

		tokenizer.clear_cache()

		# ================================================== #

		time_s = time.time()
		
		for _ in range(10):		
			for doc in dataset_1["train"]:
				tokenizer.encode(
					doc,
					apply_pretokenizer=True,
					skip_special_tokens=False,
					concurrent=False,
					algorithm="prefix",
					cache=True,
				)

		time_e = time.time()
		print(f"prefix   | sequential | with cache | tokenizer time = {time_e-time_s:.4f}s")

		tokenizer.clear_cache()

		# ================================================== #


		time_s = time.time()
		
		for _ in range(10):		
			batch_encode(
				tokenizer,
				dataset_1["train"],
				parallel=False,
				n_parts=None,
				apply_pretokenizer=True,
				skip_special_tokens=False,
				concurrent=False,
				algorithm="base",
				cache=False,
				padding="right",
			)

		time_e = time.time()
		print(f"base   | batched    | no cache   | tokenizer time = {time_e-time_s:.4f}s")

	print(f"""\n{"="*60}\n""")

	# ================================================================================ #
	# PADDING TESTS AND VIS

	if True:

		text_batch = tokenizer.apply_chat_template(dataset_1["val"][:5])

		print("===== INPUT TEXT =====")
		print("\n".join(text_batch))
		print()

		print("===== ENCODED TOKENS =====")
		for doc in text_batch:
			token_ids = tokenizer.encode(doc)
			# print(token_ids)
			print(tokenizer.convert_token_ids_to_bytes(token_ids))

		encoded_batch, padding_mask = batch_encode(tokenizer, text_batch, cache=True, padding="right")
		assert all([len(x) == len(encoded_batch[0]) for x in encoded_batch])
		assert all([len(x) == len(padding_mask[0]) for x in padding_mask])
		print()
		print(f"encoded_batch = {encoded_batch}\n")
		print(f"padding_mask = {padding_mask}")
		print()

	print(f"""\n{"="*60}\n""")

	# ================================================================================ #
	# TEST LOADING AND SAVING TOKENIZER PARTS

	if True:

		# this part checks if saving and loading tokenizer works
		tokenizer.save_tokenizer("./tokenizer_dune_part_1/")

		# load a fresh tokenizer and see if it same as tokenizer 1
		tokenizer_2 = BPE_Tokenizer()
		tokenizer_2.load_tokenizer("./tokenizer_dune_part_1/")

		ignore_keys = ["tokenization_algorithms", "merge_forest", "encoder_cache", "prefix_tree"]
		
		# assert vars(tokenizer).keys() == vars(tokenizer_2).keys()
		for k in vars(tokenizer).keys():
			if k in ignore_keys:
				continue
			assert vars(tokenizer)[k] == vars(tokenizer_2)[k], k

		print(f"loading and saving tokenizer is working fine")

	print(f"""\n{"="*60}\n""")


# ================================================================================ #

# ================================================================================ #
# THE MAIN ROUTINE TO BE TRIGGERED

if __name__ == '__main__':
	run_tests()

# ================================================================================ #