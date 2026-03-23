from bpe_tokenizer import *

if __name__ == '__main__':
	"""
	this section contains all the testing commands
	"""

	# load a base string from file
	with open("../../../data/tokenization/Virat_Kohli.txt", "r") as f:
		data_a = "".join(f.readlines())

	with open("../../../data/tokenization/Taylor_Swift.txt", "r") as f:
		data_b = "".join(f.readlines())

	tokenizer = BPE_Tokenizer()
	tokenizer.set_pretokenizer(PRETOKENIZER_PATTERNS["TIKTOKEN"])
	
	tokenizer.train_bpe([data_a] * 16, 2048, early_stop=False, verbose=True, use_multipliers=True, single_pass=True)

	tokenizer.insert_special_token("[UNK]")
	tokenizer.insert_special_token("<|eot|>")
	pad_token_id = tokenizer.insert_special_token("[PAD]")
	tokenizer.pad_token_id = pad_token_id

	total_trees = tokenizer.build_merge_forest()
	print(f"planted {total_trees} trees in the merge forest\n")
	
	data_c = f"[UNK][UNK]{data_a}[UNK]<|eot|>[UNK]{data_b}<|eot|><|eot|>"

	assert tokenizer.decode(tokenizer.encode(data_c, algorithm="base")) == data_c
	assert tokenizer.decode(tokenizer.encode(data_c, algorithm="greedy")) == data_c
	assert tokenizer.decode(tokenizer.encode(data_c, algorithm="fast")) == data_c
	assert tokenizer.decode(parallel_encode(tokenizer, data_c)) == data_c
	# assert tokenizer.decode(tokenizer.encode(data_c, algorithm="fast", concurrent=True)) == data_c

	print("base\t", tokenizer.get_compression_ratio(data_c, algorithm="base"))
	print("greedy\t", tokenizer.get_compression_ratio(data_c, algorithm="greedy"))
	print("fast\t", tokenizer.get_compression_ratio(data_c, algorithm="fast"))

	print()

	# time_s = time.time()
	# for _ in tqdm(range(100)):
	# 	tokenizer.encode(data_c, algorithm="base", concurrent=True)
	# time_e = time.time()
	# print(f"base concurrent tokenizer time = {time_e-time_s:.4f}s")

	time_s = time.time()
	for _ in tqdm(range(20)):
		tokenizer.encode(data_c, algorithm="base")
	time_e = time.time()
	print(f"base tokenizer time = {time_e-time_s:.4f}s")

	# time_s = time.time()
	# for _ in tqdm(range(20)):
	# 	tokenizer.encode(data_c, algorithm="greedy")
	# time_e = time.time()
	# print(f"greedy tokenizer time = {time_e-time_s:.4f}s")

	# time_s = time.time()
	# for _ in tqdm(range(20)):
	# 	tokenizer.encode(data_c, algorithm="fast")
	# time_e = time.time()
	# print(f"fast tokenizer time = {time_e-time_s:.4f}s")

	time_s = time.time()
	for _ in tqdm(range(20)):
		parallel_encode(tokenizer, data_c)
	time_e = time.time()
	print(f"mp tokenizer time = {time_e-time_s:.4f}s")

	time_s = time.time()
	for _ in tqdm(range(20)):
		tokenizer.encode(data_c, cache=True)
		tokenizer.clear_cache()
	time_e = time.time()
	print(f"cached base w clear tokenizer time = {time_e-time_s:.4f}s")

	time_s = time.time()
	for _ in tqdm(range(20)):
		parallel_encode(tokenizer, data_c, cache=True)
		tokenizer.clear_cache()
	time_e = time.time()
	print(f"cached mp w clear tokenizer time = {time_e-time_s:.4f}s")

	time_s = time.time()
	for _ in tqdm(range(20)):
		tokenizer.encode(data_c, cache=True)
		# tokenizer.clear_cache()
	time_e = time.time()
	print(f"cached base w/o clear tokenizer time = {time_e-time_s:.4f}s")

	time_s = time.time()
	for _ in tqdm(range(20)):
		parallel_encode(tokenizer, data_c, cache=True)
		# tokenizer.clear_cache()
	time_e = time.time()
	print(f"cached mp w/o clear tokenizer time = {time_e-time_s:.4f}s")

	text_batch = ["A man flew over some part, idk", "Virat Kohli is a legend, maybe tellllllll.", "Taylor Swift"]
	encoded_batch, padding_mask = batch_encode(tokenizer, text_batch, cache=True, padding="right")
	assert all([len(x) == len(encoded_batch[0]) for x in encoded_batch])
	assert all([len(x) == len(padding_mask[0]) for x in padding_mask])
	print()
	print(f"encoded_batch = {encoded_batch}")
	print(f"padding_mask = {padding_mask}")
	print()

	# this part checks if saving and loading tokenizer works
	tokenizer.save_tokenizer("./tokenizer_virat_kohli/")

	# load a fresh tokenizer and see if it same as tokenizer 1
	tokenizer_2 = BPE_Tokenizer()
	tokenizer_2.load_tokenizer("./tokenizer_virat_kohli/")

	ignore_keys = ["tokenization_algorithms", "merge_forest", "encoder_cache"]
	# assert vars(tokenizer).keys() == vars(tokenizer_2).keys()
	for k in vars(tokenizer).keys():
		if k in ignore_keys:
			continue
		assert vars(tokenizer)[k] == vars(tokenizer_2)[k], k
