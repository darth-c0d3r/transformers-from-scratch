"""
this script contains byte-pair encoding based tokenizers
with multiple pre-tokenizers built in (regex based)
and with an added support to add own pre-tokenizers (regex based)
there is also a provision to add special tokens to tokenizer
"""

import os
import sys
import shutil
from pathlib import Path
import functools
import multiprocessing as mp
from collections import Counter

import time
import json
from tqdm import tqdm
import ast

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import regex as re # this is a slightly faster version of default re library

from .utils import PrefixTree, DisjointSet

PRETOKENIZER_PATTERNS = {
	"GPT2": r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
	"TIKTOKEN": r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s""",
}

def batch_encode(
	tokenizer,
	text_list,
	parallel=False,
	n_parts=None,
	apply_pretokenizer=True,
	skip_special_tokens=False,
	concurrent=False,
	algorithm="base",
	cache=False,
	padding="left",
):
	"""
	run batch encoding with padding on a list of text documents
	also return padding mask with it
	"""

	# ensure padding direction is correct
	assert padding in ["left", "right", None], padding

	# give a little warning
	if padding is None:
		print(f"> warning: padding is set to None; batch_encode will return unpadded sequences")

	# keep a track of all lengths for padding
	length_list = []
	encoded_batch = []
	padding_batch = []

	for text in text_list:

		if parallel is True:
			encoded_token_ids = parallel_encode(
				tokenizer,
				text,
				n_parts = n_parts,
				apply_pretokenizer = apply_pretokenizer,
				skip_special_tokens = skip_special_tokens,
				concurrent = concurrent,
				algorithm = algorithm,
				cache = cache,
			)
		else:
			encoded_token_ids = _encode_chunk(
				text,
				tokenizer,
				apply_pretokenizer = apply_pretokenizer,
				skip_special_tokens = skip_special_tokens,
				concurrent = concurrent,
				algorithm = algorithm,
				cache = cache,
			)

		length_list.append(len(encoded_token_ids))
		encoded_batch.append(encoded_token_ids)
		if padding is not None:
			padding_batch.append([1]*len(encoded_token_ids))

	# take care of padding here
	pad_token_id = tokenizer.pad_token_id
	if padding is not None:
		if pad_token_id is None:
			pad_token_id = 0
			print(f"> warning: tokenizer.pad_token_id not set. Using default = 0.")

	# get the max length for padding
	max_length = max(length_list)

	# iterate over all elements of encoded_batch
	if padding is not None:
		for i in range(len(encoded_batch)):

			padding_arm = [pad_token_id] * (max_length - len(encoded_batch[i]))
			masking_arm = [0] * (max_length - len(encoded_batch[i]))

			if padding == "left":
				encoded_batch[i] = padding_arm + encoded_batch[i]
				padding_batch[i] = masking_arm + padding_batch[i]

			else:
				encoded_batch[i] = encoded_batch[i] + padding_arm
				padding_batch[i] = padding_batch[i] + masking_arm

	# return both encoded batch and padding mask
	return encoded_batch, padding_batch if padding is not None else None

def parallel_encode(
	tokenizer,
	text,
	n_parts=None,
	apply_pretokenizer=True,
	skip_special_tokens=False,
	concurrent=False,
	algorithm="base",
	cache=False,
):

	"""
	Split text into n_parts and encode in parallel using multiple processes.
	The tokenizer.encode() function is used as-is.
	ideally, would want this to be a member function,
	but apparently this is the neatest way
	"""

	if n_parts is None:
		n_parts = mp.cpu_count()

	# Split text roughly evenly into n_parts
	text_parts = split_text_evenly(text, n_parts)

	# create a partial function
	encode_partial = functools.partial(
		_encode_chunk,
		tokenizer=tokenizer,
		apply_pretokenizer=apply_pretokenizer,
		skip_special_tokens=skip_special_tokens,
		concurrent=concurrent,
		algorithm=algorithm,
		cache=cache,
	)

	with mp.Pool(processes=n_parts) as pool:
		results = pool.map(encode_partial, text_parts)

	# Flatten results (assuming each encode returns a list of token ids)
	return [tok for chunk in results for tok in chunk]

def _encode_chunk(
	text_part,
	tokenizer,
	apply_pretokenizer=True,
	skip_special_tokens=False,
	concurrent=False,
	algorithm="base",
	cache=False,
):
	return tokenizer.encode(
		text_part,
		apply_pretokenizer=apply_pretokenizer,
		skip_special_tokens=skip_special_tokens,
		concurrent=concurrent,
		algorithm=algorithm,
		cache=cache,
	)

def split_text_evenly(text, n_parts):
	"""Split text roughly evenly on whitespace (to avoid mid-word breaks)."""
	
	words = text.split(' ')
	n_parts = min(n_parts, len(words))
	chunk_size = len(words) // n_parts
	parts = []
	
	for i in range(n_parts):
	
		start = i * chunk_size
		end = (i + 1) * chunk_size if i < n_parts - 1 else len(words)
		parts.append(' '.join(words[start:end]))

		if i != 0:
			parts[-1] = f" {parts[-1]}"
	
	return parts


class BPE_Tokenizer():
	"""
	main wrapper class for byte-pair encoding based tokenizers
	
	default behaviour for the BPE tokenizer is with a vocab size 256
	and encode directly at the lowest byte level with no merges

	"""

	def __init__(self, encoding="utf-8"):
		"""
		initialize the default BPE tokenizer
		all the custom building happens later
		"""

		# set the encoding to be used throughout
		# default encoding is utf-8 for now
		self.encoding = encoding

		# multiple core tokenization algorithms can be used
		# this dict is a mapping of various encode cores
		self.tokenization_algorithms = {
			"base": self.encode_core_base,
			"greedy": self.encode_core_greedy_2_pointer,
			"fast": self.encode_core_fast,
			"prefix": self.encode_core_greedy_prefix_tree,
		}

		# the default vocabulary contains index -> byte([index])
		# for index = [0, 255]
		# this default mapping would be sufficient
		# to represent any byte stream in tokens
		# since any byte stream would have bytes in that range
		self.vocab = {idx: bytes([idx]) for idx in range(0, 256)}
		self.vocab_bytes2idx = {v: k for (k, v) in self.vocab.items()}
		self.vocab_size = len(self.vocab)

		# the initial set of merges would be an empty list
		# simple question - why a dict and not a list
		# since we'd want the order of merges to be stored
		# order is inherently stored because merges happen in increasing index
		self.merges = {}

		# the initial pre-tokenizer pattern would be None
		self.pretokenizer_pattern = None

		# the initial set of special tokens would be an empty mapping
		# special tokens is a map from int index to byte-stream
		self.special_tokens = {} # idx2bytes
		self.special_tokens_str2idx = {} # str2idx

		# some special tokens might be needed separately
		self.start_token_id = None
		self.pad_token_id = None
		# self.eos_token_id = None # can be taken care of by careful padding

		# keep a cache for cached encoding
		# the cache saved encodings at chunk level (after pre-tokenizer)
		self.encoder_cache = {algo: {} for algo in self.tokenization_algorithms}

		return

	def encode_core_base(self, document):
		"""
		the core encode function that operates on byte level
		created a separate function to conveniently switch out cores
		"""

		# convert the document to a byte-stream and token ids
		encoded_token_ids = [b for b in document.encode(self.encoding)]

		# iterate while it is possible to find merges
		while len(encoded_token_ids) > 1:

			# get all byte pair frequencies
			byte_pair_frequencies = self.get_pair_frequencies([encoded_token_ids])
			
			# ignore the frequencies and get the merge tokens for all pairs
			byte_pair_merges = {pair: self.merges[pair] for pair in byte_pair_frequencies if pair in self.merges}

			# break if no more merges are possible
			if len(byte_pair_merges) == 0:
				break

			# get the pair with the lowest merge_ids
			merge_idx, pair = min([(merge_idx, pair) for (pair, merge_idx) in byte_pair_merges.items()])

			# now apply the merge
			encoded_token_ids = [encoded_token_ids]
			self.merge_pair(encoded_token_ids, pair, merge_idx)
			encoded_token_ids = encoded_token_ids[0]

			# print(f"ENC-BASE {pair} --> {merge_idx}")

		return encoded_token_ids

	def encode_core_fast(self, document):
		"""
		mathematically, should be exact same as base
		instead of only picking the lowest index to merge
		merge all independent pairs

		independent pair means that they are not in the same
		merge tree as the current merge

		i think for this, we need to build a list of trees
		and the set of indices in each tree
		this can be built from merges
		"""

		# for fast tokenizer, we need to build the merge forest
		assert hasattr(self, "merge_forest")

		# convert the document to a byte-stream and token ids
		encoded_token_ids = [b for b in document.encode(self.encoding)]

		# iterate while it is possible to find merges
		while len(encoded_token_ids) > 1:

			# get all byte pair frequencies
			byte_pair_frequencies = self.get_pair_frequencies([encoded_token_ids])
			
			# ignore the frequencies and get the merge tokens for all pairs
			byte_pair_merges = {pair: self.merges[pair] for pair in byte_pair_frequencies if pair in self.merges}

			# break if no more merges are possible
			if len(byte_pair_merges) == 0:
				break

			# keep track of distinct trees
			covered_trees = set()

			for (merge_idx, pair) in sorted([(v, k) for (k, v) in byte_pair_merges.items()]):

				# mathematically, pair[0], pair[1], merge_idx will be in same tree
				# and it can't be None, every merge got covered in a tree
				# only unmerged bytes were left as seedlings
				# they won't be in merges anyway

				if self.merge_forest[merge_idx] not in covered_trees:
					covered_trees.add(self.merge_forest[merge_idx])

					# now apply the merge
					encoded_token_ids = [encoded_token_ids]
					self.merge_pair(encoded_token_ids, pair, merge_idx)
					encoded_token_ids = encoded_token_ids[0]

					# print(f"{pair} --> {merge_idx}")

		return encoded_token_ids

	def encode_core_greedy_2_pointer(self, document):
		"""
		this tokenizer iteratively looks for the longest prefix match at byte level
		we don't need merges for this, only vocab is enough
		this can definitely be made more efficient with prefix-tree search
		"""

		# convert the document to a byte-stream and token ids
		input_token_ids = [b for b in document.encode(self.encoding)]

		# get the max possible token size
		LONGEST_TOKEN_SIZE = max([len(k) for k in self.vocab_bytes2idx])

		# start being greedy here
		curr_idx = 0
		encoded_token_ids = []

		# go on while we're in the legal access zone
		while curr_idx < len(input_token_ids):

			max_jump = min(LONGEST_TOKEN_SIZE, len(input_token_ids) - curr_idx)
			for jump_size in range(max_jump, 0, -1):

				candidate_token = bytes(input_token_ids[curr_idx:curr_idx+jump_size])

				# we know for sure that bpe will always return at base case of 1 byte
				# hence not accounting it separately here 
				if candidate_token in self.vocab_bytes2idx:
					encoded_token_ids.append(self.vocab_bytes2idx[candidate_token])
					curr_idx += jump_size
					break

		return encoded_token_ids

	def encode_core_greedy_prefix_tree(self, document):
		"""
		this is a more optimized implementation of greedy 2 pointer
		uses a prefix tree to match the longest possible prefix
		"""

		# for fast tokenizer, we need to build the prefix tree
		assert hasattr(self, "prefix_tree")

		# convert the document to a byte-stream and token ids
		input_token_ids = [b for b in document.encode(self.encoding)]

		# start being greedy here
		curr_idx = 0
		encoded_token_ids = []

		# go on while we're in the legal access zone
		while curr_idx < len(input_token_ids):

			# get the longest possible jump size with prefix match
			jump_size = len(self.prefix_tree.getLongestPrefixMatch(input_token_ids, curr_idx))

			# get the candidate token bytes
			candidate_token = bytes(input_token_ids[curr_idx:curr_idx+jump_size])

			# advance the curr token accordingly
			curr_idx += jump_size

			# we know for sure that bpe will always return at base case of 1 byte
			# hence not accounting it separately here 
			if candidate_token in self.vocab_bytes2idx:
				encoded_token_ids.append(self.vocab_bytes2idx[candidate_token])

		return encoded_token_ids

	def encode(self, document, 
		apply_pretokenizer=True,
		skip_special_tokens=False,
		concurrent=False,
		algorithm="base",
		cache=False,
	):
		"""
		document is a string to be tokenized
		convert the input document to a list of token_ids
		skip_special_tokens is to handle special tokens separately
		steps:
			tokenize the special tokens
			apply pre-tokenizer if needed
			on each element, apply bpe merge algorithm
		"""

		# ensure that the algorithm is valid
		assert algorithm in self.tokenization_algorithms

		# handle empty string upfront
		if not isinstance(document, str) or len(document) == 0:
			return []

		# handle special tokens here
		if not skip_special_tokens and len(self.special_tokens) > 0:

			regex_pattern = f"({'|'.join(map(re.escape, self.special_tokens_str2idx.keys()))})"
			document_chunks = re.split(regex_pattern, document) # we get a list here
			document_chunks = [
				chunk if chunk not in self.special_tokens_str2idx
				else self.special_tokens_str2idx[chunk]
				for chunk in document_chunks
			]

			encoded_token_ids = []

			# iterate over all chunks
			for chunk in document_chunks:

				# special token case
				if isinstance(chunk, int):
					encoded_token_ids.append(chunk)
					continue

				encoded_token_ids.extend(
						self.encode(
						document=chunk,
						apply_pretokenizer=apply_pretokenizer,
						skip_special_tokens=True,
						concurrent=concurrent,
						algorithm=algorithm,
						cache=cache,
					)
				)

			return encoded_token_ids

		# handle the case where you need to apply the pretokenizer
		if apply_pretokenizer is True:

			# converts str --> list[str]
			document_chunks, _ = self.apply_pretokenizer(document)

			# concurrent or multi-threraded tokenization will only
			# give gains if the tokenizer itself has an instance on each thread
			# otherwise we won't see gains but rather see regressions 
			if concurrent: # multi-threaded processing

				with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:

					encode_partial = functools.partial(
						self.encode,
						apply_pretokenizer=False,
						skip_special_tokens=True,
						concurrent=False,
						algorithm=algorithm,
						cache=cache,
					)

					encoded_token_ids = executor.map(encode_partial, document_chunks)

				# encoded token ids here would be a list of lists, flatten it
				encoded_token_ids = [item for sublist in encoded_token_ids for item in sublist]

			else: # sequential processing
				encoded_token_ids = []

				# iterate over all chunks
				for chunk in tqdm(document_chunks, desc="encoding", disable=True):
					encoded_token_ids.extend(
						self.encode(
							document=chunk,
							apply_pretokenizer=False, 
							skip_special_tokens=True,
							concurrent=False,
							algorithm=algorithm,
							cache=cache,
						)
					)

			return encoded_token_ids

		# handle the core, byte level tokenizer here

		# call a separate encode_core function
		# so that if we are optimizing, then we can call a separate core function
		# call the corresponding required encode function

		# if cache is enabled, look up in cache and return
		if cache and document in self.encoder_cache[algorithm]:
			return self.encoder_cache[algorithm][document]

		# else, compute fresh
		encoded_token_ids = self.tokenization_algorithms[algorithm](document)

		# and add to cache
		self.encoder_cache[algorithm][document] = encoded_token_ids

		return encoded_token_ids

	def decode(self, token_ids):
		"""
		token_ids is a list of integers
		convert the token ids to a final string document
		"""

		# get the decoded bytes
		decoded_bytes = [
			self.vocab[tok_id] if tok_id < len(self.vocab) else self.special_tokens[tok_id]
			for tok_id in token_ids
		]

		# get the final decoded document
		# sometimes the llm model might return an invalid sequence of bytes
		# so errors are replaced by a `?`
		decoded_document = b"".join(decoded_bytes).decode(self.encoding, errors='replace')

		# return the decoded document
		return decoded_document
		
	def insert_special_token(self, token):
		"""
		special token is always a string
		convert the special token to a byte stream
		add the byte stream to the special tokens dict
		special tokens are always added after finalizing vocab
		"""

		self.special_tokens[self.vocab_size] = token.encode(self.encoding)
		self.special_tokens_str2idx[token] = self.vocab_size
		self.vocab_size += 1

		print(f"added special token {token} to vocab @ index {self.vocab_size-1}.")

		return self.vocab_size - 1

	def get_compression_ratio(self, document, algorithm="base"):
		"""
		return the compression ratio for a given str document
		"""

		num_chars = len(document)
		num_bytes = len(document.encode(self.encoding))
		num_tokens = len(self.encode(document, algorithm=algorithm))

		return {
			"chars per token": num_chars / num_tokens,
			"bytes per token": num_bytes / num_tokens,
		}

	def set_pretokenizer(self, regex_pattern):
		"""
		set the pre-tokenizer logic via regex here
		technically, this can be any arbitrary function
		but providing support via regex here
		"""

		self.pretokenizer_pattern = re.compile(regex_pattern)
		print("pretokenizer pattern set successfully.")
		return

	def apply_pretokenizer(self, document, return_counts=False):
		"""
		document is a string here
		returns a list of sub-components

		return counts will also obviously shuffle up the tokens
		"""

		# call this function only if pretokenizer is not None
		# assert self.pretokenizer_pattern is not None

		# if pretokenizer is None, simple convert to list and return
		if self.pretokenizer_pattern is None:
			print(f"> warning: pretokenizer not set. returning original document as is.")
			return ([document], [1] if return_counts else None)

		# if we don't have to return counts
		if return_counts is False:
			return (re.findall(self.pretokenizer_pattern, document), None)

		# if we have to return counts, assume its a big doc, use finditer
		# this is as suggested by cs336 assignment doc
		counts_doc = {}
		for match in re.finditer(self.pretokenizer_pattern, document):
			match_word = match.group()
			if match_word not in counts_doc:
				counts_doc[match_word] = 0
			counts_doc[match_word] += 1

		return ([k for k in counts_doc.keys()], [v for v in counts_doc.values()])

	def apply_pretokenizer_list(self, document_list, return_counts=False):
		"""
		document_list is a list of strings
		returns a list of sub-components (flattened to 1 list)
		"""

		# if pretokenizer_pattern is None
		# return the list as as with return counts
		if self.pretokenizer_pattern is None:
			print(f"> warning: pretokenizer not set. returning original document as is.")
			return (document_list, [1] * len(document_list) if return_counts else None)

		# if we don't have to return counts
		if return_counts is False:
			chunks = []
			for document in document_list:
				chunks.extend(self.apply_pretokenizer(document, return_counts)[0])
			return (chunks, None)

		# if we have to return counts
		counts_doc = {}
		for document in document_list:
			chunks, counts = self.apply_pretokenizer(document, return_counts)
			for (cnk, cnt) in zip(chunks, counts):
				if cnk not in counts_doc:
					counts_doc[cnk] = 0
				counts_doc[cnk] += cnt

		return ([k for k in counts_doc.keys()], [v for v in counts_doc.values()])

	def get_pair_frequencies(self, list_of_lists, multipliers=None):
		"""
		given a list of lists, return pair_frequencies
		"""

		# handle the case if multipliers is None
		if multipliers is None:
			multipliers = [1] * len(list_of_lists)

		# instantiate an empty frequency map
		frequency_map = {}

		# iterate over all lists
		for (inner_list, mult) in zip(list_of_lists, multipliers):

			# skip if length of list is less than 2
			if len(inner_list) < 2: continue

			for i in range(1, len(inner_list)):
				pair = (inner_list[i-1], inner_list[i])
				if pair not in frequency_map:
					frequency_map[pair] = 0
				frequency_map[pair] += mult

		# return the final frequency map
		return frequency_map

	def merge_pair(self, document_chunks, pair, merge_idx):
		"""
		replace all instances of pair in document_chunks with merge_idx
		works in place, returns None
		"""

		merge_count = 0

		# iterate over all chunks
		for idx in range(len(document_chunks)):
			chunk = document_chunks[idx]

			# skip if length of chunk is less than 2
			if len(chunk) < 2: continue

			for i in range(1, len(chunk)):

				# if already replaced, continue
				if chunk[i-1] is None: continue

				# identify a match
				if chunk[i-1] == pair[0] and chunk[i] == pair[1]:
					chunk[i-1] = merge_idx
					chunk[i] = None
					merge_count += 1

			chunk = [x for x in chunk if x is not None]
			document_chunks[idx] = chunk

		# print(f"MERGED {pair} --> {merge_idx} | {merge_count}")

		return

	def merge_pair_and_update_pair_frequencies(self, 
		document_chunks,
		pair,
		merge_idx,
		byte_pair_frequencies,
		multipliers=None
	):
		"""
		this is a combination of 2 functions
		- merge_pair
		- get_pair_frequencies

		essentially, both tasks can be done in a single pass easily
		works in place, returns None
		updates document_chunks and byte_pair_frequencies
		"""

		merge_count = 0

		# handle the case if multipliers is None
		if multipliers is None:
			multipliers = [1] * len(document_chunks)

		# iterate over all chunks
		for idx in range(len(document_chunks)):
			chunk = document_chunks[idx]
			mult = multipliers[idx]

			# skip if length of chunk is less than 2
			if len(chunk) < 2: continue

			for i in range(1, len(chunk)):

				# if already replaced, continue
				if chunk[i-1] is None: continue

				# identify a match
				if chunk[i-1] == pair[0] and chunk[i] == pair[1]:

					# logic for updating byte pair frequencies
					# p, a, b, q --> p, c, None, q
					# -2, -1, 0, 1

					# give better names here itself
					a = chunk[i-1]
					b = chunk[i]
					c = merge_idx

					# p might be none because it got replaced in previous round
					p = None
					if i-2 >= 0:
						p = chunk[i-2]
						if p is None:
							p = merge_idx
							assert chunk[i-3] == merge_idx # this must've been the case

					# q case is normal
					q = None
					if i+1 < len(chunk):
						q = chunk[i+1]


					# 3 prior pairs get updated (reduced)
					# 2 new pairs get formed

					# 1: (a, b) --
					byte_pair_frequencies[(a, b)] -= mult
					if byte_pair_frequencies[(a, b)] == 0:
						del byte_pair_frequencies[(a, b)]

					# 2, 3: (p, a) -- && (p, c) ++
					# if p exists
					if p is not None:

						byte_pair_frequencies[(p, a)] -= mult
						if byte_pair_frequencies[(p, a)] == 0:
							del byte_pair_frequencies[(p, a)]

						if (p, c) not in byte_pair_frequencies:
							byte_pair_frequencies[(p, c)] = 0
						byte_pair_frequencies[(p, c)] += mult

					# 4, 5: (b, q) -- && (c, q) ++
					# if q exists
					if q is not None:

						byte_pair_frequencies[(b, q)] -= mult
						if byte_pair_frequencies[(b, q)] == 0:
							del byte_pair_frequencies[(b, q)]

						if (c, q) not in byte_pair_frequencies:
							byte_pair_frequencies[(c, q)] = 0
						byte_pair_frequencies[(c, q)] += mult

					# logic for merging 
					chunk[i-1] = merge_idx
					chunk[i] = None
					merge_count += 1


			chunk = [x for x in chunk if x is not None]
			document_chunks[idx] = chunk

		# print(f"MERGED {pair} --> {merge_idx} | {merge_count}")

		# no need to return this; updated in-place
		return byte_pair_frequencies

	def train_bpe(
		self,
		document_list,
		num_merges,
		early_stop=True,
		verbose=True,
		use_multipliers=True,
		single_pass=True,
	):
		"""
		use the bpe algorithm to build vocabulary
		simplifying assumption: special tokens are not included
		and are expected to be added after train bpe step
		"""

		# start the timer
		time_s = time.time()

		# ensure that special tokens are not yet added
		assert len(self.special_tokens) == 0

		print(f"\n{'='*20}\n")

		# step 1: apply pre-tokenizer
		# converts list[str] --> flatten(list[list[str]])
		document_chunks, chunk_counts = self.apply_pretokenizer_list(document_list, return_counts=use_multipliers)

		# step 2: convert each chunk to byte-stream
		document_chunks = [chunk.encode(self.encoding) for chunk in document_chunks]
	
		# step 3: convert byte-stream to token ids
		document_chunks = [[b for b in chunk] for chunk in document_chunks]

		# get all byte pair frequencies before we start
		# we're getting them a priori because it works well with cached refactor
		byte_pair_frequencies = self.get_pair_frequencies(document_chunks, multipliers=chunk_counts)

		# step 4: iterate at max num_merges times
		for _ in range(num_merges):
			
			# break if all possible pairs are exhausted
			if len(byte_pair_frequencies) == 0:
				break
			
			# get the highest frequency pair
			# pair = max(byte_pair_frequencies, key=byte_pair_frequencies.get)

			# get the highest frequency pair but now with a tie breaker
			# tie breaker selects the smallest lexicographic pair
			pair = max(byte_pair_frequencies,
				key=lambda p: (byte_pair_frequencies[p], -sum(p)))

			# break if the max frequency is 1; this is optional though
			if early_stop and byte_pair_frequencies[pair] == 1:
				break

			# conditional refactor for single pass optimization
			if single_pass is True:

				# in place updates document chunks and updates byte pair frequencies
				byte_pair_frequencies = self.merge_pair_and_update_pair_frequencies(document_chunks, pair, self.vocab_size,
					byte_pair_frequencies, multipliers=chunk_counts)

			else: # the standard double pass approach

				# perform the merge now
				self.merge_pair(document_chunks, pair, self.vocab_size)

				# get all byte pait frequencies after merge
				byte_pair_frequencies = self.get_pair_frequencies(document_chunks, multipliers=chunk_counts)
			
			# update the vocab_size, merges, vocab
			self.merges[pair] = self.vocab_size
			self.vocab[self.vocab_size] = self.vocab[pair[0]] + self.vocab[pair[1]]

			if verbose:
				print(f"added to vocab: {self.vocab_size} --> {self.vocab[self.vocab_size]}")
			
			self.vocab_size += 1

		# step 5: update reverse vocab as well
		self.vocab_bytes2idx = {v: k for (k, v) in self.vocab.items()}

		# end the timer
		time_e = time.time()

		print(f"\nvocabulary created in {time_e-time_s:.2f}s")
		print(f"vocab size = {self.vocab_size}")

		print(f"\n{'='*20}\n")

	def build_merge_forest(self):
		"""
		the optimized implementation of building a merge forest
		"""

		# create a disjoint set class of vocab size
		disjoint_set = DisjointSet(len(self.vocab))

		# iterate over all merges
		for (pair, target) in tqdm(self.merges.items()):

			# merge both pairs and the target
			disjoint_set.union(pair[0], pair[1])
			disjoint_set.union(pair[0], target)

		self.merge_forest = [disjoint_set.find(i) for i in range(len(self.vocab))]

		print(f"\ncreated a merge forest of size {disjoint_set.num_sets}")
		print(f"set size distribution: {Counter(disjoint_set.count)}")

		return disjoint_set.num_sets

	def build_prefix_tree(self):
		"""
		builds a prefix tree for the greedy encoder
		"""

		# start with an empty prefix tree
		self.prefix_tree = PrefixTree()

		# insert all items of vocab
		for token_bytes in tqdm(self.vocab_bytes2idx):
			raw_token_idx_list = [b for b in token_bytes]
			self.prefix_tree.addToken(raw_token_idx_list)

		return len(self.vocab_bytes2idx)

	def clear_cache(self):
		"""
		clear all encoding cache
		"""

		self.encoder_cache = {algo: {} for algo in self.tokenization_algorithms}

	def convert_token_ids_to_bytes(self, token_ids):
		"""
		a simple utility function to visualize token ids
		"""

		# get the decoded bytes
		decoded_bytes = [
			self.vocab[tok_id] if tok_id < len(self.vocab) else self.special_tokens[tok_id]
			for tok_id in token_ids
		]

		return decoded_bytes

	def save_tokenizer(self, dir_path):
		"""
		this function is akin to the .save_pretrained function in huggingface
		
		design decision
		save the entire tokenizer as a single file?
		or save into 3-4 separate files like vocab, merges, etc
		i think it is nice to save as 3+ separate files: vocab, merges, config

		# attributes to be saved

		-- self.encoding: str --> bpe_config.json

		-- self.vocab = {idx: bytes([idx]) for idx in range(0, 256)} --> bpe_vocab.json (str key, list val)
		-- self.vocab_size: int --> bpe_config.json
		-- self.vocab_bytes2idx --> build from self.vocab

		-- self.merges = {} --> bpe_merges.json (serialized)

		-- self.pretokenizer_pattern = None --> save as str or None in bpe_config.json

		-- self.special_tokens = {} # idx2bytes --> can be inferred from special_tokens_str2idx
		-- self.special_tokens_str2idx = {} # str2idx

		-- self.encoder_cache --> no need to save

		"""

		# if the dir doesn't exist, create it
		Path(dir_path).mkdir(exist_ok=True, parents=True)

		# create and save the config file
		bpe_config = {
			"encoding": self.encoding,
			"vocab_size": self.vocab_size,
			"pretokenizer_pattern": None if self.pretokenizer_pattern is None else self.pretokenizer_pattern.pattern,
			"start_token_id": self.start_token_id,
			"pad_token_id": self.pad_token_id,
		}

		with open(os.path.join(dir_path, "bpe_config.json"), "w+") as f:
			f.write(json.dumps(bpe_config, indent=4))

		# create and save the merges dict
		with open(os.path.join(dir_path, "bpe_merges.json"), "w+") as f:
			merges_serialized = {str(k): v for (k, v) in self.merges.items()}
			f.write(json.dumps(merges_serialized, indent=4))

		# create and save special_tokens
		with open(os.path.join(dir_path, "bpe_special_tokens.json"), "w+") as f:
			f.write(json.dumps(self.special_tokens_str2idx, indent=4))

		# create and save the vocab
		with open(os.path.join(dir_path, "bpe_vocab.json"), "w+") as f:
			vocab_deconstructed = {k: list(v) for (k, v) in self.vocab.items()}
			f.write(json.dumps(vocab_deconstructed, indent=4))

		return

	def load_tokenizer(self, dir_path):
		"""
		build the tokenizer from the saved tokenizer files

		# attributes to be loaded

		-- self.encoding: str --> bpe_config.json

		-- self.vocab = {idx: bytes([idx]) for idx in range(0, 256)} --> bpe_vocab.json (str key, list val)
		-- self.vocab_size: int --> bpe_config.json
		-- self.vocab_bytes2idx --> build from self.vocab

		-- self.merges = {} --> bpe_merges.json (serialized)

		-- self.pretokenizer_pattern = None --> save as str or None in bpe_config.json

		-- self.special_tokens = {} # idx2bytes --> can be inferred from special_tokens_str2idx
		-- self.special_tokens_str2idx = {} # str2idx

		-- self.encoder_cache --> empty

		"""

		# read the bpe_config.json file
		with open(os.path.join(dir_path, "bpe_config.json"), "r") as f:
			bpe_config = json.loads("".join(f.readlines()))

		# set whatever can be set from bpe_config
		self.encoding = bpe_config["encoding"]
		self.vocab_size = bpe_config["vocab_size"]
		self.start_token_id = bpe_config["start_token_id"]
		self.pad_token_id = bpe_config["pad_token_id"]

		if bpe_config["pretokenizer_pattern"] is not None:
			self.set_pretokenizer(bpe_config["pretokenizer_pattern"])

		# read the bpe_merges.json file and set the merges
		with open(os.path.join(dir_path, "bpe_merges.json"), "r") as f:
			merges_serialized = json.loads("".join(f.readlines()))
			self.merges = {ast.literal_eval(k): v for (k, v) in merges_serialized.items()}

		# read the bpe_special_merges.json file and set special tokens
		with open(os.path.join(dir_path, "bpe_special_tokens.json"), "r") as f:
			self.special_tokens_str2idx = json.loads("".join(f.readlines()))
			self.special_tokens = {v: k.encode(self.encoding) for (k, v) in self.special_tokens_str2idx.items()}

		# read the bpe_vocab.json file and set the vocab
		with open(os.path.join(dir_path, "bpe_vocab.json"), "r") as f:
			vocab_deconstructed = json.loads("".join(f.readlines()))
			self.vocab = {ast.literal_eval(k): bytes(v) for (k, v) in vocab_deconstructed.items()}
			self.vocab_bytes2idx = {v: k for (k, v) in self.vocab.items()}

		# empty encoder cache
		self.encoder_cache = {a: {} for a in self.tokenization_algorithms}

		return

	def apply_chat_template(self, document_list, apply_start_token=True, apply_stop_token=True):
		"""
		apply chat template will just add <start> and <end> to document
		"""

		# skip if start or pad tokens not set
		if self.start_token_id is None or self.pad_token_id is None:
			print("> warning: ignoring chat template because special tokens not set.")
			return document

		# get the string form of start and pad tokens
		start_token = ""
		if apply_start_token is True:
			start_token = self.special_tokens[self.start_token_id].decode(self.encoding)
		
		pad_token = ""
		if apply_stop_token is True:
			pad_token = self.special_tokens[self.pad_token_id].decode(self.encoding)

		# bookend the document with start and pad tokens
		return [f"{start_token}{document}{pad_token}" for document in document_list]
