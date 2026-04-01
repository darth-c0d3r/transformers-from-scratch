"""
use this script to convert a .srt file to sentences
these sentences can be used for training a model
"""

# ================================================================================ #

from argparse import ArgumentParser
from tqdm import tqdm
from pathlib import Path
import os

import random
import nltk
import pandas as pd

# ================================================================================ #

# --- Step 0: Ensure 'punkt' tokenizer is available ---

try:
	nltk.data.find("tokenizers/punkt")
	nltk.data.find("tokenizers/punkt_tab")
	print(f"using pre-installed punkt...")
except LookupError:
	print("downloading punkt...")
	nltk.download("punkt")
	nltk.download("punkt_tab")

# ================================================================================ #

def represents_int(s):
	try:
		# Use s.strip() to handle leading/trailing whitespace
		int(s.strip()) 
		return True
	except ValueError:
		# A ValueError is raised if the string cannot be converted to an int
		return False

def merge_short_strings(strings, threshold):
	result = []
	pending = ""
	
	for s in strings:
		current = pending + s
		pending = ""
		
		if len(current) < threshold:
			pending = current + " "
		else:
			result.append(current)
	
	if pending:
		if result:
			result[-1] += " " + pending.strip()
		else:
			result.append(pending.strip())
	
	return result

# ================================================================================ #

def build_dataset(srt_filename, output_dir):
	"""
	convert the srt file to a tsv file
	"""

	print(f"\nbuilding dataset from {srt_filename} ...\n")

	# ================================================================================ #

	# read all sentence parts from srt file
	sentence_parts = []

	# go through all the lines of srt_filename
	with open(srt_filename, "r") as srt_file:

		# get all lines
		all_lines = srt_file.readlines()

		prev_empty = True

		# iterate over all lines
		for line in tqdm(all_lines, desc="watching movie..."):

			# strip the line
			line = line.strip()

			# skip if empty
			if len(line) == 0:
				prev_empty = True
				continue

			# skip if it is int and prev was empty
			if prev_empty and represents_int(line):
				# prev empty is false now
				prev_empty = False
				continue

			# skip if it is time range
			if "-->" in line:
				# prev empty is false now
				prev_empty = False
				continue

			# append to sentence parts
			sentence_parts.append(line)

	# ================================================================================ #

	# convert sentence parts to block of text
	# we'll convert this to nice and clean sentences
	block_of_text = " ".join(sentence_parts)
	del sentence_parts

	# convert the block of text to logical sentences
	sentences = nltk.sent_tokenize(block_of_text)
	sentences = [s.replace("\n", " ").replace("\t", " ").strip() for s in sentences]
	del block_of_text

	# if a sentence is too short, merge it with next
	sentences = merge_short_strings(sentences, 64)

	# shuffle all sentences
	random.shuffle(sentences)

	# ================================================================================ #
	# print some stats and do some saves

	# all the saving and stating here
	total_sentences = len(sentences)
	train_sentences = int(total_sentences * 0.9)
	longest_sentence = len(max(sentences, key=len))
	average_sentence = sum([len(s) for s in sentences])/total_sentences

	# prints
	print(f"total sentences  = {total_sentences}")
	print(f"train sentences  = {train_sentences}")
	print(f"longest sentence = {longest_sentence} chars")
	print(f"average sentence = {average_sentence:.2f} chars\n")

	# saves
	pd_train = pd.DataFrame({"Document": sentences[:train_sentences]})
	pd_train.to_csv(os.path.join(output_dir, "train.tsv"), sep="\t", index=False, header=True)

	pd_val = pd.DataFrame({"Document": sentences[train_sentences:]})
	pd_val.to_csv(os.path.join(output_dir, "val.tsv"), sep="\t", index=False, header=True)

	# ================================================================================ #

if __name__ == '__main__':
	"""
	main routine to set up files and folders
	"""

	# set up argument parser
	parser = ArgumentParser(description="srt to tsv converter for subtitles")
	parser.add_argument("--srt_filename", type=str, help="path to input srt file")
	parser.add_argument("--output_dir", type=str, required=True, help="output dir path")
	args = parser.parse_args()

	# sanity checks
	assert Path(args.srt_filename).is_file()

	# create output dir
	Path(args.output_dir).mkdir(exist_ok=True, parents=True)

	build_dataset(args.srt_filename, args.output_dir)
