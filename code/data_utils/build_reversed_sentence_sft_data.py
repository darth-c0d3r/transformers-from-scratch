"""
use this script to convert a pretraining dataset to a reversed doc set for sft
"""

import pandas as pd
import os
from pathlib import Path

INPUT_DIR = "../../data/movie_subs/dune_part_1/"
OUTPUT_DIR = "../../data/movie_subs/dune_part_1__sft_reversed/"

Path(OUTPUT_DIR).mkdir(exist_ok=True)

files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".tsv")]

def reverse_sentence(sentence):
	return " ".join(reversed(sentence.split(" ")))

for file in files:
	print(f"\nprocessing {file}")

	df = pd.read_csv(os.path.join(INPUT_DIR, file), sep='\t')
	df.rename(columns={'Document': 'Source'}, inplace=True)
	# print(df)
	df['Target'] = df['Source'].apply(reverse_sentence)

	df.to_csv(os.path.join(OUTPUT_DIR, file), sep='\t', index=False, header=True)
