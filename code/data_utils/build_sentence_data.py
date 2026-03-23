"""
use this script to build data from any url

for training tokenizer
we need a full corpus, including newlines

for training model
we need "documents"
for simplicity, we can treat a paragraph or sentence as one document
"""

from argparse import ArgumentParser
import os
from pathlib import Path
import random

import pandas as pd
import requests
from bs4 import BeautifulSoup
import nltk
from readability import Document

# --- Step 0: Ensure 'punkt' tokenizer is available ---
try:
	nltk.data.find("tokenizers/punkt")
	nltk.data.find("tokenizers/punkt_tab")
	print(f"using pre-installed punkt...")
except LookupError:
	print("downloading punkt...")
	nltk.download("punkt")
	nltk.download("punkt_tab")

def build_dataset(url, output_dir):

	headers = {
		"User-Agent": "Mozilla/5.0 (compatible; ArticleExtractor/1.0; +https://github.com/yourname)"
	}

	# Step 1: Fetch the webpage
	response = requests.get(url, headers=headers)
	html_content = response.text

	# --- Step 2: Extract main readable content ---
	doc = Document(html_content)
	title = doc.title()
	html = doc.summary()  # returns HTML of the main article body

	# --- Step 3: Clean to plain text ---
	soup = BeautifulSoup(html, "html.parser")
	text = soup.get_text(separator=" ")
	text = " ".join(text.split())

	# add all the replaces
	text = text.replace(" .", ".").replace(" ,", ",")

	# --- Step 4: Split into sentences ---
	sentences = nltk.sent_tokenize(text)
	sentences = [title] + sentences
	sentences = [s.replace("\n", " ") for s in sentences]

	# shuffle all sentences
	random.shuffle(sentences)

	# all the saving and stating here
	total_sentences = len(sentences)
	train_sentences = int(total_sentences * 0.9)
	longest_sentence = len(max(sentences, key=len))
	average_sentence = sum([len(s) for s in sentences])/total_sentences

	# prints
	print(f"total sentences = {total_sentences}")
	print(f"train sentences = {train_sentences}")
	print(f"longest sentence = {longest_sentence} chars")
	print(f"average sentence = {average_sentence:.2f} chars\n")

	# saves
	pd_train = pd.DataFrame({"Document": sentences[:train_sentences]})
	pd_train.to_csv(os.path.join(output_dir, "train.tsv"), sep="\t", index=False, header=True)

	pd_val = pd.DataFrame({"Document": sentences[train_sentences:]})
	pd_val.to_csv(os.path.join(output_dir, "val.tsv"), sep="\t", index=False, header=True)

	with open(os.path.join(output_dir, "url"), "w+", encoding="utf-8") as f:
		f.write(url)

if __name__ == '__main__':

	# set up argument parser
	parser = ArgumentParser(description="build a train + validation dataset for pre-training from a url")
	parser.add_argument("--url", type=str, required=True, help="url to the article")
	parser.add_argument("--output_dir", type=str, required=True, help="output dir path")
	args = parser.parse_args()

	# create output dir
	Path(args.output_dir).mkdir(exist_ok=True, parents=True)

	# call the main function
	build_dataset(args.url, args.output_dir)
