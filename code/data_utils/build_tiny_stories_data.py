"""
build the tiny stories dataset fron hugging-face
"""

from datasets import load_dataset

ds = load_dataset("roneneldan/TinyStories")

print(ds)

# print([sample for sample in ds['train'][:4]])
print(ds['train'][:4])