"""
this script contains the custom dataset classes
"""

import pandas as pd

from torch.utils.data import Dataset

class CSV_Dataset(Dataset):
	"""
	read any csv/tsv dataset
	"""

	def __init__(self, filename, sep=','):

		# call the super constructor
		super().__init__()

		self.filename = filename
		self.df = pd.read_csv(filename, sep=sep)

	def __len__(self):
		return len(self.df)

	def __getitem__(self, idx):

		# convert idx row to dict and return
		return self.df.iloc[idx].to_dict()
