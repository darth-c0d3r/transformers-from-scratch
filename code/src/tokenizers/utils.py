"""
this file contains common utils for tokenizers
mainly keeping these separate to keep bpe_tokenizer.py clean
"""

class PrefixTree:
	"""
	the prefix tree class will help in implementing
	the greedy encoder for bpe tokenizer
	this offers a more optimized implementation
	rather than checking through the entire list
	"""

	def __init__(self):

		# initially, we will have no tokens
		self.is_token = False
		self.children = {}

	def addToken(self, byte_seq, idx=0):
		"""
		insert a byte sequence into the prefix tree
		"""

		# base case
		if len(byte_seq) == idx:
			self.is_token = True
			return

		# if byte_seq[idx] is not a member of children
		if byte_seq[idx] not in self.children:
			self.children[byte_seq[idx]] = PrefixTree()

		# insert the rest of the sequence into the correct child
		self.children[byte_seq[idx]].addToken(byte_seq, idx+1)

		return

	def getLongestPrefixMatch(self, byte_seq, idx=0):
		"""
		given a byte sequence, return the longest sequence that matches

		no match = None
		empty match = []
		"""

		# to start with, there could be an empty match or no match
		best_match = None
		if self.is_token:
			best_match = []

		# if we're out of byte seq, return current best
		if len(byte_seq) == idx:
			return best_match

		# or if there's nothing further to match, return current best
		if byte_seq[idx] not in self.children:
			return best_match 

		# try to get the longest match in children
		best_match_child = self.children[byte_seq[idx]].getLongestPrefixMatch(byte_seq, idx+1)

		# if the child return NULL
		if best_match_child is None:
			return best_match

		# else append current char
		return [byte_seq[idx]] + best_match_child
