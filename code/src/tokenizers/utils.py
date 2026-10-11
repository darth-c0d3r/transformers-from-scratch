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

class DisjointSet:
	"""
	the disjoint set class will help in implementing the
	"fast" encoding algorithm
	we will put all the co-ocurring tokens in a single set
	that way, for r.g., for multi-lingual vocabs
	we can merge multiple tokens at the same time
	"""

	def __init__(self, size):
		"""
		initialize a disjoint set class of size
		"""

		# initially, all items are in their own sets
		self.parent = [i for i in range(size)]

		# keep track of counts under each tree (only relevnt for top level)
		self.count = [1] * size

		# keep a track of num sets
		self.num_sets = size

	def find(self, idx):
		"""
		return the base set of item idx
		implements an optimization
		"""

		# base case
		if self.parent[idx] == idx:
			return idx

		# otherwise get the next node
		next_node = self.parent[idx]

		# get the base set of next node
		top_node = self.find(next_node)

		# create a direct connection
		self.parent[idx] = top_node

		return top_node

	def union(self, idx_1, idx_2):
		"""
		take a union of sets containing idx_1 and idx_2
		"""

		# get parents of both nodes
		p1 = self.find(idx_1)
		p2 = self.find(idx_2)

		# if they are both already in same set, return
		if p1 == p2:
			return

		# otherwise, pin the smaller tree onto larger one
		if self.count[p1] < self.count[p2]:
			p1, p2 = p2, p1

		# count[p1] >= count[p2]
		# add p2 to p1
		self.parent[p2] = p1
		self.count[p1] += self.count[p2]
		self.count[p2] = None

		# reduce the set size by 1
		self.num_sets -= 1

		return

	def set_size(self, idx):
		"""
		return the set size of an idx
		"""

		return self.count[self.find(idx)]
