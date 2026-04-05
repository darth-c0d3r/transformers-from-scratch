# Overview

A from-scratch implementation of the modern LLM stack. Covers BPE tokenizer training and encoding (with optimizations like fast parallel merges, phantom token pruning, and greedy prefix matching), and end-to-end transformer training with AdamW, RoPE, GQA, and KV caching. Built to understand what's really happening under the hood. Started off as an assignment for the Stanford CS336 (2025) course.

# Task 1: Build Tokenizers

## Tokenization Optimizations & Modifications

### Byte-Pair Encoding
- Standard BPE Implementations.
- Add necessities like pre-tokenization and optimizations like parallelization.

### 2-Pointer Greedy Approach
- Rather than merging individual pairs, just try to get the longest prefix match.
- Current implementation is slower than even naive BPE, but is almost similar in compression ratio.
- I wonder if use prefix-trees etc. can help speeden it up.

### Fast Tokenizer
- In a single iteration, can club multiple byte pairs if they have non overlapping bytes.
- Need to create a forest of disjoint node sets of merges.
- Returns exact same merges as vanilla BPE.
- But number of disjoint node sets is usually very low.
- Perhaps strict disjointness is too strict?

### Phantom Tokens
- While training bpe, a lot of merges are actually subsets of larger words.
	- For e.g., the order of merges is: resul, result, results or ass, asso, associ, associated.
- The thing is that when encoding, the earlier merges would be likely merged to higher indices.
- This means that lower index tokens might be very rare (even non-present) in a corpus.
- So won't it be wasteful to have those in the vocabulary?
- Can we algorithmically prune them out?
- Phantom Tokens: keep them around in bpe but remove them from model embeddings.
- If they ever come up after training the tokenizer, just decompose them to bytes.
	- This should be very rare
- Nearly 15% of the entire vocabulary doesn't come up even when tokenizing the entire training data.
- Chat-GPT Brainstorming: https://chatgpt.com/share/68ecbcb1-3f78-800f-8d92-0af2104cdbf4

### Tokenizing across Word Boundaries
- We already have a 2 stage tokenization.
- We can continue like this, and add another stage.
- To tokenize co-occurring words together.
	- If 2 adjacent words are tokenized into single tokens.
	- And they come together quite often; they can be merged.
	- This way, we can efficiently step outside of bounds set by pre-tokenization.

# Model Training
	
## On AdamW Optimizer
- AdamW Optimizer combines weight updates and weight decay in a decoupled way
- In weight updates, it uses first and second order moments
- It's updates are proportional to m1/m2
- m1 gives the directional momentum while m2 gives the magnitude of steps

- m1 being large means that we've gained a lot of momentum in that direction
- so it makes sense to keep going in that direction, especially if m2 is small
- m2 being small means we have a lot of small small steps accumulated in that direction
- and now is a good time to go harder in the same direction

- m1 being small and m2 being large
- means that we have been taking large steps but going back and forth
- we need to take smaller steps

# Commands

```bash
# ===== TRAINING THE TOKENIZER ===== #

# for sanity tests
cd ./code/src/tokenizers/
python test.py

# for training
cd ./code/src/
python train_tokenizer.py --config_filename ../config/train_tokenizer.json

# ===== TRAINING THE TRANSFORMER MODEL ===== #

# for sanity tests
cd ./code/src/transformer_lm/
python test.py

# for training
cd ./code/src/
python train_model.py --config_filename ../config/train_model.json

# ===== GENERATING FROM A TRAINED MODEL ===== #

# for generation
cd ./code/src/
python generate_interactive.py --config_filename ../config/generate.json

```


# To-Do

Implement all Tokenizers
	~~ character-based tokenizer [ditch]
	-- byte-based tokenizer
	~~ word-based tokenizer [ditch]
	- sentence piece tokenizer
	-- bpe tokenizer
		-- add provision for special rules
			-- special tokens
			-- numbers always tokenized separately (llama)
		-- dictionary of merges implementation of encode
		-- double pointer greedy implementation of encode
		-- compare the 2 for correctness and time complexity
		-- optimized encoding (cs 336)
	- hoffmann encoding?
	-- implement helper functions
		-- bytes per token
		-- chars per token
	-- pre tokenization
		-- gpt2
		-- gpt4
	-- special tokens provision
	- a pure c++ implementation of tokenizer encoding and decoding?
	-- multi-threaded: just split into n by space and run n processes? class fxn?
	-- move testing code out of main file
	-- counter over pre-tokenizer while training: 2x speedup in training
	-- cached decoding
	- on scale training / encoding
	-- Train on List of Documents

E2E Transformers
	- Understand Initializations
	-- Multiple Non Linearities
	-- Multiple Position Embeddings
		-- RoPE
		-- Sinusoidal
		-- NoPE
		-- make fwd pass compatible
	- Weight tying and untying
	- Attentions:
		-- MultiHeadAttention
		-- MultiQueryAttention
		-- GroupedQueryAttention
		-- LatentMultiHeadAttention
		- SparseAttention, etc.
	- .generate
		-- Greedy
		- Sampled
		-- Max Tokens
		- Top P
		- Top K
		-- Temperature
		- Beam Search
	- Naive KV Cache
	-- Padding Tokens in Attention Mask
	-- FLOPs Analysis
		-- Relation of L vs D
		-- Verify with ChatGPT
	-- Verify RoPE Similarity by plotting
		-- random distributions of K,Q
		-- exact same values of K,Q
	- Train on Film Subtitles
		-- Write a build-up function
	- Create a Document of Learnings and Experiments (Graphs)
	- Shouldn't we ideally also save dataloader state for continued training?
	- Label smoothing
	-- Think about ideal CE Loss given V (max=log(V), min or ideal = ??)
	- How large a model to learn IID data? correlated data?
	- Train on Tiny Stories
	- Ablations
