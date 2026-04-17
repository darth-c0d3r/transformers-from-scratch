# Overview

A from-scratch implementation of the modern LLM stack. Covers BPE tokenizer training and encoding (with optimizations like fast parallel merges, phantom token pruning, and greedy prefix matching), and end-to-end transformer training with AdamW, RoPE, GQA, and KV caching. Built to understand what's really happening under the hood. Started off as an assignment for the Stanford CS336 (2025) course. Some features are not yet implemented, it's a WIP.

# Commands

```bash
# ===== TRAINING THE TOKENIZER ===== #

# for sanity tests
cd ./code/src/tokenizers/
python test.py

# for training
# the dataset tsv files must have `Document` column
cd ./code/src/
python train_tokenizer.py --config_filename ../config/train_tokenizer.json

# ===== TRAINING THE TRANSFORMER MODEL ===== #

# for sanity tests
cd ./code/src/transformer_lm/
python test.py

# for training (pretraining)
# the dataset tsv files must have `Document` column
cd ./code/src/
python train_model__pretrain.py --config_filename ../config/train_model__pretrain.json

# for training (supervised fine-tuning)
# the dataset tsv files must have `Source` and `Target` columns
cd ./code/src/
python train_model__sft.py --config_filename ../config/train_model__sft.json

# ===== GENERATING FROM A TRAINED MODEL ===== #

# for generation
cd ./code/src/
python generate_interactive.py --config_filename ../config/generate.json

```
