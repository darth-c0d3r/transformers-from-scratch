import sys
sys.path.append("../")

from bpe_tokenizer import BPE_Tokenizer
import ast

logfile = "./logs"
tokenizer_dir = "./temp"

tokenizer = BPE_Tokenizer()
tokenizer.load_tokenizer(tokenizer_dir)

# initialize tokenizer logs data here
with open(logfile, "r") as f:
	all_lines = f.readlines()

def parse(x):

	x = x.split("-->")
	pair = ast.literal_eval(x[0])
	data = x[1].split("|")
	target = int(data[0])
	count = int(data[1])
	return [pair, target, count]


merge_data = [parse(l[len("MERGED"):].strip()) for l in all_lines if l.startswith("MERGED")]
final_tokens = ast.literal_eval(all_lines[-1][len("FINAL"):])

print(merge_data[:5], len(merge_data))
# pair, target, count

print(final_tokens[:5], len(final_tokens))
# final token

# we want to capture journey of each token in general
# we'll see how many tokens came up in between
# but were always merged up
# we'll save a touple: (in_process_count, final_count)

process_data = {
	k: [0, 0] for k in tokenizer.vocab
}

for idx in final_tokens:
	if idx in process_data:
		process_data[idx][1] += 1

for data in merge_data:

	process_data[data[0][0]][0] += data[2]
	process_data[data[0][1]][0] += data[2]

# we can use this process_data to print multiple stats

print("tokens that occur most in the process but never eventually")

curr = sorted([(v[0], k) for (k, v) in process_data.items() if v[1] == 0 and v[0] > 0], reverse=True)
for a, b in curr[:20]:
	print(tokenizer.vocab[b], a)

print(len(curr), len(tokenizer.vocab), len(curr)*100./len(tokenizer.vocab))
