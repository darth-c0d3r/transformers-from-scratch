# use this scimple script to estimate number of flops in decoder LM

import json
import sys
# sys.path.append("../")
# from decoder_lm import Decoder_LM

"""
Counting FLOPs for Forward Pass

Decoder Block x n

	Layer Norm x 2
		5 * b * l * d_m [x]

	Self Attention x 1
		Projection 1
			3 * b * l * (2 * d_m * h * d_k)
		Projection 2
			1 * b * l * (2 * d_m * h * d_k)
		Ro-PE Embeddings
			3 * b * l * h * d_k [x]
		Self Attention
			Similarities
				b * h * l * (2 * l * d_k)
			Softmax
				2 * b * h * l * l [x]
			Multiply with V
				b * h * l * (2 * l * d_k)
		Residual Connection
			b * l * d_m  [x]

	FFN x 1
		Up-Project
			b * l * (2 * d_m * d_h)
		Down-Project
			b * l * (2 * d_h * d_m)
		SwiGLU
			Projection
				b * l * (2 * d_m * d_h)
			Element Wise Gating + SiLU
				3 * b * l * d_h [x]

		Residual Connection
			b * l * d_m [x]

LM Head
	Projection
		b * l * (2 * d_m * v)
	Softmax
		2 * b * (l * v)

4 * b * l * (2 * d_m * h * d_k)
b * h * l * (2 * d_k * d_k)
b * h * l * (2 * d_k * l)
3 * b * l * (2 * d_m * d_h)

	2 * b * l * h * [ (4 * d_m * d_k) + (d_k * d_k) + (d_k * l)] = O(b * l * h * (d_m^2 + d_k^2 + d_k*l))
	6 * b * l * (d_m * d_h) = O(b * l * h * (d_m * d_h))


b * l * (2 * d_m * v) = O(b * l * d_m * v)
2 * b * l * v [x]

Assuming d_h = O(d_m), d_k * h = O(d_m) = O(d)

O(n * b * l * d^2) + O(n * b * d * l^2) + O(b * l * d * v)

O(b * l * d * (n * d + n * l + v))

"""

def compute_flops(b, l, d_m, d_h, d_k, h, n, v):

	flops = {}

	flops["decoder.layer_norms"] = 2 * 5 * b * l * d_m

	flops["decoder.self_attn.projections"] = 4 * b * l * (2 * d_m * h * d_k)
	flops["decoder.self_attn.rope"] = 3 * b * l * h * d_k
	flops["decoder.self_attn.similarities"] = b * h * l * (2 * l * d_k)
	flops["decoder.self_attn.softmax"] = 2 * b * h * l * l
	flops["decoder.self_attn.values"] = b * h * l * (2 * l * d_k)
	flops["decoder.self_attn.residual"] = b * l * d_m

	flops["decoder.self_attn"] = sum([flops[v] for v in flops if v.startswith("decoder.self_attn.")])

	flops["decoder.ffn.projections"] = 3 * b * l * (2 * d_m * d_h)
	flops["decoder.ffn.silu"] = 3 * b * l * d_h
	flops["decoder.ffn.residual"] = b * l * d_m

	flops["decoder.ffn"] = sum([flops[v] for v in flops if v.startswith("decoder.ffn.")])

	flops["decoder_block"] = flops["decoder.self_attn"] + flops["decoder.ffn"]

	flops["decoder_full"] = n * flops["decoder_block"]

	flops["lm_head.projection"] = b * l * (2 * d_m * v)
	flops["lm_head.softmax"] = 2 * b * (l * v)
	flops["lm_head"] = flops["lm_head.projection"] + flops["lm_head.softmax"]

	flops["total_flops"] = flops["decoder_full"] + flops["lm_head"]

	# print(json.dumps(flops, indent=4))
	return flops

def compute_flops_kv_cache(b, l, d_m, d_h, d_k, h, n, v):

	flops = {}

	flops["decoder.layer_norms"] = 2 * 5 * b * 1 * d_m

	flops["decoder.self_attn.projections"] = 4 * b * 1 * (2 * d_m * h * d_k)
	flops["decoder.self_attn.rope"] = 3 * b * 1 * h * d_k
	flops["decoder.self_attn.similarities"] = b * h * 1 * (2 * l * d_k)
	flops["decoder.self_attn.softmax"] = 2 * b * h * 1 * l
	flops["decoder.self_attn.values"] = b * h * 1 * (2 * l * d_k)
	flops["decoder.self_attn.residual"] = b * 1 * d_m

	flops["decoder.self_attn"] = sum([flops[v] for v in flops if v.startswith("decoder.self_attn.")])

	flops["decoder.ffn.projections"] = 3 * b * 1 * (2 * d_m * d_h)
	flops["decoder.ffn.silu"] = 3 * b * 1 * d_h
	flops["decoder.ffn.residual"] = b * 1 * d_m

	flops["decoder.ffn"] = sum([flops[v] for v in flops if v.startswith("decoder.ffn.")])

	flops["decoder_block"] = flops["decoder.self_attn"] + flops["decoder.ffn"]

	flops["decoder_full"] = n * flops["decoder_block"]

	flops["lm_head.projection"] = b * 1 * (2 * d_m * v)
	flops["lm_head.softmax"] = 2 * b * (1 * v)
	flops["lm_head"] = flops["lm_head.projection"] + flops["lm_head.softmax"]

	flops["total_flops"] = flops["decoder_full"] + flops["lm_head"]

	# print(json.dumps(flops, indent=4))
	return flops

b = 1
v = 151936
l = 16384
n = 28
d_m = 128 * 16
h = 16
d_k = 128
d_h = 6144

# model = Decoder_LM(v, d_m, d_h, d_k, h, n, l)
# print(f"params = {sum([p.numel() for p in model.parameters()])}")

print(compute_flops(b, l, d_m, d_h, d_k, h, n, v))
print(compute_flops_kv_cache(b, l, d_m, d_h, d_k, h, n, v))


# vary d_model and seq_len and % self attn and % ffn
# b = 1
# v = 1024
# l_list = [2**i for i in range(32)]
# d_list = [2**i for i in range(3, 12)]
# n = 16
# h = 8
# d_k = None
# d_h = None

# for l in l_list:
# 	for d in d_list:
# 		flops = compute_flops(
# 			b,
# 			l,
# 			d,
# 			(d*4),
# 			(d*2)//h,
# 			h,
# 			n,
# 			v
# 			)

# 		p_attn = flops["decoder.self_attn"]*100./(flops["decoder.self_attn"]+flops["decoder.ffn"])
# 		p_ffn = flops["decoder.ffn"]*100./(flops["decoder.self_attn"]+flops["decoder.ffn"])

# 		print(f"l = {l} \t d = {d} \t p_attn = {p_attn:.2f}%")
# 		print(f"l = {l} \t d = {d} \t p_ffn = {p_ffn:.2f}%")

