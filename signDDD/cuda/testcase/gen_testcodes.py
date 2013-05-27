#!/usr/bin/env python
#-*- coding:utf8 -*-

import itertools as its

query_out_fn="test_query_hashcodes.txt"
corpus_out_fn = "test_corpus_hashcodes.txt"
corpus_out_bin_fn = "test_corpus_hashcodes_bin.txt"
pad_bin_code = lambda code_str,code_width:(code_width-len(bin(code_str))+2)*"0"+bin(code_str)[2:]

def gen_corpus_hashcodes(code_len=64, bit_set=2):
    q_code = 1 + (1<<(code_len/4)) + (1<<(code_len/2)) \
                                + (1<<(code_len*3/4)) + (1<<(code_len-1))
    with open(query_out_fn, "w") as out_f:
        out_f.write("%d\n" % q_code)

    bin_f = open(corpus_out_bin_fn, "w")
    with open(corpus_out_fn, "w") as out_f: 
        for p1, p2 in its.combinations(range(code_len), bit_set):
            code = (1<<p1) + (1<<p2)
            out_f.write("%d\n" % code)
            bin_f.write("%s\n" % pad_bin_code(code, code_len))
    bin_f.close()

def hamming_dist(c1, c2):
    xor = c1 ^ c2
    num = 0
    while xor:
        num += 1
        xor = xor & (xor-1)
    return num

def gen_golden(thd=3, golden_fn="golden_res.txt"):
    with open(golden_fn, "w") as f:
       for ln in open(query_out_fn):
           q_code = int(ln.strip())
           cnt = 0
           for c_ln in open(corpus_out_fn):
                c_code = int(c_ln.strip())
                if hamming_dist(c_code, q_code) < thd:
                    f.write("%d " % cnt)
                cnt += 1

if __name__ == "__main__":
    gen_corpus_hashcodes()
#    gen_query_hashcodes()
#    gen_golden(thd=4)
#    print hamming_dist(3<<10, 7<<8)
