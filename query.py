#!/usr/bin/env python
#-*- coding:utf8 -*-

import sys
import os
from gen_corpus_hashcode import gen_corpus_hashcode
from glo import CODE_WIDTH
from glo import BIT_DIFF_THD
from glo import INT_WIDTH
from glo import WORD_HASHCODE_PATH
from lib import pad_bin_code

def show_progress(cur, total):
    show_cur = 100*cur/total
    print "[%s%s] %d %.2f%%" % (">"*show_cur, " "*(100-show_cur), cur, 100.0*cur/total), "\r",
    sys.stdout.flush()

def load_corpus_hashcode(codes_path):
    src_f = open(codes_path, "r")
    corpus_hashcodes = []
    for ln in src_f:
        corpus_hashcodes.append(int(ln.strip()))
    return corpus_hashcodes

def cnt_ln(file_path):
    cnt = 0
    for ln in open(file_path):
        cnt += 1
    return cnt

def execute_gpu(query_lst, corpus_hashcode_path, output_path, encoding="utf8"):
    query_path = "query/query"
    with open(query_path, "w") as out_f:
        out_f.write("\n".join([q.encode(encoding) for q in query_lst]))
    query_hashcode_path = "query/query_hashcode"
    gen_corpus_hashcode(query_path, query_hashcode_path, word_hashcode_path=WORD_HASHCODE_PATH, encoding="utf8")
    #TODO: currently GPU method only supports 32bit
    gpu_exe_fn = "cuda_int32_match" if CODE_WIDTH==32 else "cuda_int64_match"
    cmd = "./%s %s %s %s %d %d %d" % (gpu_exe_fn, corpus_hashcode_path, query_hashcode_path, \
                                        output_path, cnt_ln(corpus_hashcode_path), cnt_ln(query_path), BIT_DIFF_THD)
    #print cmd
    ret = os.popen(cmd).read().strip()
    #print ret

def execute_cpu(query_lst, corpus_hashcode_path, output_path, encoding="utf8"):
    query_path = "query/query"
    with open(query_path, "w") as out_f:
        out_f.write("\n".join([q.encode(encoding) for q in query_lst]))
    query_hashcode_path = "query/query_hashcode"
    gen_corpus_hashcode(query_path, query_hashcode_path, word_hashcode_path=WORD_HASHCODE_PATH, encoding="utf8")
    query_hashcode_lst = load_corpus_hashcode(query_hashcode_path)
    corpus_hashcode_lst = load_corpus_hashcode(corpus_hashcode_path)
    corpus_code_num = len(corpus_hashcode_lst)
    query_code_num = len(query_hashcode_lst)
    result = []
    for i in range(query_code_num):
        query = query_hashcode_lst[i]
        result.append([])
        for j in range(corpus_code_num):
            diff_num = 0
            xor_val = query^corpus_hashcode_lst[j]
            while xor_val!=0:
                if xor_val&1 == 1:
                    diff_num += 1
                xor_val = xor_val>>1
            if diff_num < BIT_DIFF_THD:
                result[i].append(j)
        show_progress(i+1, query_code_num)
    print "\n"
#    print ">>finish query NO.%d @ %s" % (i, time.ctime())
#    print "### finish comparison ###"
    out_f = open(output_path, "w")
    for q_idx,dups in enumerate(result):
#        out_f.write("###%s\t%d\n" % (pad_bin_code(query_hashcode_lst[q_idx], CODE_WIDTH), q_idx))
#        for dup_idx in dups:
#            out_f.write("%s\t%d\n" % (pad_bin_code(corpus_hashcode_lst[dup_idx], CODE_WIDTH), dup_idx))
        out_f.write("%s\n" % "\t".join([str(idx) for idx in dups]))
    out_f.close()

def example():
    query_lst = [u"谁的歌最好听？", u"陈奕迅的富士山下"]
    corpus_path = "corpus/zhidao_1000"
    corpus_hashcode_path = corpus_path + "_hashcodes"
    print "generating corpus hashcode"
    gen_corpus_hashcode(corpus_path, corpus_hashcode_path, word_hashcode_path=WORD_HASHCODE_PATH)

    print "executing queries"
    result_path = "result/zhidao_1000"
    execute_cpu(query_lst, corpus_hashcode_path, result_path)
    #execute_gpu(query_lst, corpus_hashcode_path, result_path)
    print "finished"

if __name__ == "__main__":
    example()
