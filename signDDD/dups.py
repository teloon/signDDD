#!/usr/bin/env python
#-*- coding:utf8 -*-

import sys
import os
from gen_corpus_hashcode import gen_corpus_hashcode
from query import cnt_ln
from query import load_corpus_hashcode
from query import show_progress
from glo import CODE_WIDTH
from glo import BIT_DIFF_THD
from glo import INT_WIDTH
from glo import WORD_HASHCODE_PATH
from lib import pad_bin_code

def execute_gpu(corpus_hashcode_path, output_path, encoding="utf8"):
    if not os.path.exists(corpus_hashcode_path):
        print "corpus hashcode path(", corpus_hashcode_path, ") not exists"
        print "generate the corpus hashcode first"
        return
    #TODO: currently GPU method only supports 32bit
    gpu_exe_fn = "cuda_int32_match" if CODE_WIDTH==32 else "cuda_int64_match"
    ln_no = cnt_ln(corpus_hashcode_path)
    cmd = "./%s %s %s %s %d %d %d" % (gpu_exe_fn, corpus_hashcode_path, corpus_hashcode_path, \
                                        output_path, ln_no, ln_no, BIT_DIFF_THD)
    #print cmd
    ret = os.popen(cmd).read().strip()
    #print ret

def execute_cpu(corpus_hashcode_path, output_path, encoding="utf8"):
    if not os.path.exists(corpus_hashcode_path):
        print "corpus hashcode path(", corpus_hashcode_path, ") not exists"
        print "generate the corpus hashcode first"
        return
    query_hashcode_lst = corpus_hashcode_lst = load_corpus_hashcode(corpus_hashcode_path)
    query_code_num = corpus_code_num = len(corpus_hashcode_lst)
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
    corpus_path = "corpus/zhidao_1000"
    corpus_hashcode_path = corpus_path + "_hashcode"
    encoding = "utf8"
    print "generating corpus hashcode"
    gen_corpus_hashcode(corpus_path, corpus_hashcode_path, word_hashcode_path=WORD_HASHCODE_PATH, encoding=encoding)

    print "executing queries"
    result_path = "result/zhidao_1000"
    #execute_cpu(corpus_hashcode_path, result_path, encoding=encoding)
    execute_gpu(corpus_hashcode_path, result_path, encoding=encoding)
    print "finished"

if __name__ == "__main__":
    example()
