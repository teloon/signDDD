#!/usr/bin/env python
#-*- coding:utf8 -*-

"""This module implements the core process of finding all near
duplicate documents in the given corpus. Given a **corpus hashcode file**,
this module will output all the duplicates in some specified file.

In the module there're two ways of finding dups(:func:`signDDD.dups.execute_cpu` and
:func:`signDDD.dups.execute_gpu`), whose main difference lies in how to compute the
hamming distance of two hashcodes. But the main algorithms are the same:

1. **XOR** two hashcodes to index the bits that are different
2. **Count** the number of bit difference

Generally, the GPU-based version achieves about 1500X speedup in
our GPU and CPU platform.

Example usage for CPU-based querying:

.. code-block:: python
   :linenos:

   corpus_path, result_path = "corpus/zhidao_1000", "result/zhidao_1000"
   corpus_hashcode_path = corpus_path + "_hashcodes"
   gen_corpus_hashcode(corpus_path, corpus_hashcode_path, \\
                        word_hashcode_path=WORD_HASHCODE_PATH, encoding="utf8")
   execute_cpu(corpus_hashcode_path, result_path, encoding="utf8")

"""

import os
from hashcode import gen_corpus_hashcode
from lib import cnt_ln
from lib import load_corpus_hashcode
from lib import show_progress
from glo import CODE_WIDTH
from glo import BIT_DIFF_THD
from glo import INT_WIDTH
from glo import WORD_HASHCODE_PATH
from lib import pad_bin_code

def execute_gpu(corpus_hashcode_path, output_path, encoding="utf8"):
    """This function implements the GPU-based version of finding duplicates.

    .. caution::

       The first parameter is the path of the corpus **hashcode** file. So if not
       existing, the hashcode file needs to be generated in advance by calling
       :func:`signDDD.hashcode.gen_corpus_hashcode` function.

    """
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
    """This function implements the CPU-based version of finding duplicates.

    .. caution::

       The first parameter is the path of the corpus **hashcode** file. So if not
       existing, the hashcode file needs to be generated in advance by calling
       :func:`signDDD.hashcode.gen_corpus_hashcode` function.

    """
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
    out_f = open(output_path, "w")
    for q_idx,dups in enumerate(result):
        out_f.write("%s\n" % "\t".join([str(idx) for idx in dups]))
    out_f.close()

def example():
    corpus_path = "corpus/zhidao_1000"
    corpus_hashcode_path = corpus_path + "_hashcode"
    encoding = "utf8"
    print "generating corpus hashcode"
    gen_corpus_hashcode(corpus_path, corpus_hashcode_path, \
                            word_hashcode_path=WORD_HASHCODE_PATH, encoding=encoding)

    print "executing queries"
    result_path = "result/zhidao_1000"
    execute_cpu(corpus_hashcode_path, result_path, encoding=encoding)
    #execute_gpu(corpus_hashcode_path, result_path, encoding=encoding)
    print "finished"

if __name__ == "__main__":
    example()
