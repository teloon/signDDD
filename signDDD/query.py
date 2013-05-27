#!/usr/bin/env python
#-*- coding:utf8 -*-

"""This module implements the core process of querying near duplicate documents
against an existing corpus. Given **a list of queris** and **corpus hashcode file**,
this module will output all the duplicates in some specified file.

In the module there're two ways of querying(:func:`signDDD.query.execute_cpu` and :func:`signDDD.query.execute_gpu`), whose
main difference lies in how to compute the hamming distance of two hashcodes. But
the main algorithms are the same:

1. **XOR** two hashcodes to index the bits that are different
2. **Count** the number of bit difference

Generally, the GPU-based version achieves about 1500X speedup in our GPU and CPU platform.

Example usage for CPU-based querying:

.. code-block:: python
   :linenos:

   query_lst = [u"谁的歌最好听？", u"陈奕迅的富士山下"]
   corpus_path, result_path = "corpus/zhidao_1000", "result/zhidao_1000"
   corpus_hashcode_path = corpus_path + "_hashcode"
   gen_corpus_hashcode(corpus_path, corpus_hashcode_path, \\
                            word_hashcode_path=WORD_HASHCODE_PATH, encoding="utf8")
   execute_cpu(query_lst, corpus_hashcode_path, result_path, encoding="utf8")

"""

import os
from hashcode import gen_corpus_hashcode
from glo import CODE_WIDTH
from glo import BIT_DIFF_THD
from glo import INT_WIDTH
from glo import WORD_HASHCODE_PATH
from glo import CUDA32_PATH
from glo import CUDA64_PATH
from lib import pad_bin_code
from lib import cnt_ln
from lib import load_corpus_hashcode
from lib import show_progress

def execute_gpu(query_lst, corpus_hashcode_path, output_path, encoding="utf8"):
    """This function implements the GPU-based version of querying.

    .. caution::

       - The second parameter is the path of the corpus **hashcode** file. So if not
         existing, the hashcode file needs to be generated in advance by calling
         :func:`signDDD.hashcode.gen_corpus_hashcode` function.
       - This method will generate a temporary query file in directory *query*, which
         will be autumatically generated if not existing.

    """
    print "executing queries by GPU"
    if not os.path.exists("query"):
        os.mkdir("query")
    query_path = "query/query"
    with open(query_path, "w") as out_f:
        out_f.write("\n".join([q.encode(encoding) for q in query_lst]))
    query_hashcode_path = "query/query_hashcode"
    gen_corpus_hashcode(query_path, query_hashcode_path, \
                            word_hashcode_path=WORD_HASHCODE_PATH, encoding="utf8")
    gpu_exe_fn = CUDA32_PATH if CODE_WIDTH==32 else CUDA64_PATH
    cmd = "./%s %s %s %s %d %d %d" % (gpu_exe_fn, corpus_hashcode_path, query_hashcode_path, \
                                        output_path, cnt_ln(corpus_hashcode_path), \
                                                    cnt_ln(query_path), BIT_DIFF_THD)
    #print cmd
    ret = os.popen(cmd).read().strip()
    #print ret

def execute_cpu(query_lst, corpus_hashcode_path, output_path, encoding="utf8"):
    """This function implements the CPU-based version of querying. However, the
       GPU-based version is much more efficient. So whenever runnable, the GPU-based
       version is often preferable.

     .. caution::

       - The second parameter is the path of the corpus **hashcode** file. So if not
         existing, the hashcode file needs to be generated in advance by calling
         :func:`signDDD.hashcode.gen_corpus_hashcode` function.
       - This method will generate a temporary query file in directory *query*, which
         will be autumatically generated if not existing.

    """
    print "executing queries by CPU"
    if not os.path.exists("query"):
        os.mkdir("query")
    query_path = "query/query"
    with open(query_path, "w") as out_f:
        out_f.write("\n".join([q.encode(encoding) for q in query_lst]))
    query_hashcode_path = "query/query_hashcode"
    gen_corpus_hashcode(query_path, query_hashcode_path, \
                            word_hashcode_path=WORD_HASHCODE_PATH, encoding="utf8")
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
    out_f = open(output_path, "w")
    for q_idx,dups in enumerate(result):
        out_f.write("%s\n" % "\t".join([str(idx) for idx in dups]))
    out_f.close()

def example():
    query_lst = [u"谁的歌最好听？", u"陈奕迅的富士山下"]
    corpus_path = "corpus/zhidao_1000"
    corpus_hashcode_path = corpus_path + "_hashcode"
    print "generating corpus hashcode"
    gen_corpus_hashcode(corpus_path, corpus_hashcode_path, \
                            word_hashcode_path=WORD_HASHCODE_PATH)

    result_path = "result/zhidao_1000"
    #execute_cpu(query_lst, corpus_hashcode_path, result_path)
    execute_gpu(query_lst, corpus_hashcode_path, result_path)
    print "finish"

if __name__ == "__main__":
    example()
