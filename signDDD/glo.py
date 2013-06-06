#!/usr/bin/env python
#-*- coding:utf8 -*-

"""This module defines some gobal const values.
Change these value according to the environment or need.

"""

import sys

PUNCS_PATH = "res/puncs"
STOPWORDS_PATH = "res/stopwords"

#width of the hashcode
CODE_WIDTH = 32

#number of bits that are set in each hashcode
BIT_SET = 2

#threshold of duplicate
BIT_DIFF_THD = 3 #not included

#file path of some words' hashcodes that are optimized
#if not provided, it will use the default MD5 to get the hashcode
WORD_HASHCODE_PATH = "res/words_cn"

#path of CUDA program
CUDA32_PATH = "cuda/cuda_match.32"
CUDA64_PATH = "cuda/cuda_match.64"

#auto detect
INT_WIDTH = 32 if sys.maxint < 1<<31 else 64
