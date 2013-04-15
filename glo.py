#!/usr/bin/env python
#-*- coding:utf8 -*-

import sys

PUNCS_PATH = "res/puncs"
STOPWORDS_PATH = "res/stopwords"
CODE_WIDTH = 32
BIT_SET = 2
INT_WIDTH = 32 if sys.maxint < 1<<31 else 64 
BIT_DIFF_THD = 3 #not included
WORD_HASHCODE_PATH = "res/words_cn"

