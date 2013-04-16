#!/usr/bin/env python
#-*- coding:utf8 -*-

import itertools, hashlib, os
from extracted_tknz import Tokenizer
from glo import PUNCS_PATH
from glo import STOPWORDS_PATH
from glo import CODE_WIDTH
from glo import BIT_SET
from glo import INT_WIDTH
from lib import pad_bin_code

DEBUG_BINARY_INFO = 0

def gen_corpus_hashcode(corpus_path, out_path, word_hashcode_path="", encoding="utf8"):
    out_f = open(out_path, "w")
    word_codes_dict = {}
    puncs = load_punc()
    stopwords = load_stopwords()
    rand_code_dict = gen_rand_code_dict()
    if word_hashcode_path:
        word_codes_dict = load_word_hashcode(word_hashcode_path, encoding=encoding)
    for ln in open(corpus_path):
        txt = ln.strip().decode(encoding)
        hash_code = do_gen_sent_hashcode(txt, word_codes_dict, rand_code_dict, puncs, stopwords)
        out_f.write("%d\n" % hash_code)
#        print pad_bin_code(hash_code,32)
    out_f.close()
    return

def do_gen_sent_hashcode(txt, word_codes_dict, rand_code_dict, puncs, stopwords):
    """naive approach of hashing text: original "signiture-indexing"

    Args:
        txt: text needed to be hashed
        CODE_WIDTH: hashcode width, <=64
        BIT_SET: number of bits set in hashcode
        use_optmz: flag indicating whether to use optimized words hashcode

    Returns:
        hash_val: an integer, corresponding to the hash code of txt(binary), e.g.: 5(000000000000101)

    """
    #generate txt's hash code
    tknzr = Tokenizer()
    tk_list = tknzr.tokenize(txt)
    #if the "txt" is already tokenized
    #tk_list = txt.split("\t")
    hash_val = 0L
    CODE_LEN = len(rand_code_dict)
#    print tk_list
    for tk in tk_list:
        if tk in puncs:
            continue
        if tk in stopwords:
#            print "---find stopwords---"
            continue
        tk_code = 0
#        print tk,"#",tk.encode("utf8")
        if word_codes_dict and word_codes_dict.has_key(tk):
            tk_code = word_codes_dict[tk];
        else:
            tk = tk.encode("utf8")
            md5 = hashlib.md5()
            md5.update(tk)
            if CODE_WIDTH <= INT_WIDTH:
                int_md5 = int(md5.hexdigest(), 16)
            else:
                int_md5 = long(md5.hexdigest(), 16)
            tk_code = rand_code_dict[int_md5 % CODE_LEN]
        if DEBUG_BINARY_INFO:
            print pad_bin_code(hash_val, CODE_WIDTH)
        hash_val = hash_val | tk_code
        if DEBUG_BINARY_INFO:
            print pad_bin_code(tk_code,CODE_WIDTH)
            print pad_bin_code(hash_val,CODE_WIDTH)
            print "-"*50
    return hash_val



def load_word_hashcode(word_hashcode_path, encoding="utf8"):
    word_codes_dict = {}
    for ln in open(word_hashcode_path):
        ln = ln.decode(encoding)
        word, code = ln.strip().split()
        word_codes_dict[word] = int(code)
    return word_codes_dict

def load_punc():
    if not os.path.exists(PUNCS_PATH):
        return []
    puncs = set(open(PUNCS_PATH, "r").readline().strip().decode("utf8"))
    #print puncs, len(puncs)
    return puncs

def load_stopwords():
    stopwords = []
    if not os.path.exists(STOPWORDS_PATH):
        return []
    for ln in open(STOPWORDS_PATH):
        ln = ln.strip().decode("utf8")
        stopwords.append(ln.split()[0])
#    print "stopwords:",stopwords
    return stopwords

def gen_rand_code_dict():
    #initialize code_dict
    code_dict = []
    for comb in itertools.combinations(range(CODE_WIDTH), BIT_SET):
        hash_code = 0L
        for pos in comb:
            hash_code = hash_code | (1L<<pos)
#        print comb,bin(hash_code)
        code_dict.append(hash_code)
    return code_dict 

