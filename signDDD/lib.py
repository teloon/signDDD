#!/usr/bin/env python
#-*- coding:utf8 -*-

"""This module contains some common functions that
are used in other mudules.

"""

import sys

def pad_bin_code(code_str, code_width):
    """Convert a code string(int string) into a fixed-with binary string,
    padding with 0s.

    """
    return (code_width - len(bin(code_str)) + 2) * "0" + bin(code_str)[2 : ]

def show_progress(cur, total):
    """Show the current process in the terminal.

    """
    show_cur = 100 * cur / total
    print "[%s%s] %d %.2f%%" % (">" * show_cur, " " * (100 - show_cur), cur, 100.0 * cur / total), "\r",
    sys.stdout.flush()

def load_corpus_hashcode(codes_path):
    """Load the hashcodes of all the sentences in the corpus from a given path.

    """
    src_f = open(codes_path, "r")
    corpus_hashcodes = []
    for ln in src_f:
        corpus_hashcodes.append(int(ln.strip()))
    return corpus_hashcodes

def cnt_ln(file_path):
    """Count the line number of the given file

    """
    cnt = 0
    for ln in open(file_path):
        cnt += 1
    return cnt
