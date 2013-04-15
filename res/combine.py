#!/usr/bin/env python
#-*- coding:utf8 -*-

def combine(code_path, word_path, out_path):
    with open(out_path, "w") as out_f:
        with open(code_path) as code_f:
            with open(word_path) as word_f:
                for ln in code_f:
                    out_f.write("%s\t%d\n" % (word_f.readline().strip(), \
                                                int("".join(ln.strip().split()), 2)))

if __name__ == "__main__":
    combine("zhidao_0.3_hashcode_overlap.txt", "zhidao_0.3_words.txt", "words_cn")

