#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
from collections import defaultdict
import pdb
import pdb

class preprocess(object):

    def __init__(self, paracorp_path, monocorp_path):
        self.paracorp_path = paracorp_path
        self.monocorp_path = monocorp_path
        self.vocab_hindi = defaultdict(int)
        self.vocab_eng = defaultdict(int)

    def gen_vocab(self):
        # Reads the parallel corpus file and generates the vocabulary for
        # both languages
        eng_tokenidx = 0
        hind_tokenidx = 0
        f = open(self.paracorp_path)
        count = 0
        for line in f:
            src_id, align_type, align_quality, eng_seg, hind_seg = line.split("\t")
            #data = line.split()
            #pdb.set_trace()
            #hind_seg = hind_seg.encode("ascii", "replace")
            #eng_seg = eng_seg.encode("ascii", "replace")
            #pdb.set_trace()
            # handle cases with two segments in one paragraph
            if eng_seg.find("<s>") == -1:
                eng_tokens = eng_seg.lower().split()
                for token in eng_tokens:
                    if token not in self.vocab_eng:
                        self.vocab_eng[token] = eng_tokenidx
                        eng_tokenidx += 1

                hind_tokens = hind_seg.lower().split()
                for token in hind_tokens:
                    if token not in self.vocab_hindi:
                        self.vocab_hindi[token] = hind_tokenidx
                        hind_tokenidx += 1

            elif eng_seg.find("<s>") != -1:
                 eng_seg.split("<s>")
        print count

if __name__ == "__main__":
    preproces = preprocess("/home/shashank/datasets/MT-dataset/hindencorp05.plaintext",
                           "/home/shashank/datasets/MT-dataset/hin_corp_unicode")
    preproces.gen_vocab()


