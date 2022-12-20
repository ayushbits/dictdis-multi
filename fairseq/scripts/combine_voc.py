from __future__ import print_function

import os
import sys
import inspect
import warnings
import argparse
import codecs

from collections import Counter

def clean_vocab(in_vocab_fname1, in_vocab_fname2, out_vocab_fname):
    c = Counter()
    with codecs.open(in_vocab_fname1, "r", encoding="utf-8") as infile1, codecs.open(in_vocab_fname2, "r", encoding="utf-8") as infile2, codecs.open(
        out_vocab_fname, "w", encoding="utf-8"
    ) as outfile:
        for i, line in enumerate(infile1):
            fields = line.strip("\r\n ").split(" ")
            if len(fields) == 2:
                c[fields[0]] += int(fields[1])
        for i, line in enumerate(infile2):
            fields = line.strip("\r\n ").split(" ")
            if len(fields) == 2:
                c[fields[0]] += int(fields[1])        
            if len(fields) != 2:
                print("{}: {}".format(i, line.strip()))
                for c in line:
                    print("{}:{}".format(c, hex(ord(c))))
        for key,f in sorted(c.items(), key=lambda x: x[1], reverse=True):
            outfile.write(key+" "+ str(f) + "\n")


if __name__ == "__main__":
    clean_vocab(sys.argv[1], sys.argv[2],  sys.argv[3])