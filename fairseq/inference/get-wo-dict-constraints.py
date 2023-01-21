''' The scripts returns two files containing sentences which does not include constraints and sentences only including constraints'''

import sys

path = sys.argv[1]

enfile = path +"/"+"wmt14-dict-constraints"+".en"
hifile = path +"/"+"wmt14-dict-constraints"+".de"
en_constraints_file = path +"/"+"wmt14-dict-constraints"+".en.constraints"

en_constraints, en, hi =  [],[],[]

wo_constraints_en, dict_constraints_en = [], []
wo_constraints_hi, dict_constraints_hi = [], []

with open(en_constraints_file,'r') as f:
    for i in f.readlines():
        en_constraints.append(i)

with open(enfile,'r') as f:
    for i in f.readlines():
        en.append(i)

with open(hifile,'r') as f:
    for i in f.readlines():
        hi.append(i)

for e,h,c in zip(en, hi, en_constraints):
    if c.strip() =='':
        wo_constraints_en.append(e)
        wo_constraints_hi.append(h)
    else:
        dict_constraints_en.append(e)
        dict_constraints_hi.append(h)

write_path ="eval_final"
path="wmt14"
filename = write_path +"/" +path +"/"+"wmt14"+"-wo-constraints1.en"
with open(filename, 'w') as f:
    for i in wo_constraints_en:
        f.write(i)

filename = write_path +"/" +path +"/"+"wmt14"+"-wo-constraints1.de"
with open(filename, 'w') as f:
    for i in wo_constraints_hi:
        f.write(i)

filename = write_path +"/" +path +"/"+"wmt14"+"-dict-constraints1.en"
with open(filename, 'w') as f:
    for i in dict_constraints_en:
        f.write(i)

filename = write_path +"/" +path +"/"+"wmt14"+"-dict-constraints1.de"
with open(filename, 'w') as f:
    for i in dict_constraints_hi:
        f.write(i)
