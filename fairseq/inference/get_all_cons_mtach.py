''' The scripts returns two files containing sentences which does not include constraints and sentences only including constraints'''

import sys

path = sys.argv[1]

enfile = path +"/"+path+".en"
hifile = path +"/"+path+".hi"
en_constraints_file = path +"/"+path+".en.constraints"

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
    # else:
    #     dict_constraints_en.append(e)
    #     dict_constraints_hi.append(h)

write_path ="eval_final"
filename = write_path +"/" +path +"/"+path+"-all-match-constraints.en"
with open(filename, 'w') as f:
    for i in wo_constraints_en:
        f.write(i)

filename = write_path +"/" +path +"/"+path+"-all-match-constraints.hi"
with open(filename, 'w') as f:
    for i in wo_constraints_hi:
        f.write(i)

# filename = write_path +"/" +path +"/"+path+"-dict-constraints.en"
# with open(filename, 'w') as f:
#     for i in dict_constraints_en:
#         f.write(i)

# filename = write_path +"/" +path +"/"+path+"-dict-constraints.hi"
# with open(filename, 'w') as f:
#     for i in dict_constraints_hi:
#         f.write(i)
