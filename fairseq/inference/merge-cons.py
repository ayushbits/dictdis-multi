import sys
dirs = sys.argv[1]

enfile = "eval_final/" + "/" + dirs +"/" + dirs + "-dict-constraints.en"
consfile = "eval_final/" + "/" + dirs +"/" + dirs +  "-dict-constraints.en.constraints"

en, cons = [], []
with open(enfile, 'r') as f:
    for i in f.readlines():
        en.append(i)

with open(consfile, 'r') as f:
    for i in f.readlines():
        cons.append(i)
cons_full_file = "eval_final/" + dirs + "/" + dirs +"-cons.en"
cons_full = open(cons_full_file, 'w')
for e,c in zip(en, cons):
    push_to = e.replace("\n","") + c.replace("<sep>","\\t").strip().replace("<isep>","\\t").strip().lower() + "\n"
    # print('string ', push_to)
    # break
    cons_full.write(push_to)
