import sys
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict
 

# dirname = sys.argv[1]
app = ['2','5','10','15','20','24','25', '28']
# dicdis = dirname
# if dirname == 'air-space':
#     dicdis = dirname.replace('-','')
dicdis = 'results-training-data-leca96ep'

app.append(dicdis)
alldict = {}
for a in app:
    path = dicdis  + a + ".txt" # | sed -n 15p | cut -d'{' -f2 | cut -d'}' -f1 "
    with open(path, 'r') as f :
        spdi = f.readlines()[0]
        # line = spdi.split('): ')[1][1:-2] # Model Translation Vs GroundTruth i.e (model_trans hit/GroundTruth hit): {1: 0.9938556067588326, 2: 0.6842105263157895, 4: 1.0, 3: 1e-10}
        
        gtdict = {}
        lines = spdi.split(',')
        for idx, v in enumerate(lines):
            if idx > 1:
                # for v in vals:
                # print(v)
                key = v.split(':')[0].strip()
                val = v.split(':')[1].strip()
                gtdict[key] = val
            
        # disambiguation accuracy =  (\sum_i i*(precision of all items falling in ith bucket)) / (sum of i's)
        # where i is number of constraints
        # {1: 0.9938556067588326, 2: 0.6842105263157895, 4: 1.0, 3: 1e-10}
        disacc = np.sum([(float(k))*float(gtdict[k]) for k in gtdict.keys()])/np.sum(list(map(int,gtdict.keys())))


        # micro_disacc = np.sum([float(k)*float(gtdict[k]) for k in gtdict.keys()])/len(list(map(int,gtdict.keys())))
        print('Micro disambiguation accuracy for Epoch  ', a , ' is ', disacc)
        # print('Macro disambiguation accuracy for Epoch ', a , ' is ', micro_disacc)
        # gtdict = sorted(gtdict.items(), key=lambda x: int(x[0]))
        # alldict[a] = gtdict
        