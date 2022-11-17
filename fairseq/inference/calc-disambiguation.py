import sys
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict
 
checks = [2,5,10,15,20,25, 28]
# checks = [24, 28 ] 
for c in checks:
    path = 'results-airspace-epwise-leca96ep' + str(c) + '.txt'
    with open(path, 'r') as f :
        spdi = f.readlines()[-1]
        vals = spdi.split(',')
        gtdict = {}
        i=0
        for v in vals:
            i+= 1
            key = v.split(':')[0].strip()
            val = v.split(':')[1].strip()
            gtdict[key] = val
            
            
        
        # disambiguation accuracy =  (\sum_i i*(precision of all items falling in ith bucket)) / (sum of i's)
        # where i is number of constraints
        # {1: 0.9938556067588326, 2: 0.6842105263157895, 4: 1.0, 3: 1e-10}
        deno  = (np.sum(list(map(int,gtdict.keys()))))
        
        numerator = np.sum([np.log2(float(k)+1)*float(gtdict[k]) for k in gtdict.keys()])
        # print('num %s , deno %s ', numerator, deno)
        disacc = numerator/deno
        # print('gt dict ', gtdict)
        # nume = 0
        # for k in gtdict.keys():
        #     # print(gtdict[k]
        #     nume += float(k)*float(gtdict[k])
        # print('nume ', nume)
        # macro_disacc = np.sum([float(k)*float(gtdict[k]) for k in gtdict.keys()])/len(list(map(int,gtdict.keys())))
        print('Micro disambiguation accuracy for ', c , ' is ', disacc)
        # print('Macro disambiguation accuracy for ', a , ' is ', macro_disacc)

        