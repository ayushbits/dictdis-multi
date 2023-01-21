import sys
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict
 

dirname = sys.argv[1]
app = ['rbi-pred-1','rbi-pred-0', 'rbi-bi-indic'] # For rbi
# app = ['bob-pred-1','bob-pred-0', 'bob-bi-indic'] # For bob
# app = ['engg-pred1','engg-pred0', 'engg-indic-bi'] # For engg

dicdis = dirname.lower()
# if dirname == 'air-space':
#     dicdis = dirname.replace('-','')
# dicdis += '-leca96ep28'

# app.append(dicdis)
alldict = {}
for a in app:
    path = dirname + '/' + a + "-spdi.txt" # | sed -n 15p | cut -d'{' -f2 | cut -d'}' -f1 "
    with open(path, 'r') as f :
        spdi = f.readlines()[-1]
        line = spdi.split('): ')[1][1:-2] # Model Translation Vs GroundTruth i.e (model_trans hit/GroundTruth hit): {1: 0.9938556067588326, 2: 0.6842105263157895, 4: 1.0, 3: 1e-10}
        # print('SPDI ', spdi)
        vals = line.split(',')
        gtdict = {}
        i=0
        for v in vals:
            i+= 1
            # if i > 5:
            #     break
            key = v.split(':')[0].strip()
            val = v.split(':')[1].strip()
            gtdict[key] = val
        # disambiguation accuracy =  (\sum_i i*(precision of all items falling in ith bucket)) / (sum of i's)
        # where i is number of constraints
        # {1: 0.9938556067588326, 2: 0.6842105263157895, 4: 1.0, 3: 1e-10}
        deno  = (np.sum(list(map(int,gtdict.keys()))))
        
        numerator = np.sum([float(k)*float(gtdict[k]) for k in gtdict.keys()])
        # print('num %s , deno %s ', numerator, deno)
        disacc = numerator/deno
        
        print('Micro disambiguation accuracy for ', a , ' is ', disacc)
        # print('Macro disambiguation accuracy for ', a , ' is ', macro_disacc)
        gtdict = sorted(gtdict.items(), key=lambda x: int(x[0]))
        alldict[a] = gtdict
        

labels = {}
for a in app:
    if 'google' in a:
        labels[a] = 'Google Translate'
    elif 'bart' in a:
        labels[a] = 'mBART-50'
    elif 'original' in a:
        labels[a] = 'Leca'
    elif 'ep28' in a:
        labels[a] = 'DictDis'
    elif 'indic' in a:
        labels[a] = 'IndicTrans'
    
for d in alldict.keys():
    spdi_gt_list = alldict[d]
    # spdi_gt_list = spdi_gt_score.items()
    gt_x, gt_y = zip(*spdi_gt_list)
    # print(gt_y)
    list_y = list(gt_y)
    # print('list y is ', list_y)
    newy = [np.round(float(y)*100,2) for y in gt_y]

    # label = d
    plt.plot(gt_x[0:7], newy[0:7],label=labels[d])
    # plt.plot(pred_gt_x, pred_gt_y,label='CSR')
    if dirname =='aerospace':
        plt.legend(loc='upper right')
    else:
        plt.legend()
    plt.xlabel('Polysemous Degree')
    plt.ylabel('Precision')
    # plt.title(dirname+': SPDI')
    # plt.savefig(dirname+'-spdi.jpg')
    plt.savefig(dirname+'-spdi.pdf', format="pdf", bbox_inches="tight")



# spdi_pred_list = spdi_pred_score.items()
# pred_x, pred_y = zip(*spdi_pred_list)

# spdi_pred_gt_list = spdi_pred_gt_score.items()
# pred_gt_x, pred_gt_y = zip(*spdi_pred_gt_list)

# # plt.plot(gt_x, gt_y,label=' SPDI')
# plt.plot(pred_x, pred_y,label='Model SPDI')
# plt.plot(pred_gt_x, pred_gt_y,label='CSR')
# plt.legend()
# plt.xlabel('No. of Candidate Translations')
# plt.ylabel('Accuracy')
# plt.title(outfile+': SPDI')
# plt.savefig(outfile+'-spdi.jpg')
