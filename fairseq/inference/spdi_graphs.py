import sys
import os
import glob
import ast
import matplotlib.pyplot as plt

dir_name = sys.argv[1]

spdi_file = glob.glob(dir_name+"/*spdi.txt")
# print(spdi_file)
graphdict = {}
for file in spdi_file:
    with open(file, 'r') as f:
        last_line = f.readlines()[-1]
    pos = 'hit): '
    spdi = last_line[last_line.index(pos)+6:]
    fname = file[file.index('/')+1:file.index('-spdi')]
    graphdict[fname] = ast.literal_eval(spdi)

print(graphdict)
    
for name in graphdict.keys():
    spdi_list = sorted(graphdict[name].items())
    # print(name)
    # print(spdi_list)
    
    pred_x, pred_y = zip(*spdi_list)
    plt.plot(pred_x[1:], pred_y[1:],label=name)

plt.legend()
plt.xlabel('No. of Candidate Translations')
plt.ylabel('Accuracy')
plt.title('SPDI')
plt.savefig(dir_name +'/spdi.jpg')

# spdi_gt_list = spdi_gt_score.items()
# gt_x, gt_y = zip(*spdi_gt_list)

# spdi_pred_list = spdi_pred_score.items()
# pred_x, pred_y = zip(*spdi_pred_list)

# spdi_pred_gt_list = spdi_pred_gt_score.items()
# pred_gt_x, pred_gt_y = zip(*spdi_pred_gt_list)

# # plt.plot(gt_x, gt_y,label=' SPDI')
# plt.plot(pred_x, pred_y,label='Model SPDI')
# plt.plot(pred_gt_x, pred_gt_y,label='CSR')
