import sys
import numpy as np
import ast

infile = sys.argv[1]
infile2 = sys.argv[2]

outfile = sys.argv[3]

inf, inf2, gt = [], [], []
i = 0 
with open(infile, 'r') as f:
    f.readline()
    for a in f.readlines():
        i+=1
        arr =  ast.literal_eval(a)
        inf.append(arr[2])
        gt.append(arr[3])

with open(infile2, 'r') as f:
    f.readline()
    for a in f.readlines():
        i+=1
        arr =  ast.literal_eval(a)
        inf2.append(arr[2])#, arr[3])
ind = 0
for ip, ip2, gtp in zip(inf, inf2, gt):
    # print(ip, ip2, gtp)
    ind+=1
    i = ip # .split(",") #ast.literal_eval(ip)
    i2 = ip2 # .split(",") #ast.literal_eval(i2p)
    gt = gtp # .split(",") # ast.literal_eval(gtp)
    if i != i2 and i ==gt and i2!=gt:
        print('Index is ', ind,  'Correct f1 ', i , i2, gt)
    elif i!= i2  and i2==gt:
        print('Index is ', ind, 'Incorrect filename 1 ', i, i2, gt)

        
        
        