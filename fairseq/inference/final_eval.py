# python final_eval.py enfr/pred-1.fr enfr/test.fr enfr/test.en  ../data-en-fr/lexicalDict/ en-fr-final
import pandas as pd
import re
import csv
import sys
# import matplotlib.pyplot as plt
from tqdm import tqdm
from flashtext import KeywordProcessor

default_missing = pd._libs.parsers.STR_NA_VALUES
default_missing = default_missing.remove('null')

def create_dictionary(glossPath, glossaries):
    glossaries = glossaries.split(',')
    print(glossaries)
    dictionaries = {}
    for glossary in glossaries:
        df = pd.read_csv(glossPath+'/'+glossary+'.csv', na_values=default_missing)
        headings = list(df.keys())
        for j in range(len(df)):
            src_phrase = df.loc[j,'English']
            # print(j,src_phrase)
            src_phrase = src_phrase.lower()
            src_phrase = re.sub(r"\(\'\"[^()]*\)", "", src_phrase).strip()
            tgt_phrase = (str)(df.loc[j,'Hindi'])
            tgt_phrase = tgt_phrase.split(',')
            #cleaning dict entries
            tgt_phrase=[meaning.strip('[]') for meaning in tgt_phrase]
            tgt_phrase=[meaning.strip("'") for meaning in tgt_phrase]
            tgt_phrase=[re.sub("\(.*?\)","",meaning) for meaning in tgt_phrase]
            tgt_phrase=[re.sub(r'[0-9]+.', ',', meaning) for meaning in tgt_phrase] 
            tgt_phrase=[meaning.strip() for meaning in tgt_phrase]
            # tgt_phrase = [re.sub(r"\(\'\"[^()]*\)", "", meaning).strip() for meaning in tgt_phrase]
            tgt_phrase = ','.join(tgt for tgt in tgt_phrase)
            if(src_phrase not in dictionaries):
                dictionaries[src_phrase] = tgt_phrase
            else:
                dictionaries[src_phrase] += ', ' + tgt_phrase
    return dictionaries

# def Disambiguation_SPDI(dictionaries, parallelFile, srcCol, tgtCol):
def Disambiguation_SPDI(dictionaries, src_arr, pred_arr, gt_arr, outfile):
    # df = pd.read_csv(parallelFile + '.csv', sep=',')

    kp = KeywordProcessor()
    for key in dictionaries.keys():
        kp.add_keyword(key, (key, dictionaries[key]))
        

    total, pred_hit, gt_hit, sent_count = 0,0,0,0
    spdi_gt, spdi_pred, spdi_tot = {}, {}, {}
    # for i in range(1,10):
    #     spdi_tot[i] = 0
    #     spdi_gt[i] = 0
    #     spdi_pred[i] = 0
    rows = []

    # for ind, line in tqdm(df.iterrows(), total=df.shape[0]):
    for ind, (src, pred, gt) in tqdm(enumerate(zip(src_arr, pred_arr, gt_arr)), total=len(src_arr)):
        # print('line',type(ind),line[srcCol])
        # src = line[srcCol].lower()
        print_src = src
        print_gt = gt
        src = src.lower()
        pred = pred.lower()
        gt = gt.lower()
        src = re.sub(r"\(\'\"[^()]*\)", "", src).strip()
        matchings = kp.extract_keywords(src) # matched with source side ie English
        # print('matchings are ', len(matchings))
        if(len(matchings)>0):
            sent_count += 1
            gt_cons = []
            pred_cons = []
            for matches in matchings:
                gt_cons_trans = "xxx"
                pred_cons_trans = "xxx"
                total += 1
                pred_found = False
                gt_found = False
                cons_cand_trans = matches[1].split(',') # Change this to ';' for enfr and ende and keep ',' for enhi
                cons = list(set([con for con in [con.strip().lower() for con in cons_cand_trans] if con]))
                if(len(cons) in spdi_tot.keys()):
                    spdi_tot[len(cons)] +=1
                else:
                    spdi_tot[len(cons)] =1
                # print('spdi_tot ', spdi_tot)
                # print('Cons is ', cons)
                for tgt_phrase in cons:
                    # if(tgt_phrase in line[tgtCol] and tgt_phrase in line['groundTruth']):
                    # print('pred ', pred)
                    # print('gt ', gt)
                    if(tgt_phrase in pred and tgt_phrase in gt):
                        pred_found = True
                        pred_hit += 1
                        pred_cons_trans = tgt_phrase
                        if(len(cons) in spdi_pred.keys()):
                            spdi_pred[len(cons)] += 1
                        else:
                            spdi_pred[len(cons)] = 1
                        break
                    elif tgt_phrase not in pred and tgt_phrase in gt:
                        incorrect_pred = tgt_phrase
                        # print('not ingested constraint ', matchings, pred , gt,src)
                        # print("\n")
                        # print(print_gt)
                        # print('sent count ', sent_count)

                    # if(tgt_phrase in gt):
                    #     gt_hit += 1
                    #     gt_found = True
                    #     gt_cons_trans = tgt_phrase
                    #     if(len(cons) in spdi_gt.keys()):
                    #         spdi_gt[len(cons)] += 1
                    #     else:
                    #         spdi_gt[len(cons)] = 1
                    
                # if ind > 1380 and ind < 1390:
                #     print('Above ', ind, cons, gt)
                for tgt_phrase in cons:
                    # if(tgt_phrase in line['groundTruth']):
                    # if ind > 1380 and ind < 1390:
                    #         print(ind, tgt_phrase, gt)
                    if(tgt_phrase in gt):
                        # if ind > 1380 and ind < 1390:
                        #     print(ind, cons, gt)
                        gt_hit += 1
                        gt_found = True
                        gt_cons_trans = tgt_phrase
                        if(len(cons) in spdi_gt.keys()):
                            spdi_gt[len(cons)] += 1
                        else:
                            spdi_gt[len(cons)] = 1
                        break

                gt_cons.append(gt_cons_trans)
                pred_cons.append(pred_cons_trans)

            # row = [ind, line[srcCol], line[tgtCol], line['groundTruth'], matchings, pred_cons, gt_cons]
            # row = [ind, src, pred, gt, matchings, pred_cons, gt_cons]
            row = [ind, matchings, pred_cons, gt_cons]
            rows.append(row)
    
    # fields = ['Index, Source, Predicted, GroundTruth, Constraint_Matches, Predicted_Cons_Trans, GroundTruth_Cons_Trans']
    fields = ['Index, Constraint_Matches, Predicted_Cons_Trans, GroundTruth_Cons_Trans']
    output_filename = outfile + '_eval.csv'
    with open(output_filename, 'w') as csvFile:
        csvwriter = csv.writer(csvFile, delimiter ='\n')
        csvwriter.writerow(fields)
        csvwriter.writerow(rows)

    spdi_gt_score, spdi_pred_score, spdi_pred_gt_score = {}, {}, {}
    # print(spdi_tot.keys())
    # print(spdi_pred.keys())
    # print(spdi_gt.keys())
    for i in spdi_tot.keys():
        if(i not in spdi_pred.keys()):
            spdi_pred[i] = 1e-10
        if(i not in spdi_gt.keys()):
            spdi_gt[i] = 1e-10
        if(spdi_tot[i]>0):
            spdi_gt_score[i] = float(spdi_gt[i]/spdi_tot[i])
            spdi_pred_score[i] = float(spdi_pred[i]/spdi_tot[i])
            spdi_pred_gt_score[i] = float(spdi_pred[i]/spdi_gt[i])
    print('SPDI Total',spdi_tot)
    print('SPDI Model_Trans', spdi_pred)
    print('SPDI GroundTruth', spdi_gt)

    # spdi_gt_list = spdi_gt_score.items()
    # gt_x, gt_y = zip(*spdi_gt_list)

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

    with open(outfile+'-spdi.txt', 'w') as cm:
        cm.write('######################### Disambigaution Accuracy #########################)\n')
        cm.write('Total Hits '+ str(pred_hit)+ '(model_trans hit) out of '+ str(gt_hit)+ '(groundTruth hit)\n')
        cm.write('Disamb Acc wrt model_trans & groundTruth aka Copy Success Rate: '+ str(pred_hit*1.0/gt_hit)+'\n')
        cm.write('Disamb Acc wrt source & groundTruth i.e. (groundTruth hit/source side hit): '+ str(gt_hit*1.0/total)+'\n')
        cm.write('Disamb Acc wrt source & model_trans i.e. (model_trans hit/source side hit) : '+ str(pred_hit*1.0/total)+'\n')
        cm.write('Total Sentences having dictionary words: '+ str(len(src)) + ' out of '+ str(sent_count)+'\n')
        cm.write('No of times dictionary word doesnt occur in groundTruth :'+ str(total - gt_hit)+'\n')

        cm.write('######################### Sense Polysemy Degree Importance #########################)\n')

        cm.write('SPDI Total' +  str(spdi_tot)+'\n')
        cm.write('SPDI Model_Trans'+  str(spdi_pred) + '\n')
        cm.write('SPDI GroundTruth'+  str(spdi_gt) + '\n')

        cm.write('Total SPDI'+ str(spdi_tot)+'\n')
        cm.write('Model Translation SPDI i.e. (model_trans hit/Total SPDI): '+ str(spdi_pred_score)+'\n')
        cm.write('GroundTruth SPDI i.e. (GroundTruth hit/Total SPDI): '+ str(spdi_gt_score)+'\n')
        cm.write('Model Translation Vs GroundTruth i.e (model_trans hit/GroundTruth hit): '+ str(spdi_pred_gt_score)+'\n')

if __name__ == "__main__":
    
    tgtfile = sys.argv[1]
    gtfile = sys.argv[2]
    srcfile = sys.argv[3]
    glossPath = '../full_data/lexicalDict' 
    glossPath = sys.argv[4]
    glossary = sys.argv[5]
    # directory_storage = sys.argv[6]
    # srcCol = sys.argv[6]
    # tgtCol = sys.argv[7]

    outfile = tgtfile[:tgtfile.rfind('.')]
    with open(srcfile,'r') as src, open(tgtfile,'r') as tgt, open(gtfile, 'r') as gt:
        srcSents = src.readlines()
        tgtSents = tgt.readlines()
        gtSents = gt.readlines()
        srcTgt = []
        # for i in range(len(srcSents)):
        #     srcTgt.append([srcSents[i][:-1], tgtSents[i][:-1], gtSents[i][:-1]])

    # pd.DataFrame(srcTgt, columns=['Source', 'Predicted', 'GroundTruth']).to_csv(parallelFile + '.csv', index=None)
    dictionaries = create_dictionary(glossPath, glossary)
    # Disambiguation_SPDI(dictionaries, parallelFile, 'Source', 'Predicted')
    Disambiguation_SPDI(dictionaries, srcSents, tgtSents, gtSents, outfile)




