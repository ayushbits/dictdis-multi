import sys
from tqdm import tqdm 
import re
import mmap
import pandas as pd
from flashtext import KeywordProcessor

default_missing = pd._libs.parsers.STR_NA_VALUES
default_missing = default_missing.remove('null')


def numberOfLines(inputFile):
    fp = open(inputFile,'r+')
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines

def create_dictionary(glossPath, glossaries):
    glossaries = glossaries.split(';')
    print(glossaries)
    dictionaries = {}
    print('gloss path is ', glossPath)
    for glossary in glossaries:
        df = pd.read_csv(glossPath+'/'+glossary, na_values=default_missing)
        headings = list(df.keys())
        for j in range(len(df)):
            src_phrase = df.loc[j,'English']
            # print(j,src_phrase)
            src_phrase = src_phrase.lower()
            src_phrase = re.sub(r"\(\'\"[^()]*\)", "", src_phrase).strip()
            tgt_phrase = (str)(df.loc[j,'Hindi'])
            tgt_phrase = tgt_phrase.split(';')
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


def Retriever(dictionaries, inputEnFile, inputHiFile, outputDir):
    # line = 0
    # print(inputFile[:inputFile.rfind('/')]+'/'+outputDir+'/'+inputFile[inputFile.rfind('/')+1:inputFile.rfind('.')]+".constraints.log")
    # 1/0
    # logFile = open(inputEnFile[:inputEnFile.rfind('/')]+'/'+outputDir+'/'+inputEnFile[inputEnFile.rfind('/')+1:inputEnFile.rfind('.')]+".constraints.log",'w')
    # consFile = open(inputEnFile[:inputEnFile.rfind('/')]+'/'+outputDir+'/'+inputEnFile[inputEnFile.rfind('/')+1:inputEnFile.rfind('.')]+".constraints",'w')
    logFile = open(inputEnFile+".constraints.log",'w')
    consFile = open(inputEnFile+".constraints",'w')


    kp = KeywordProcessor()
    # kp_hi = KeywordProcessor()
    for key in dictionaries.keys():
        kp.add_keyword(key, (key, dictionaries[key]))
    print('inputenfile ', inputEnFile)
    print('inputhifile', inputHiFile)
    with open(inputEnFile,'r') as enFile: #, open(inputHiFile,'r') as hiFile:
        enLines = enFile.readlines()
        # hiLines = hiFile.readlines()
        for i, line in enumerate(tqdm(enLines, total=len(enLines))):
            line = line.lower()
            line = re.sub(r"\(\'\"[^()]*\)", "", line).strip()
            matchings = kp.extract_keywords(line)
            # print('matching',i,matchings)
            for matches in set(matchings):
                
                constraints = matches[1].split(',')
                constraints = list(set([cons for cons in [cons.strip() for cons in constraints] if cons]))
                # if(any(tgt_phrase in hiLines[i] for tgt_phrase in constraints)):
                #     # print('cons',constraints)
                logFile.write(str(matches))
                consFile.write(' <sep> '+constraints[0])
                for i in range(1,len(constraints)):
                    consFile.write(' <isep> '+constraints[i])
            logFile.write('\n')
            consFile.write('\n')

def main(glossPath, glossaries, inputEnFile, inputHiFile, outputDir):
    dictionaries = create_dictionary(glossPath, glossaries)
    Retriever(dictionaries,inputEnFile,inputHiFile,outputDir)

if __name__ == "__main__":
    glossPath = sys.argv[1]
    glossaries = sys.argv[2]
    inputEnFile = sys.argv[3]
    inputHiFile = sys.argv[4]
    outputDir = sys.argv[5]
    main(glossPath, glossaries, inputEnFile, inputHiFile, outputDir)
    
