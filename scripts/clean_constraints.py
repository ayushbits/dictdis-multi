import sys
import mmap
from tqdm import tqdm

### This method takes more time(by factor of 25) than simple string replacement
# from flashtext import KeywordProcessor
# keyword_processor = KeywordProcessor()
# keyword_processor.add_keyword('<@@ se@@ p@@ >','<sep>')
# keyword_processor.add_keyword('<@@ is@@ ep@@ >','<isep>')

def numberOfLines(inputFile):
    fp = open(inputFile,'r+')
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines

def cleanConstraints(bped_constraints, cleaned_bped_constraints):
    '''Function to replace bped version of <sep> and <isep>'''
    with open(bped_constraints,'r') as Bpe, open(cleaned_bped_constraints,'w') as cleanedBpe:
        for line in tqdm(Bpe, total=numberOfLines(bped_constraints)):
            cleanedBpe.write(line.replace('<@@ se@@ p@@ >','<sep>').replace('<@@ is@@ ep@@ >','<isep>').replace('<@@ s@@ e@@ p@@ >','<sep>').replace('<@@ i@@ s@@ e@@ p@@ >','<isep>').replace('< se@@ p >','<sep>').replace('< ise@@ p >','<isep>').replace('< is@@ ep >','<isep>'))
            #cleanedBpe.write(line.replace('<@@ s@@ e@@ p@@ >','<sep>').replace('<@@ i@@ s@@ e@@ p@@ >','<isep>'))
            #cleanedBpe.write(line.replace('< se@@ p >','<sep>').replace('< is@@ ep >','<isep>'))
            # cleanedBpe.write(keyword_processor.replace_keywords(line))

if __name__ == "__main__":
    bped_constraints = sys.argv[1]
    cleaned_bped_constraints = sys.argv[2]
    cleanConstraints(bped_constraints,cleaned_bped_constraints)