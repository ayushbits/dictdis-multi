## Glossary
- Glossary should be prepared in a csv format with two columns 'Source' and 'Target' (do not leave space after comma)
- Multiple candidate translations should be given in the form of array.
- refer `glossary.csv` for the format

## Final File - final_eval.py

python final_eval.py RBI/rbi-source.csv RBI/leca.csv RBI/rbi-target.csv ../improved_leca/Data/lexicalDict/ RBI.csv

##‌ How to Translate

1. Google Translate - Use google sheets
2. mBart - python mbart-translate.py <input-file>
3. IndicTrans 
- CHange to username <ganesh>
- Go to /home/ganesh/udaan-indictrans/udaan-deploy-pipeline/translate
- conda activate translation
- ./joint_translate.sh <input-file> <output-file> en hi ../../en-indic/ 


3a) Indictrans- Bilingual
- Chnage to username <krishna>.. pass is kkbhatt@123
- Go to /home/krishna/indictrans_bilingual/krishnakant/training/indicTrans
- models present at ../exp-en-hi
- conda activate bilingual
- CUDA_VISIBLE_DEVICES=2 bash joint_translate.sh air-space.en air-bi-indic.^Ct en hi ../exp-en-hi/  kk

4. Leca Original
- Go to  directory -- /home/piyush/leca/scripts #
- conda activate leca
- ./complete_translate_pipeline.sh air-space.en leca-original.txt  it,phy,mech,chem,math en hi 1
./complete_translate_pipeline.sh ../Data  bob bob lexicalDict rbi en hi 1
Path of stored dictionaries - /home/piyush/leca/Data/lexicalDict

5. Leca Disambiguation - Our approach
Go to directory - /home/piyush/improved_leca/translate
conda activate leca

(leca) piyush@airavat:~/improved_leca/translate$ ./complete_translate_pipeline.sh ../disambigation_eval/aerospace/aero-source.csv aero-leca96.txt  chem,phy,math,it en hi 1

There are currently two models in ./improved_leca/Data folder..
a) models - trained on the source side whose coverage is 96% while 4% is trained without any constraints
b) models_lecaplusrand - above plus 4% is trained on random single constraints.

If you want to change the model path, then rename the directory to the models/


## Below one not applicable
- Go to  /home/piyush/LecaDisambiguation-2/scripts 
#/home/piyush/improved_leca/improved_leca/scripts
- bash complete_translate_pipeline.sh ~/improved_leca/improved_leca/Data/ medical medical lexicalDict med.csv en hi 1

## Create Graphs
python create_graphs.py <directory-name>
python create_graphs.py bob/aerospace/airspace
- It will output weighted disambiguation accuracy and create graphs of SPDI.

- bash create-graphs-run.sh --> Script to run forall datasets


## Find percentage of polysemous degree of the constraint
- python constraints-percentage-testset.py <directory-name>

## Find sentence and constraint mapping

- python ../sent-constraints-mapping.py
- It finds the sentence mapped with the constraint and the candidate constraints
- Given a threshold (specified in line 122), find ambiguous constraints and corresponding sentences
- Reads 4 files, namely, eng sentnce , hindi sentence file, dictionary constraints and constraint log file which contains fired constraint for the given sentence.
- It also calculates frequency of occurence of each source side constraint.


- Writes output in this file 'english_ambiguous.txt', hindi_ambiguous.txt,  and 'constraint_ambiguous.txt

## Extract ambiguous pairs in testset

python extract-ambiguous-testset.py air-space/airspace-leca96ep28_eval.csv  leca96ep28

## Find micro-disambiguation score for training dataset 
python training-wise-epoch.py



<!-- ## Disambiguation Accuracy -
- It finds disambiguation acurracy i.e. how many hits from glossary in the predicted sentences. 
- Logic: Looks at glossary of the source word and finds whether any words from glossary is present in the predicted sentences
- `python eval.py <translation_csv_file> <name_of_source_column> <name_of_target_column> <output_file>`
- It creates `<output_file>` which will be used for calculating SPDI
- glossary file is hard-coded with the name `glossary.csv`

## SPDI - Sense Polysemous Degree Importance (SPDI) 
- Reporting disambiguation accuracy by grouping based on polysemy degree of the source word i.e. how many candidate translations of source word are possible

- `python spdi.py <translation_csv_file> <name_of_source_column> <name_of_target_column> <output_file>`
- <output_file> is created from `eval.py` -->
<!-- - It will produces a plot of SPDI with name <translation_csv_file>.jpg -->