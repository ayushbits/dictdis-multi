# bash translate_file.sh inference/flores/flores.en inference/flores/flores-pred-1-new.hi  None  en hi 0
#!/bin/bash
echo `date`
exp_dir='full_data' #Dir containing model final_bin vocab etc.
#exp_dir='full_data/bobdata' #Dir containing model final_bin vocab etc.

# exp_dir='../LecaDisambiguationExp-2/LecaExp2/Data' #Dir containing model final_bin vocab etc.
#exp_dir='../../leca/Data' #Dir containing model final_bin vocab etc.
#exp_dir='../../../udaan-deploy-flask/leca_model' #Dir containing model final_bin vocab etc.
dataset='.' #Dataset for inference
infname=$1 #filename inside dataset folder (for eg aerospace dataset folder has aerospace.en as file. Here infname=aerospace and extension en is added using src_lang)
outfname=$2
glossaryList=$3 #Dir where dictionaries are stored
src_lang=$4
tgt_lang=$5
cons=$6 #Constraints: Expected -- 0/1 --> [1->'True',0->'False']

#datasetDir=$exp_dir/devtest/en-hi/$dataset #Given dataset construct full path where dataset resides.
datasetDir=$dataset
SRC_PREFIX='SRC'
TGT_PREFIX='TGT'
outputDir='.' #name of output Dir inside datasetDir as defined above.
# outputDir='sampleTest'
# mkdir -p $datasetDir/$outputDir
#glossaryPath=$exp_dir/$glossaryDir #construct full dict path
#glossaryPath='../improved_leca/Data/lexicalDict' #$glossaryDir
glossaryPath='full_data/lexicalDict'
#`dirname $0`/env.sh
SUBWORD_NMT_DIR='../../../subword-nmt'
# model_dir=$exp_dir/align_model
# data_bin_dir=$exp_dir/sub_align_binarised

# model_dir=$exp_dir/align_model
# model_dir="/home/souvik/improved_leca/trial_v12/dictdis_multigpu/fairseq/checkpoints/"
# model_dir="checkpoints-18ep/"
model_dir="checkpoints_en_hi_4x/"
#model_dir = "checkpoints_bob_param/"
# model_dir="checkpoints"
#-wmt-arch/"
#$exp_dir/models
# data_bin_dir=$exp_dir/align_binarised
data_bin_dir=$exp_dir/final_bin

# Constraints retriever script
constraints=$cons # [1->'True',0->'False']
if [[ $constraints -eq 1 ]]
then
echo "########################################## Constraints File Processing ##########################################"
# glossary='chemistryGlossary.csv,physicsGlossary.csv,mathGlossary.csv,itGlossary.csv,MechGlossary.csv' #specify which all dict is needed as string seperated by comma
echo 'glossary is ': $glossaryList
# glossaryList = $glossaryList.split(',')
glossary=""
IFS=','
read -a lst <<< "$glossaryList"
len=${#lst[@]}
j=0
for i in ${lst[@]}
do
    j=$((j+1))
    echo "$i"
    glossary+="$i.csv"
    if [[ $j -ne $len ]]
    then
        glossary+=';'
    fi
done
# len=`ls ./$glossaryList | wc -l`
# j=0
# echo $len
# for gloss in $glossaryList/*.csv; do
#     j=$((j+1))
#     glossName=$(basename $gloss)
#     glossary+=$glossName
#     if [[ $j -le 2 ]]; then
#         glossary+=','
#     else
#         break 1
#     fi
# done
echo "$glossary"
# glossary='rbi.csv,it.csv'

# print('glossary is ', glossary)
# glossary='chemistryGlossary.csv,physicsGlossary.csv,mathGlossary.csv,itGlossary.csv'

echo "Applying normalization and script conversion to test file"
input_size=`python scripts/preprocess_translate.py $datasetDir/$infname $datasetDir/$outputDir/$infname.norm $src_lang true`
echo "Number of sentences in input: $input_size"

cp $datasetDir/$outputDir/$infname.norm $datasetDir/$outputDir/$infname

echo "Retrieving Constraints"
# python retriever_vdba.py $glossaryPath $glossary $datasetDir/$infname $src_lang $outputDir
python scripts/create_constraints_inference.py $glossaryPath $glossary $datasetDir/$infname $outputDir/$infname.$tgt_lang $outputDir

# echo "Retrieving Constraints"
# python scripts/retriever_vdba.py $glossaryPath $glossary $datasetDir/$infname $outputDir

echo "Applying normalization and script conversion to constraints"
python scripts/preprocess_translate.py $datasetDir/$outputDir/$infname.constraints $datasetDir/$outputDir/$infname.constraints.norm $tgt_lang true

echo "Applying BPE to constraints"
python $SUBWORD_NMT_DIR/subword_nmt/apply_bpe.py \
    -c $exp_dir/vocab/bpe_codes.32k.${src_lang}_${tgt_lang} \
    --vocabulary $exp_dir/vocab/vocab.$tgt_lang \
    --vocabulary-threshold 5 \
    < $datasetDir/$outputDir/$infname.constraints.norm \
    > $datasetDir/$outputDir/$infname.constraints._bpe

echo "Cleaning Train Constraints"
python scripts/clean_constraints.py $datasetDir/$outputDir/$infname.constraints._bpe $datasetDir/$outputDir/$infname.constraints.bpe
fi

### normalization and script conversion

echo "########################################## Test File Processing ##########################################"

echo "Applying normalization and script conversion to test file"
input_size=`python scripts/preprocess_translate.py $datasetDir/$infname $datasetDir/$outputDir/$infname.norm $src_lang true`
echo "Number of sentences in input: $input_size"

### apply BPE to input file

echo "Applying BPE to test file"
python $SUBWORD_NMT_DIR/subword_nmt/apply_bpe.py \
    -c $exp_dir/vocab/bpe_codes.32k.${src_lang}_${tgt_lang} \
    --vocabulary $exp_dir/vocab/vocab.$src_lang \
    --vocabulary-threshold 5 \
    < $datasetDir/$outputDir/$infname.norm \
    > $datasetDir/$outputDir/$infname.bpe

if [[ $constraints -eq 1 ]]
then
echo "########################################## Concatenating test and constraints file ##########################################"
python scripts/combine.py $datasetDir/$outputDir/$infname.bpe $datasetDir/$outputDir/$infname.constraints.bpe $datasetDir/$outputDir/$infname.leca
fi


# ### run decoder

# echo "########################################## Decoding ##########################################"
if [[ $constraints -eq 1 ]]
then
src_input_bpe_fname=$datasetDir/$outputDir/$infname.leca
else
src_input_bpe_fname=$datasetDir/$outputDir/$infname.bpe
fi

# tgt_output_fname=$datasetDir/$outputDir/$infname.predicted.$tgt_lang
tgt_output_fname=$datasetDir/$outputDir/$outfname

pwd

echo "Translation Started"
useptr='--use-ptrnet'
# CUDA_VISIBLE_DEVICES=0 fairseq-interactive  $data_bin_dir \
CUDA_VISIBLE_DEVICES=2 python fairseq_cli/interactive2.py $data_bin_dir \
    -s $src_lang -t $tgt_lang  --batch-size 1 --buffer-size 2500 \
    --path $model_dir/checkpoint_best.pt \
    --beam 5  --remove-bpe --consnmt $useptr \
    --model-overrides "{'beam':5}" \
    --input $src_input_bpe_fname  >  $tgt_output_fname.log 2>&1

#echo "fairseq-interactive $data_bin_dir -s $SRC_PREFIX -t $TGT_PREFIX --distributed-world-size 1  --path $model_dir/checkpoint_best.pt  --batch-size 64  --buffer-size 2500 --beam 5  --remove-bpe --skip-invalid-size-inputs-valid-test  --user-dir model_configs--input $src_input_bpe_fname  >  $tgt_output_fname.log "
#pwd
input_size=`wc --lines  < $src_input_bpe_fname`
echo $input_size

echo "Extracting translations, script conversion and detokenization"
# this part reverses the transliteration from devnagiri script to target lang and then detokenizes it.
python scripts/postprocess_translate.py $tgt_output_fname.log $tgt_output_fname $input_size $tgt_lang true

echo `date`
echo "Translation completed"
# echo "Computing Bleu Score"
# bash compute_bleu.sh $tgt_output_fname $datasetDir/$infname.$tgt_lang $src_lang $tgt_lang
