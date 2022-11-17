# bash complete_translate_pipeline.sh Data aerospace aerospace lexicalDict RBI.csv en hi 1 
# bash complete_translate_pipeline.sh Data medical medical lexicalDict med.csv en hi 1 
#!/bin/bash
#echo `date`
exp_dir=$1 #Dir containing model final_bin vocab etc.
dataset=$2 #Dataset for inference
infname=$3 #filename inside dataset folder (for eg aerospace dataset folder has aerospace.en as file. Here infname=aerospace and extension en is added using src_lang)
glossaryDir=$4 #Dir where dictionaries are stored
glossary=$5
src_lang=$6
tgt_lang=$7
cons=$8 #Constraints: Expected -- 0/1 --> [1->'True',0->'False']

pwd
cd ..
pwd

datasetDir=$exp_dir/devtest/en-hi/$dataset #Given dataset construct full path where dataset resides.
SRC_PREFIX='SRC'
TGT_PREFIX='TGT'
outputDir='constraint_decoding13' #name of output Dir inside datasetDir as defined above.
# outputDir='sampleTest'
mkdir -p $datasetDir/$outputDir
glossaryPath=$exp_dir/$glossaryDir #construct full dict path

#`dirname $0`/env.sh
SUBWORD_NMT_DIR='../../subword-nmt'
# model_dir=$exp_dir/align_model
# data_bin_dir=$exp_dir/sub_align_binarised

# model_dir=$exp_dir/align_model
# data_bin_dir=$exp_dir/align_binarised

model_dir="/home/souvik/improved_leca/trial_v12/fairseq/checkpoints"
data_bin_dir=$exp_dir/final_bin

# Constraints retriever script
constraints=$cons # [1->'True',0->'False']
if [[ $constraints -eq 1 ]]
then
echo "########################################## Constraints File Processing ##########################################"
# glossary='chemistryGlossary.csv,physicsGlossary.csv,mathGlossary.csv,itGlossary.csv,MechGlossary.csv' #specify which all dict is needed as string seperated by comma
# glossary='NGMA_Dict.csv'
#='chemistryGlossary.csv,physicsGlossary.csv,mathGlossary.csv,itGlossary.csv'
echo "Retrieving Constraints"
# python retriever_vdba.py $glossaryPath $glossary $datasetDir/$infname $src_lang $outputDir
python scripts/create_constraints.py $glossaryPath $glossary $datasetDir/$infname.$src_lang $datasetDir/$infname.$tgt_lang $outputDir

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
input_size=`python scripts/preprocess_translate.py $datasetDir/$infname.$src_lang $datasetDir/$outputDir/$infname.norm $src_lang true`
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

tgt_output_fname=$datasetDir/$outputDir/$infname.predicted.$tgt_lang



echo "Translation Started"
useptr='--use-ptrnet'
#CUDA_VISIBLE_DEVICES=0 fairseq-interactive  $data_bin_dir \
CUDA_VISIBLE_DEVICES=2 python fairseq_cli/interactive.py $data_bin_dir \
    -s $src_lang -t $tgt_lang \
    --path $model_dir/checkpoint_best.pt \
    --beam 5  --remove-bpe --quiet --consnmt $useptr \
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
echo "Computing Bleu Score"
bash scripts/compute_bleu.sh $tgt_output_fname $datasetDir/$infname.$tgt_lang $src_lang $tgt_lang

#echo "Computing Disambiguation Accuracy"
#python scripts/constraints_metrics.py $datasetDir/$infname.$src_lang $tgt_output_fname $datasetDir/$infname.$tgt_lang $glossaryPath $glossary 'English' 'Hindi'
#echo "All Done"
