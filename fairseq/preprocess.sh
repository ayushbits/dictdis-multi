exp_dir=$1 # Data folder to be given
glossPath=$2 # path where all dict are stored .. lexicalDict
src_lang=$3 # en
tgt_lang=$4 #hi

num_operations=35000

trainingData=$exp_dir/raw_data
mkdir -p $trainingData/processedData
mkdir -p $trainingData/processedData/norm
mkdir -p $trainingData/processedData/bpe

SUBWORD_NMT_DIR='../../../subword-nmt'

# echo "########################################## Train File Processing ##########################################"

# echo "Applying normalization and script conversion to train.en file"
# input_size=`python scripts/preprocess_translate.py $trainingData/train.$src_lang $trainingData/processedData/norm/train.$src_lang.norm $src_lang true`
# echo "Number of sentences in input: $input_size"

# echo "Applying normalization and script conversion to train.hi file"
# input_size=`python scripts/preprocess_translate.py $trainingData/train.$tgt_lang $trainingData/processedData/norm/train.$tgt_lang.norm $tgt_lang true`
# echo "Number of sentences in input: $input_size"

# ### learning BPE
# mkdir -p $exp_dir/vocab
# echo "learning joint BPE"
# cat $trainingData/processedData/norm/train.$src_lang.norm  $trainingData/processedData/norm/train.$tgt_lang.norm > $train_file.ALL
# python $SUBWORD_NMT_DIR/subword_nmt/learn_bpe.py \
#    --input $train_file.ALL \
#    -s $num_operations \
#    -o $exp_dir/vocab/bpe_codes.32k.${src_lang}_${tgt_lang} \
#    --num-workers -10

# echo "computing SRC vocab"
# python $SUBWORD_NMT_DIR/subword_nmt/apply_bpe.py \
#     -c $exp_dir/vocab/bpe_codes.32k.${src_lang}_${tgt_lang} \
#     --num-workers -100  \
#     -i  $trainingData/processedData/norm/train.$src_lang.norm  | \
# python $SUBWORD_NMT_DIR/subword_nmt/get_vocab.py \
#     > $exp_dir/vocab/vocab.tmp.$src_lang
# python scripts/clean_vocab.py $exp_dir/vocab/vocab.tmp.$src_lang $exp_dir/vocab/vocab.$src_lang
# #rm $expdir/vocab/vocab.tmp.SRC

# echo "computing TGT vocab"
# python $SUBWORD_NMT_DIR/subword_nmt/apply_bpe.py \
#     -c $exp_dir/vocab/bpe_codes.32k.${src_lang}_${tgt_lang} \
#     --num-workers -100  \
#     -i  $trainingData/processedData/norm/train.$tgt_lang.norm  | \
# python $SUBWORD_NMT_DIR/subword_nmt/get_vocab.py \
#     > $exp_dir/vocab/vocab.tmp.$tgt_lang
# python scripts/clean_vocab.py $exp_dir/vocab/vocab.tmp.$tgt_lang $exp_dir/vocab/vocab.$tgt_lang
# #rm $expdir/vocab/vocab.tmp.TGT

# rm $train_file.ALL

# ### apply BPE to input file

# echo "Applying BPE to train.en file"
# python $SUBWORD_NMT_DIR/subword_nmt/apply_bpe.py \
#     -c $exp_dir/vocab/bpe_codes.32k.${src_lang}_${tgt_lang} \
#     --vocabulary $exp_dir/vocab/vocab.$src_lang \
#     --vocabulary-threshold 5 \
#     < $trainingData/processedData/norm/train.$src_lang.norm \
#     > $trainingData/processedData/bpe/train.$src_lang

# echo "Applying BPE to train.hi file"
# python $SUBWORD_NMT_DIR/subword_nmt/apply_bpe.py \
#     -c $exp_dir/vocab/bpe_codes.32k.${src_lang}_${tgt_lang} \
#     --vocabulary $exp_dir/vocab/vocab.$tgt_lang \
#     --vocabulary-threshold 5 \
#     < $trainingData/processedData/norm/train.$tgt_lang.norm \
#     > $trainingData/processedData/bpe/train.$tgt_lang


# echo "########################################## Validation File Processing ##########################################"

# echo "Applying normalization and script conversion to valid.en file"
# input_size=`python scripts/preprocess_translate.py $trainingData/valid.$src_lang $trainingData/processedData/norm/valid.$src_lang.norm $src_lang true`
# echo "Number of sentences in input: $input_size"

# echo "Applying normalization and script conversion to valid.hi file"
# input_size=`python scripts/preprocess_translate.py $trainingData/valid.$tgt_lang $trainingData/processedData/norm/valid.$tgt_lang.norm $src_lang true`
# echo "Number of sentences in input: $input_size"

# ### apply BPE to input file

# echo "Applying BPE to valid.en file"
# python $SUBWORD_NMT_DIR/subword_nmt/apply_bpe.py \
#     -c $exp_dir/vocab/bpe_codes.32k.${src_lang}_${tgt_lang} \
#     --vocabulary $exp_dir/vocab/vocab.$src_lang \
#     --vocabulary-threshold 5 \
#     < $trainingData/processedData/norm/valid.$src_lang.norm \
#     > $trainingData/processedData/bpe/valid.$src_lang

# echo "Applying BPE to valid.hi file"
# python $SUBWORD_NMT_DIR/subword_nmt/apply_bpe.py \
#     -c $exp_dir/vocab/bpe_codes.32k.${src_lang}_${tgt_lang} \
#     --vocabulary $exp_dir/vocab/vocab.$tgt_lang \
#     --vocabulary-threshold 5 \
#     < $trainingData/processedData/norm/valid.$tgt_lang.norm \
#     > $trainingData/processedData/bpe/valid.$tgt_lang

# echo "########################################## Test File Processing ##########################################"

# echo "Applying normalization and script conversion to test.en file"
# input_size=`python scripts/preprocess_translate.py $trainingData/test.$src_lang $trainingData/processedData/norm/test.$src_lang.norm $src_lang true`
# echo "Number of sentences in input: $input_size"

# echo "Applying normalization and script conversion to test.hi file"
# input_size=`python scripts/preprocess_translate.py $trainingData/test.$tgt_lang $trainingData/processedData/norm/test.$tgt_lang.norm $src_lang true`
# echo "Number of sentences in input: $input_size"

# ### apply BPE to input file

# echo "Applying BPE to test.en file"
# python $SUBWORD_NMT_DIR/subword_nmt/apply_bpe.py \
#     -c $exp_dir/vocab/bpe_codes.32k.${src_lang}_${tgt_lang} \
#     --vocabulary $exp_dir/vocab/vocab.$src_lang \
#     --vocabulary-threshold 5 \
#     < $trainingData/processedData/norm/test.$src_lang.norm \
#     > $trainingData/processedData/bpe/test.$src_lang

# echo "Applying BPE to test.hi file"
# python $SUBWORD_NMT_DIR/subword_nmt/apply_bpe.py \
#     -c $exp_dir/vocab/bpe_codes.32k.${src_lang}_${tgt_lang} \
#     --vocabulary $exp_dir/vocab/vocab.$tgt_lang \
#     --vocabulary-threshold 5 \
#     < $trainingData/processedData/norm/test.$tgt_lang.norm \
#     > $trainingData/processedData/bpe/test.$tgt_lang

# echo "########################################## Binarising Datasets ##########################################"

# CUDA_VISIBLE_DEVICES=2 python fairseq_cli/preprocess.py --source-lang en --target-lang hi \
# 		      --trainpref $trainingData/processedData/bpe/train --validpref $trainingData/processedData/bpe/valid \
# 		      --testpref $trainingData/processedData/bpe/test --destdir $exp_dir/final_bin --workers 8

# # echo "########################################## Constraints File Processing ##########################################"

glossaries='final-eng-hindi.csv'
# echo "Creating constraints for train.en"
# outputDir='constraints'
# mkdir -p $trainingData/constraints
# python scripts/create_constraints.py $exp_dir/$glossPath $glossaries $trainingData/train.$src_lang $trainingData/train.$tgt_lang $outputDir

# echo "Applying BPE to constraints"
# python $SUBWORD_NMT_DIR/subword_nmt/apply_bpe.py \
#     -c $exp_dir/vocab/bpe_codes.32k.${src_lang}_${tgt_lang} \
#     --vocabulary $exp_dir/vocab/vocab.$tgt_lang \
#     --vocabulary-threshold 5 \
#     < $trainingData/constraints/train.constraints \
#     > $exp_dir/final_bin/train.constraints.temp

# echo "Cleaning Train Constraints"
# python scripts/clean_constraints.py $exp_dir/final_bin/train.constraints.temp $exp_dir/final_bin/train.${src_lang}-${tgt_lang}.constraints

# rm $exp_dir/final_bin/train.constraints.temp

echo "Creating constraints for valid.en"
outputDir='constraints'
python scripts/create_constraints.py $exp_dir/$glossPath $glossaries $trainingData/valid.$src_lang $trainingData/train.$src_lang $outputDir

echo "Applying BPE to constraints"
python $SUBWORD_NMT_DIR/subword_nmt/apply_bpe.py \
    -c $exp_dir/vocab/bpe_codes.32k.${src_lang}_${tgt_lang} \
    --vocabulary $exp_dir/vocab/vocab.$tgt_lang \
    --vocabulary-threshold 5 \
    < $trainingData/constraints/valid.constraints \
    > $exp_dir/final_bin/valid.constraints.temp

echo "Cleaning Valid Constraints"
python scripts/clean_constraints.py $exp_dir/final_bin/valid.constraints.temp $exp_dir/final_bin/valid.${src_lang}-${tgt_lang}.constraints

rm $exp_dir/final_bin/valid.constraints.temp
