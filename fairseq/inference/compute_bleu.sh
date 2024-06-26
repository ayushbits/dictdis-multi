pred_fname=$1
ref_fname=$2
src_lang=$3
tgt_lang=$4

# we compute and report tokenized bleu scores.
# For computing BLEU scores, systems should output detokenized outputs. Your MT system might be doing it out of the box if you are using SentencePiece - nothing to do in that case.
# If you are using BPE then:
# 1. For English, you can use MosesDetokenizer (either the scripts in moses or the sacremoses python package)
# 2. For Indian languages, you can use the IndicNLP library detokenizer (note: please don't skip this step, since detok/tokenizer are not guaranteed to be reversible**.
# ^ both 1. and 2. are scripts/postprocess_translate.py


# For computing BLEU, we use sacrebleu:
# For English output: sacrebleu reffile < outputfile. This internally tokenizes using mteval-v13a
# For Indian language output, we need tokenized output and reference since we don't know how well the sacrebleu tokenizer works for Indic input.
# Hence we tokenize both preds and target files with IndicNLP tokenizer and then run: sacrebleu --tokenize none reffile < outputfile
if [ $tgt_lang == 'en' ] || [ $tgt_lang == 'de' ]; then
    # indic to en models
    sacrebleu --tokenizer none $ref_fname < $pred_fname >  ${pred_fname}.bleu
else
    # if input is not detokenized, first detokenize it
    python ../scripts/detokenize.py $pred_fname $pred_fname.detok $tgt_lang
    `sed -i '/^$/d' $pred_fname.detok`
    # indicnlp tokenize predictions and reference files before evaluation
    input_size=`python ../scripts/preprocess_translate.py $ref_fname $ref_fname.tok $tgt_lang`
    input_size=`python ../scripts/preprocess_translate.py $pred_fname.detok $pred_fname.tok $tgt_lang`

    # since we are tokenizing with indicnlp separately, we are setting tokenize to none here
    sacrebleu --tokenize none $ref_fname.tok < $pred_fname.tok > ${pred_fname}.bleu
fi