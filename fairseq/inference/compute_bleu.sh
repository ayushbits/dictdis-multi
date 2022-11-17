#bash compute_bleu.sh aerospace/aero-mbart50.txt aerospace/aero-target.csv en hi
pred_fname=$1
ref_fname=$2
src_lang=$3
tgt_lang=$4



if [ $tgt_lang == 'en' ]; then
    # indic to en models
    sacrebleu $ref_fname < $pred_fname
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
