#!/bin/bash

train_model(){
    src=$1
    tgt=$2
    workloc=$3 
    modelname=$4

    # fseq=$workloc/align_binarised
    fseq=$workloc/final_bin
    # fseq=$workloc/5k_bin
    #fseq=$workloc/5M_bin


    modeldir=$workloc/engg-enhi-trans4x
    #modeldir=$workloc/models_alpha_5M
    #modeldir=$workloc/models_check_samantar_newfinal
    echo "This is model $modeldir"
    mkdir -p $fseq $modeldir

    if [[ $modelname == *"ptr"* ]]; then
        useptr='--use-ptrnet'   
        fp16='' ## TODO:loss is not convergent when using fp16 for pointer network. 
    else
        useptr=''
        fp16='--fp16'
    fi
    
    if [[ $src == *"zh"* ]] || [[ $tgt == *"zh"* ]]; then
        MaxUpdates=60000    
    else
        MaxUpdates=300000
    fi

    echo "train ${modelname} NMT on $src-to-$tgt ..."
    now=$(date +"%T")
    echo "Start time : $now" 
    python fairseq_cli/train.py $fseq \
        -a transformer_4x --optimizer adam --lr 0.0005 -s $src -t $tgt \
        --distributed-world-size 3 --num-workers 0 --ddp-backend no_c10d \
        --label-smoothing 0.1 --dropout 0.2 --max-tokens 2048 --update-freq 1 --seed 1 --patience 5 \
        --stop-min-lr '1e-09' --lr-scheduler inverse_sqrt --weight-decay 0.0001 \
        --criterion label_smoothed_cross_entropy --max-update $MaxUpdates --exp-name "${modelname}-${src}-${tgt}" \
        --warmup-updates 4000 --warmup-init-lr '1e-07' --keep-last-epochs 30 \
        --adam-betas '(0.9, 0.98)'  --save-dir bobdata/bob_enhi-trans4x \
        --clip-norm 1.0 \
        --tensorboard-logdir $modeldir/tensorboard --consnmt $useptr \
        --eval-bleu \
        --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
        --eval-bleu-detok moses \
        --eval-bleu-remove-bpe \
        --eval-bleu-print-samples \
        --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
        --restore-file checkpoints_en_hi_4x/checkpoint_best.pt \
        2>&1 | tee   $modeldir/log.txt
}
# --restore-file checkpoints_latest/checkpoint_best.pt \

# --eval-bleu \
#         --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
#         --eval-bleu-detok moses \
#         --eval-bleu-remove-bpe \
#         --eval-bleu-print-samples \
#         --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \

get_test_BLEU(){
    src=$1
    tgt=$2
    workloc=$3 
    modelname=$4
    testclean=$5

    fseq=$workloc/binarised
    modeldir=$workloc/model
    resdir=$workloc/result && mkdir -p $resdir 
    # raw_reference=$workloc/raw/test.$tgt

    if [[ $modelname == *"ptr"* ]]; then
        useptr='--use-ptrnet'    
    else
        useptr=''
    fi

    if [[ $testclean == "1" ]]; then
        echo "test on clean dataset..."
        python generate.py $fseq -s $src -t $tgt \
            --path $modeldir/checkpoint_best_bleu.pt \
            --batch-size 20 --remove-bpe --sacrebleu \
            --decoding-path $resdir --quiet --testclean --consnmt $useptr \
            --model-overrides "{'beam':10}"
    else 
        echo "test on cons dataset..."
        python generate.py $fseq -s $src -t $tgt \
            --path $modeldir/checkpoint_best_bleu.pt \
            --batch-size 20 --remove-bpe --sacrebleu \
            --decoding-path $resdir --quiet --consnmt $useptr \
            --model-overrides "{'beam':10}"
    fi 

    # detok=$workloc/detokenize.perl
    # perl $detok -l $tgt < $resdir/decoding.txt > $resdir/decoding.detok
    # perl $detok -l $tgt < $raw_reference > $resdir/target.detok
    # cat $resdir/decoding.detok | sacrebleu $resdir/target.detok

    if [[ $testclean == "0" ]]; then
        python scripts/cal_CSR.py --src $resdir/source.txt --tgt $raw_reference \
            --hyp $resdir/decoding.txt 
    fi

}


set -e 
cd ..
pwd
export CUDA_VISIBLE_DEVICES=1,2,3
export CUDA_LAUNCH_BLOCKING=1
# export datadir=trial_data
# export datadir=full_data
export datadir=bobdata


##The processed data locates in $datadir/processed_data 
##The raw format of test reference locates in $datadir/raw/test.$tgt
##The moses detokenize scripts de
src=en
tgt=hi

ModelType=leca_ptrnet  ## choice=['leca','leca_ptrnet']

echo "start to train ${ModelType} NMT model ..."
train_model $src $tgt $datadir $ModelType

# echo "(1) Test on constraint-free test set"
# testclean=1
# get_test_BLEU $src $tgt $datadir $ModelType $testclean

#echo "(2) Test on target constraints test set"
#testclean=0
#get_test_BLEU $src $tgt $datadir $ModelType $testclean 
