pred_fname=$1
ref_fname=$2
src_fname=$3
CUDA_VISIBLE_DEVICES=3 python comet_compute.py $src_fname $pred_fname $ref_fname > ${pred_fname}-comet-chrf.txt
CDUA_VISIBLE_DEVICES=3 python chrf_compute.py $pred_fname $ref_fname >> ${pred_fname}-comet-chrf.txt
