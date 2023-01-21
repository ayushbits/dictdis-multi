#!/bin/bash


# BOB Dict Constraints 
bash translate_file.sh inference/eval_final/wat2021/wat2021-dict-constraints.en inference/eval_final/wat2021/dictdis-dict-constraints-1.hi administrative en hi 1
bash translate_file.sh inference/eval_final/wat2021/wat2021-dict-constraints.en inference/eval_final/wat2021/dictdis-dict-constraints-0.hi administrative en hi 0

cd inference
# # Computing BLEU 
bash compute_bleu.sh eval_final/wat2021/dictdis-dict-constraints-1.hi eval_final/wat2021/wat2021-dict-constraints.hi en hi
bash compute_bleu.sh eval_final/wat2021/dictdis-dict-constraints-0.hi eval_final/wat2021/wat2021-dict-constraints.hi en hi

python final_eval.py eval_final/wat2021/dictdis-dict-constraints-1.hi eval_final/wat2021/wat2021-dict-constraints.hi eval_final/wat2021/wat2021-dict-constraints.en administrative

python final_eval.py eval_final/wat2021/dictdis-dict-constraints-0.hi eval_final/wat2021/wat2021-dict-constraints.hi eval_final/wat2021/wat2021-dict-constraints.en administrative

# BOB Without Constraints 
cd ..
bash translate_file.sh inference/eval_final/wat2021/wat2021-wo-constraints.en inference/eval_final/wat2021/dictdis-wo-constraints-1.hi administrative en hi 1
bash translate_file.sh inference/eval_final/wat2021/wat2021-wo-constraints.en inference/eval_final/wat2021/dictdis-wo-constraints-0.hi administrative en hi 0

# Computing BLEU 

cd inference
bash compute_bleu.sh eval_final/wat2021/dictdis-wo-constraints-1.hi eval_final/wat2021/wat2021-wo-constraints.hi en hi
bash compute_bleu.sh eval_final/wat2021/dictdis-wo-constraints-0.hi eval_final/wat2021/wat2021-wo-constraints.hi en hi

# python final_eval.py eval_final/wat2021/dictdis-wo-constraints-1.hi eval_final/wat2021/wat2021-wo-constraints.hi eval_final/wat2021/wat2021-wo-constraints.en wat2021

# python final_eval.py eval_final/bob/dictdis-wo-constraints-0.hi eval_final/wat2021/wat2021-wo-constraints.hi eval_final/wat2021/wat2021-wo-constraints.en wat2021