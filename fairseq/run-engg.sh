#!/bin/bash


# BOB Dict Constraints 
bash translate_file.sh inference/eval_final/bob/bob-dict-constraints.en inference/eval_final/bob/dictdis-dict-constraints-1.hi rbi en hi 1
bash translate_file.sh inference/eval_final/bob/bob-dict-constraints.en inference/eval_final/bob/dictdis-dict-constraints-0.hi rbi en hi 0

cd inference
# # Computing BLEU 
bash compute_bleu.sh eval_final/bob/dictdis-dict-constraints-1.hi eval_final/bob/bob-dict-constraints.hi en hi
bash compute_bleu.sh eval_final/bob/dictdis-dict-constraints-0.hi eval_final/bob/bob-dict-constraints.hi en hi

python final_eval.py eval_final/bob/dictdis-dict-constraints-1.hi eval_final/bob/bob-dict-constraints.hi eval_final/bob/bob-dict-constraints.en rbi

python final_eval.py eval_final/bob/dictdis-dict-constraints-0.hi eval_final/bob/bob-dict-constraints.hi eval_final/bob/bob-dict-constraints.en rbi

# BOB Without Constraints 
cd ..
bash translate_file.sh inference/eval_final/bob/bob-wo-constraints.en inference/eval_final/bob/dictdis-wo-constraints-1.hi rbi en hi 1
bash translate_file.sh inference/eval_final/bob/bob-wo-constraints.en inference/eval_final/bob/dictdis-wo-constraints-0.hi rbi en hi 0

# Computing BLEU 

cd inference
bash compute_bleu.sh eval_final/bob/dictdis-wo-constraints-1.hi eval_final/bob/bob-wo-constraints.hi en hi
bash compute_bleu.sh eval_final/bob/dictdis-wo-constraints-0.hi eval_final/bob/bob-wo-constraints.hi en hi

# python final_eval.py eval_final/bob/dictdis-wo-constraints-1.hi eval_final/bob/bob-wo-constraints.hi eval_final/bob/bob-wo-constraints.en bob

# python final_eval.py eval_final/bob/dictdis-wo-constraints-0.hi eval_final/bob/bob-wo-constraints.hi eval_final/bob/bob-wo-constraints.en bob