#!/bin/bash

## BOB
bash compute_comet_chrf.sh eval_final/bob/leca-dict-constraints.hi eval_final/bob/bob-dict-constraints.hi eval_final/bob/bob-dict-constraints.en
bash compute_comet_chrf.sh eval_final/bob/dictdis-dict-constraints-1.hi eval_final/bob/bob-dict-constraints.hi eval_final/bob/bob-dict-constraints.en
bash compute_comet_chrf.sh eval_final/bob/indic-dict-constraints-final.hi eval_final/bob/bob-dict-constraints.hi eval_final/bob/bob-dict-constraints.en

bash compute_comet_chrf.sh eval_final/bob/vdba-dict-constraints.hi eval_final/bob/bob-dict-constraints.hi eval_final/bob/bob-dict-constraints.en


# RBI
bash compute_comet_chrf.sh eval_final/rbi/dictdis-dict-constraints-1.hi eval_final/rbi/rbi-dict-constraints.hi eval_final/rbi/rbi-dict-constraints.en
bash compute_comet_chrf.sh eval_final/rbi/indic-dict-constraints.hi eval_final/rbi/rbi-dict-constraints.hi eval_final/rbi/rbi-dict-constraints.en
bash compute_comet_chrf.sh eval_final/rbi/leca-dict-constraints.hi eval_final/rbi/rbi-dict-constraints.hi eval_final/rbi/rbi-dict-constraints.en

bash compute_comet_chrf.sh eval_final/rbi/vdba-dict-constraints.hi eval_final/rbi/rbi-dict-constraints.hi eval_final/rbi/rbi-dict-constraints.en

# Flores
bash compute_comet_chrf.sh eval_final/flores/dictdis-dict-constraints-1.hi eval_final/flores/flores-dict-constraints.hi eval_final/flores/flores-dict-constraints.en
bash compute_comet_chrf.sh eval_final/flores/indic-dict-constraints.hi eval_final/flores/flores-dict-constraints.hi eval_final/flores/flores-dict-constraints.en
bash compute_comet_chrf.sh eval_final/flores/leca-dict-constraints.hi eval_final/flores/flores-dict-constraints.hi eval_final/flores/flores-dict-constraints.en

bash compute_comet_chrf.sh eval_final/flores/vdba-dict-constraints.hi eval_final/flores/flores-dict-constraints.hi eval_final/flores/flores-dict-constraints.en


# Wat2021
bash compute_comet_chrf.sh eval_final/wat2021/dictdis-dict-constraints-1.hi eval_final/wat2021/wat2021-dict-constraints.hi eval_final/wat2021/wat2021-dict-constraints.en
bash compute_comet_chrf.sh eval_final/wat2021/indic-dict-constraints.hi eval_final/wat2021/wat2021-dict-constraints.hi eval_final/wat2021/wat2021-dict-constraints.en
bash compute_comet_chrf.sh eval_final/wat2021/leca-dict-constraints.hi eval_final/wat2021/wat2021-dict-constraints.hi eval_final/wat2021/wat2021-dict-constraints.en

bash compute_comet_chrf.sh eval_final/wat2021/vdba-dict-constraints.hi eval_final/wat2021/wat2021-dict-constraints.hi eval_final/wat2021/wat2021-dict-constraints.en


# # ENGG
bash compute_comet_chrf.sh eval_final/engg/dictdis-dict-constraints-1.hi eval_final/engg/engg-dict-constraints.hi eval_final/engg/engg-dict-constraints.en
bash compute_comet_chrf.sh eval_final/engg/indic-dict-constraints.hi eval_final/engg/engg-dict-constraints.hi eval_final/engg/engg-dict-constraints.en
bash compute_comet_chrf.sh eval_final/engg/leca-dict-constraints.hi eval_final/engg/engg-dict-constraints.hi eval_final/engg/engg-dict-constraints.en

bash compute_comet_chrf.sh eval_final/engg/vdba-dict-constraints.hi eval_final/engg/engg-dict-constraints.hi eval_final/engg/engg-dict-constraints.en

# WMT14
bash compute_comet_chrf.sh eval_final/wmt14/dictdis-dict-constraints-1-wiktionary1.de eval_final/wmt14/wmt14-dict-constraints.de eval_final/wmt14/wmt14-dict-constraints.en
bash compute_comet_chrf.sh eval_final/wmt14/indic-dict-constraints.de eval_final/wmt14/wmt14-dict-constraints.de eval_final/wmt14/wmt14-dict-constraints.en
bash compute_comet_chrf.sh eval_final/wmt14/leca-dict-constraints.de eval_final/wmt14/wmt14-dict-constraints.de eval_final/wmt14/wmt14-dict-constraints.en

bash compute_comet_chrf.sh eval_final/wmt14/vdba-dict-constraints.de eval_final/wmt14/wmt14-dict-constraints.de eval_final/wmt14/wmt14-dict-constraints.en