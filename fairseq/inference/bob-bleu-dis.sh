bash compute_bleu.sh bob/google-translate.txt bob/bob.hi  en hi
bash compute_bleu.sh bob/indictrans.txt bob/bob.hi  en hi
bash compute_bleu.sh bob/mbart50.txt bob/bob.hi  en hi
bash compute_bleu.sh bob/leca-original.txt bob/bob.hi  en hi
bash compute_bleu.sh bob/leca-ours.txt bob/bob.hi  en hi

python final_eval.py  bob/bob.en bob/google-translate.txt bob/bob.hi  ../../leca/Data/lexicalDict/  rbi
python final_eval.py  bob/bob.en bob/indictrans.txt bob/bob.hi  ../../leca/Data/lexicalDict/  rbi
python final_eval.py  bob/bob.en bob/mbart50.txt bob/bob.hi  ../../leca/Data/lexicalDict/  rbi
python final_eval.py  bob/bob.en bob/leca-original.txt bob/bob.hi  ../../leca/Data/lexicalDict/  rbi
python final_eval.py  bob/bob.en bob/leca-ours.txt bob/bob.hi  ../../leca/Data/lexicalDict/  rbi