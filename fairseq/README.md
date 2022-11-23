
--------------------------------------------------------------------------------

## Installation

- fairseq directory contain the v0.12 of fairseq repo
- install indic_nlp_library -and indic_nlp_resources from indictrans repo

```
git clone https://github.com/anoopkunchukuttan/indic_nlp_resources.git
git clone https://github.com/anoopkunchukuttan/indic_nlp_library.git
```

## Preprocessing
- bash preprocess.sh Data lexicalDict en hi
- Data/ contains raw_data folder where train.(en/hi), test.(en/hi), val.(en/hi) is present

## Train
- bash script/run.sh
- Check datadir (#109) and model variable
- trained model will be stored in checkpoints/

## Inference

- Inside fairseq folder, Run
- bash translate_file.sh <src_file>  <tgt_file> phy,math,chem,mech en hi 0/1
- Go inside inference folder 
- bash compute_bleu.sh <predicted_file> <tgt_file> en hi
- python final_eval.py <src_file> <predicted_file> <tgt_file>  <comma_separated_lexicon>




Please cite as:

``` bibtex

}
```
