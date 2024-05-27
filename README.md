
Source Code for the paper `DictDis: Dictionary Constrained Disambiugation for Improved NMT`
## Installation

- fairseq directory contain the v0.12 of fairseq repo
- install indic_nlp_library -and indic_nlp_resources from indictrans repo

```
git clone https://github.com/anoopkunchukuttan/indic_nlp_resources.git
git clone https://github.com/anoopkunchukuttan/indic_nlp_library.git
```
- `pip install flashtext evaluate unbabel-comet`
## Preprocessing
- bash preprocess.sh Data lexicalDict en hi
- Data/ contains raw_data folder where train.(en/hi), test.(en/hi), val.(en/hi) is present

## Path change in files
- change indic_nlp_lib_home path in scripts/preprocess_translate.py, scripts/postprocess_translate.py and scripts/detokenize.py

## Train
- bash script/run.sh
- Check datadir (#109) and model variable
- trained model will be stored in checkpoints/

## Inference

- Inside fairseq folder, Run
- bash translate_file.sh <src_file>  <tgt_file> phy,math,chem,mech en hi 0/1
- Go inside inference folder 
- bash compute_bleu.sh <predicted_file> <ref_file> en hi
- python final_eval.py<predicted_file> <ref_file>  <src_file>  <comma_separated_lexicon>

### COMET and ChrF
- conda activate translation
- Execute bash script: `bash compute_comet_chrf.sh <pred> <ref> <src> `
- python comet_compute.py flores/flores.en flores/flores.hi flores/flores-pred-1-new.hi
- python chrf_compute.py  flores/flores.hi flores/flores-pred-1-new.hi


Please cite as:

``` bibtex

}
```
