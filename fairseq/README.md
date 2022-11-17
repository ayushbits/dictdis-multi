
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

- bash translate_file.sh full_data/devtest/en-hi/testing/check.en pred-check.hi final-eng-hindi en hi 1
- bash inference/compute_bleu.sh pred-med.hi full_data/devtest/en-hi/medical/medical.hi hi

Please cite as:

``` bibtex

}
```
