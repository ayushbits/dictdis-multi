
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

Please cite as:

``` bibtex
@inproceedings{ott2019fairseq,
  title = {fairseq: A Fast, Extensible Toolkit for Sequence Modeling},
  author = {Myle Ott and Sergey Edunov and Alexei Baevski and Angela Fan and Sam Gross and Nathan Ng and David Grangier and Michael Auli},
  booktitle = {Proceedings of NAACL-HLT 2019: Demonstrations},
  year = {2019},
}
```
