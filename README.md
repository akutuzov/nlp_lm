# NLP through language modeling

Examples of language modeling approaches (for HSE workshop)

Python version >= 3.6 is required

# Training models

usage: `lm.py [-h] --train TRAIN --model {random,freq,trigram,rnn} [--save SAVE]`

optional arguments:

  `-h, --help  show this help message and exit`

  `--train TRAIN, -t TRAIN Path to training file (plain text)`

  `--model {random,freq,trigram,rnn}, -m {random,freq,trigram,rnn}`

  `--save SAVE, -s SAVE  Save model to...`

*Example*

`python3.6 lm.py -t my_corpus.txt.gz -m rnn -s model.h5`

# Testing models

usage: `test_lm.py [-h] --test TEST --model {random,freq,trigram,rnn} --modelfile MODELFILE`

optional arguments:
  `-h, --help            show this help message and exit`
  
  `--test TEST, -t TEST  Path to testing file (plain text)`
  
  `--model {random,freq,trigram,rnn}, -m {random,freq,trigram,rnn}`
  
  `--modelfile MODELFILE, -mf MODELFILE File name`


More corpora and a non-lemmatized word embedding model for Russian can be found at: 

http://ls.hpc.uio.no/~andreku/lm/


PS. To save RAM in the process of training a trigram model, 
consider using [bounter](https://github.com/RaRe-Technologies/bounter) instead of `Counter()`.
However, it doesn't play well with `TensorFlow`.
