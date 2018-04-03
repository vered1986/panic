## Noun-Compound Compositionality Prediction

A noun-compound `[w1] [w2]` is considered compositional if the meaning of the compound is derived from the meaning of its constituent words `[w1]` and `[w2]`.
In this task, the model needs to predict compositionality scores for noun-compounds, and is evaluated against human judgements.

Human judgements are taken from: 
Siva Reddy, Diana McCarthy and Suresh Manandhar. [An Empirical Study on Compositionality in Compound Nouns](http://www.aclweb.org/anthology/I11-1024) IJCNLP (2011), which is also available in [Kaggle](https://www.kaggle.com/rtatman/noun-compositionality-judgements).

In this dataset, noun-compounds are scored according to what extent they are compositional, in a scale of 0-5, 0 being non-compositional and 5 being compositional:

1.  To what extent is `[w1] [w2]` derived from `[w1]`? e.g. `guilt trip` is about `guilt` but is not really a `trip`.
2.  To what extent is `[w1] [w2]` derived from `[w2]`? e.g. `snail mail` is `mail` which is as slow as a `snail` but is not directly derived from `snail`.
3.  To what extent is `[w1] [w2]` derived from `[w1]` and `[w2]`? e.g. `mailing list` is derived from both `mailing` and `list`.

The file in the `data` directory is tab-separated with the following fields: `[w1]`, `[w2]`, `score 1`, `score 2`, `score 3`.

The script `predictor.py` was used in our compositionality analysis:

```
usage: predictor.py [-h] [--k K]
                    paraphrase_model_dir word_embeddings_for_model
                    dataset_file output_file

positional arguments:
  paraphrase_model_dir  the path to the trained paraphrasing model
  word_embeddings_for_model
                        word embeddings to be used for the language model
  dataset_file          the path to the human judgements
  output_file           where to store the result

optional arguments:
  -h, --help            show this help message and exit
  --k K                 the number of paraphrases to retrieve, default = 15
```