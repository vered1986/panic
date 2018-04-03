## Noun-Compound Paraphrasing

**[SemEval 2013 Task 4: Free Paraphrases of Noun Compounds](https://www.cs.york.ac.uk/semeval-2013/task4/index.php)** - 
Given a two-word noun compound, the participating system is asked to produce 
an explicitly ranked list of its free-form paraphrases. The list is automatically compared and evaluated against a similarly ranked list of paraphrases proposed by human annotators.

We predict for each noun-compound the `k` best paraphrases, and then learn to re-rank the suggested paraphrases using the SemEval training data. Training script:

```
usage: semeval_2013_train.py [-h] [--k K] [--minimum_score MINIMUM_SCORE]
                             [--unrelated_threshold UNRELATED_THRESHOLD]
                             train_gold_file language_model_dir patterns_file
                             word_embeddings

positional arguments:
  train_gold_file       a tsv file with gold train paraphrases and their scores
  language_model_dir    the path to the trained language model
  patterns_file         the file with the POS patterns
  word_embeddings       word embeddings to be used for the language model

optional arguments:
  -h, --help            show this help message and exit
  --k K                 the number of paraphrases to retrieve for re-rankning,
                        default = 1000
  --minimum_score MINIMUM_SCORE
                        the minimum score to keep a paraphrase
  --unrelated_threshold UNRELATED_THRESHOLD
                        the minimal score the "is unrelated to" paraphrase has to get to be included

```

Test script, produces the test paraphrases and evaluates them using the task scorer:

```
usage: semeval_2013_test.py [-h] [--k K] [--minimum_score MINIMUM_SCORE]
                            [--unrelated_threshold UNRELATED_THRESHOLD]
                            test_gold_file language_model_dir patterns_file
                            word_embeddings reranker

positional arguments:
  test_gold_file        a tsv file with gold test paraphrases and their scores
  language_model_dir    the path to the trained language model
  patterns_file         the file with the POS patterns
  word_embeddings       word embeddings to be used for the language model
  reranker              the pkl file for the trained re-ranker

optional arguments:
  -h, --help            show this help message and exit
  --k K                 the number of paraphrases to retrieve for re-rankning,
                        default = 1000
  --minimum_score MINIMUM_SCORE
                        the minimum score to keep a paraphrase
  --unrelated_threshold UNRELATED_THRESHOLD
                        the minimal score the "is unrelated to" paraphrase has to get to be included
```

Note that the directory needs to include the code for the SemEval scorer, that can be installed from [here](https://www.cs.york.ac.uk/semeval-2013/task4/index.php%3Fid=data.html).


Download pre-trained ranker from [here](https://drive.google.com/open?id=1M-sDIh3HNEXXGMR5jyqfebH4F4iVpR5n).