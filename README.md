# **PANiC!**
## **PA**raphrasing **N**oun-**C**ompounds

## What's in this repository?

The repository contains the code used in the following paper:

Paraphrase to Explicate: Revealing Implicit Noun-Compound Relations. Vered Shwartz and Ido Dagan. ACL 2018.

This code is used for training a model that captures the semantic relations between two nouns, expressed in free text. We use this model to interpret noun-compounds in two task variants. 
For example, the model can describe the relation between `olive` and `oil` (in `olive oil`) as 
`oil is extracted from olives` or `oil is made of olives`. 
It can answer the question "what can be extracted from olives?" with `oil` 
(where "what can be extracted from almonds?" would produce different results, such as `milk`), 
and the question "what can oil be extracted from?" with `olive`.

## How and why?

[Nakov and Hearst (2006)](https://link.springer.com/chapter/10.1007/11861461_25) suggested that the semantics of a noun-compound could be expressed with multiple prepositional and verbal paraphrases. 
For example, `olive oil` is an oil `extracted from`, `made of`, or `obtained from` olives. 
We build upon this assumption, which was broadly used in the literature. 

We train a prarphrasing model using [Google N-grams](https://books.google.com/ngrams) as a corpus providing paraphrases. The model tries to predict one of the components 
`[w1]` (a word), `[w2]` (a word) or `relation` (a sequence of words) given the other two,
 practically answering the following three questions:

1. What is the distribution of `[w1]`s in the world which have a `relation` relation to `[w2]`?
2. What is the distribution of `[w2]`s in the world to which `[w1]` have a `relation` relation?
3. What is the distribution of relations between `[w1]` and `[w2]`?

## How to use the code?

### Prerequisites

- Python 3
- [dyNET](https://dynet.readthedocs.io)
- ScikitLearn

### Loading the Pre-trained Model

The pre-trained model can be downloaded from [here](https://drive.google.com/file/d/1TRB_hnwBkTKZVASV7iWJVjYIyxdv7yE-/view?usp=sharing).

### Training the Paraphrasing Model

```
usage: train.py [-h] [--dynet-devices DYNET_DEVICES] [--dynet-mem DYNET_MEM]
                [--dynet-seed DYNET_SEED] [--dynet-autobatch DYNET_AUTOBATCH]
                [--nepochs NEPOCHS] [--batch_size BATCH_SIZE]
                [--patience PATIENCE] [--update] [--dropout DROPOUT]
                [--negative_sampling_ratio NEGATIVE_SAMPLING_RATIO]
                [--negative_samples_weight NEGATIVE_SAMPLES_WEIGHT]
                [--continue_training] [--prune_paraphrases PRUNE_PARAPHRASES]
                [--filter_vocab]
                dataset_dir model_dir embeddings_file

positional arguments:
  dataset_dir           path to the data directory, where the files train.tsv and val.tsv are expected
  model_dir             where to store the result
  embeddings_file       path to word embeddings files (.npy and .vocab)

optional arguments:
  -h, --help            show this help message and exit
  --dynet-devices DYNET_DEVICES
                        the devices to use, e.g. "CPU,GPU:0,GPU:31"
                        default=CPU
  --dynet-mem DYNET_MEM
                        set dynet memory
  --dynet-seed DYNET_SEED
                        Dynet random seed, default=3016748844
  --dynet-autobatch DYNET_AUTOBATCH
                        whether to use autobatching (0/1)
  --nepochs NEPOCHS     number of epochs
  --batch_size BATCH_SIZE
                        number of instance per minibatch
  --patience PATIENCE   how many epochs to wait without improvement
  --update              whether to update the embeddings
  --dropout DROPOUT     dropout rate
  --negative_sampling_ratio NEGATIVE_SAMPLING_RATIO
                        the ratio from the training set of negative samples to add
  --negative_samples_weight NEGATIVE_SAMPLES_WEIGHT
                        the weight to assign to negative samples
  --continue_training   whether to load and keep training an existing model
  --prune_paraphrases PRUNE_PARAPHRASES
                        the minimal score for include paraphrases
  --filter_vocab        whether to load only the vocabulary embeddings (to save memory)
```

where `embeddings_file` is the path to word embeddings files 
(.npy and .vocab, created using this [script](https://github.com/vered1986/PythonUtils/blob/master/word_embeddings/format_convertion/convert_text_embeddings_to_binary.py)).

### Noun-Compounds Interpretation Tasks

#### Paraphrasing

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


#### Relation Classification

In this task, noun-compounds are annotated to a pre-defined set of relations, and the model has to predict the correct 
relation between the constituents of an unobserved compound. For example, `olive oil` may belong to the `SOURCE` relation 
while `morning meeting` belongs to the `TIME` relation. 

We train 3 models which are tuned on the validation sets to select the number of predicted paraphrases `k` classifier type (logistic regression and SVM), and the regularization types and values. The models differ by:

* Distributional (baseline) - in which `[w1]` and `[w2]` are represented by a concatenation of their 300d GloVe embeddings.
* Paraphrase-based - in which we use our paraphrasing model to get a vector representing the relation between `[w1]` and `[w2]`.
* Integrated - a concatenation of both feature vectors detailed above.  

```
uusage: classifier.py [-h] [--use_w1_w2_embeddings] [--use_paraphrase_vectors]
                     language_model_dir word_embeddings_for_model
                     word_embeddings_for_dist dataset_prefix model_dir

positional arguments:
  language_model_dir    the path to the trained language model
  word_embeddings_for_model
                        word embeddings to be used for the language model
  word_embeddings_for_dist
                        word embeddings to be used for w1 and w2 embeddings
  dataset_prefix        path to the train/test/val/rel data
  model_dir             where to store the result

optional arguments:
  -h, --help            show this help message and exit
  --use_w1_w2_embeddings
                        use w1 and w2 word embeddings as features
  --use_paraphrase_vectors
                        use the paraphrase vectors as features

```
