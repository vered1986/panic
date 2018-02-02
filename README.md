# **PANiC!**
## using **P**redicate-**A**rguments tuples to interpret **N**oun-**C**ompounds

## What's in this repository?

This code is used for training a model of "world knowledge" about the relations between two nouns, 
and is used in interpreting noun-compounds in various task variants. 
For example, the model can describe the relation between `olive` and `oil` (in `olive oil`) as 
`oil is extracted from olives` or `oil is made from olives`. 
It can answer the question "what can be extracted from olives?" with `oil` 
(where "what can be extracted from almonds?" would produce different results, such as `milk`), 
and the question "what can oil be extracted from?".

## How and why?

[Nakov and Hearst (2006)](https://link.springer.com/chapter/10.1007/11861461_25) suggested that the semantics of a noun-compound 
could be expressed with multiple prepositional and verbal paraphrases. 
For example, `olive oil` is an oil `extracted from`, `made of`, or `obtained from` olives. 
We build upon this assumption, which was broadly used in the literature. 

We train a "world-knowledge" model using ReVerb [(Fader et al., 2011)](http://reverb.cs.washington.edu/emnlp11.pdf), 
a large dataset of OpenIE triplets of `(arg1, relation, arg2)` extracted from ClueWeb. 
We use the 15 million high-precision [ReVerb extractions](http://reverb.cs.washington.edu/reverb_clueweb_tuples-1.1.txt.gz) 
which we filter such that `arg1` and `arg2` each consist of a single word with at least 3 characters, 
and each of the components appears at least 5 times in the dataset. 
We train a model that tries to predict one of the components 
`arg1` (a word), `arg2` (a word) or `relation` (a sequence of words) given the other two,
 practically answering the following three questions:

1. What is the distribution of `arg1`s in the world which have a `relation` relation to `arg2`?
2. What is the distribution of `arg2`s in the world to which `arg1` have a `relation` relation?
3. What is the distribution of relations between `arg1` and `arg2`?

We then use this to interpret noun-compounds - for a noun-compound `[w1] [w2]` 
(e.g. `[w1] = olive` and `[w2] = oil`) we can ask the model for the distribution of 
predicates between `arg1 = [w1] and arg2 = [w2]` or `arg1 = [w2] and arg2 = [w1]`. 
We use this in several tasks detailed below.

## How to use the code?

### Prerequisites

- Python 3
- [dyNET](https://dynet.readthedocs.io)
- ScikitLearn

### Training the Model

```
usage: train.py [-h] [--dynet_requested_gpus DYNET_REQUESTED_GPUS]
                [--dynet_mem DYNET_MEM] [--dynet_seed DYNET_SEED]
                [--nepochs NEPOCHS] [--batch_size BATCH_SIZE]
                [--patience PATIENCE] [--update]
                dataset model_dir embeddings_file

positional arguments:
  dataset               path to the training data
  model_dir             where to store the result
  embeddings_file       path to word embeddings files (.npy and .vocab)

optional arguments:
  -h, --help            show this help message and exit
  --dynet_requested_gpus DYNET_REQUESTED_GPUS
                        number of gpus to use 0-4, default=0
  --dynet_mem DYNET_MEM
                        set dynet memory
  --dynet_seed DYNET_SEED
                        Dynet random seed, default=3016748844
  --nepochs NEPOCHS     number of epochs
  --batch_size BATCH_SIZE
                        number of instance per minibatch
  --patience PATIENCE   how many epochs to wait without improvement
  --update              whether to update the embeddings
```

Where `dataset` is the filtered ReVerb file, 
`model_dir` is where the model should be saved, and `embeddings_file` is the path to word embeddings files 
(.npy and .vocab, created using this [script](https://github.com/vered1986/PythonUtils/blob/master/word_embeddings/format_convertion/convert_text_embeddings_to_binary.py)).

#### Noun-Compounds Interpretation Tasks

##### Paraphrasing

There are two variants of the noun-compound paraphrasing task:

1. **[SemEval-2010 Task 9: The Interpretation of Noun Compounds Using Paraphrasing Verbs and Prepositions](http://semeval2.fbk.eu)** -
A list of verb/preposition paraphrases are provided for each noun-compound, and for each list the system is asked to provide scores 
that correlate well (in terms of frequency distribution) with the human judgments.

2. **[SemEval 2013 Task 4: Free Paraphrases of Noun Compounds](https://www.cs.york.ac.uk/semeval-2013/task4/index.php)** - 
Given a two-word noun compound, the participating system is asked to produce 
an explicitly ranked list of its free-form paraphrases. The list is automatically compared and evaluated against a similarly ranked list 
of paraphrases proposed by human annotators.

We attempt to solve the first one by predicting the paraphrase vector of each noun-compound, 
and ranking the suggested paraphrases by the similarity of their vector (obtained by encoding the paraphrase using the model) 
and the predicted vector. 

For the second variant, we predict the paraphrase vector of each noun-compound, 
and then retrieve the `k` most similar paraphrases to it from the ReVerb corpus. 


##### Relation Classification

In this task, noun-compounds are annotated to a predefined set of relations, and the model has to predict the correct 
relation between the constituents of an unobserved compound. For example, `olive oil` may belong to the `SOURCE` relation 
while `morning meeting` belongs to the `TIME` relation. 

We train 3 models which are all tuned on the validation sets 
to select the classifier type (logistic regression and SVM with or without RBF kernel) 
and the regularization types and values. The models differ by:

* Distributional (baseline) - in which `w1` and `w2` are represented by a concatenation of their 300d GloVe embeddings.
* Paraphrase-based - in which we use our world-knowledge model to get a vector representing the relation from `w1` to `w2` and another representing the relation from `w2` to `w1`, and concatenate these to vectors.
* Integrated - a concatenation of both feature vectors detailed above.  

```
usage: classifier.py [-h] [--nepochs NEPOCHS] [--patience PATIENCE]
                     [--w1_w2_embeddings W1_W2_EMBEDDINGS]
                     [--paraphras_matrix PARAPHRAS_MATRIX]
                     dataset_prefix model_dir

positional arguments:
  dataset_prefix        path to the train/test/val/rel data
  model_dir             where to store the result

optional arguments:
  -h, --help            show this help message and exit
  --nepochs NEPOCHS     number of epochs
  --patience PATIENCE   how many epochs to wait without improvement
  --w1_w2_embeddings W1_W2_EMBEDDINGS
                        word embeddings to be used for the constituent words
  --paraphras_matrix PARAPHRAS_MATRIX
                        the path to the paraphrase matrix

```


##### Compositionality Grading

TBD