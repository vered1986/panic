# PANiC - using Predicate-Arguments tuples to interpret Noun-Compounds

## What's in this repository?

This code is used for training a model of "world knowledge" about the relations between two nouns, and is used in interpreting noun-compounds in various task variants. For example, the model can describe the relation between `olive` and `oil` (in `olive oil`) as `oil is extracted from olive` or `oil is made from olives`. It can answer the question "what can be extracted from olives?" with `oil` (where "what can be extracted from almonds?" would produce different results, for example `milk`), and the question "what can oil be extracted from?".

## How and why?

[Nakov and Hearst (2006)](https://link.springer.com/chapter/10.1007/11861461_25) suggested that the semantics of a noun-compound could be expressed with multiple prepositional and verbal paraphrases. For example, `olive oil` is an oil `extracted from`, `made of`, or `obtained from` olives. We build upon this assumption, which was broadly used in the literature. 

We train a "world-knowledge" model using ReVerb [(Fader et al., 2011)](http://reverb.cs.washington.edu/emnlp11.pdf), a large dataset of OpenIE (arg1, relation, arg2) extractions from ClueWeb. We sample from the 15 million high-precision [ReVerb extractions](http://reverb.cs.washington.edu/reverb_clueweb_tuples-1.1.txt.gz) such that `arg1` and `arg2` each consist of a single word. We train a model that tries to predict one of the components `arg1` (a word), `arg2` (a word) or `relation` (a sequence of words) given the other two, practically answering the following three questions:

1. What is the distribution of `arg1`s in the world which have a `relation` relation to `arg2`?
2. What is the distribution of `arg2`s in the world to which `arg1` have a `relation`?
3. What is the distribution of relations between `arg1` and `arg2`?

We then use this to interpret noun-compounds - for a noun-compound `[w1] [w2]` (e.g. `[w1] = olive` and `[w2] = oil`) we can ask the model for the distribution of predicates between `arg1 = [w1] and arg2 = [w2]` or `arg1 = [w2] and arg2 = [w1]`. We use this in several tasks detailed below.

## How to use the code?

### Prerequisites

- Python 3
- docopt
- [dyNET](https://dynet.readthedocs.io)

### Training the Model

```
usage: train.py [-h] [--dynet_requested_gpus DYNET_REQUESTED_GPUS]
                [--dynet_mem DYNET_MEM] [--dynet_seed DYNET_SEED]
                [--nepochs NEPOCHS] [--batch_size BATCH_SIZE]
                [--patience PATIENCE]
                dataset model_dir embeddings_file
```

Where `dataset` is the filtered ReVerb file, `model_dir` is where the model should be saved, and `embeddings_file` is the path to word embeddings files (.npy and .vocab, created using this [script](https://github.com/vered1986/PythonUtils/blob/master/word_embeddings/format_convertion/convert_text_embeddings_to_binary.py)).

#### Noun-Compounds Interpretation Tasks

TBD

##### Paraphrasing

##### Relation Classification

##### Compositionality Grading
