## Noun-Compound Relation Classification

In this task, noun-compounds are annotated to a pre-defined set of relations, and the model has to predict the correct 
relation between the constituents of an unobserved compound. For example, `olive oil` may belong to the `SOURCE` relation 
while `morning meeting` belongs to the `TIME` relation. 

We train 3 models which are tuned on the validation sets to select the number of predicted paraphrases `k` classifier type (logistic regression and SVM), and the regularization types and values. The models differ by:

* Distributional (baseline) - in which `[w1]` and `[w2]` are represented by a concatenation of their 300d GloVe embeddings.
* Paraphrase-based - in which we use our paraphrasing model to get a vector representing the relation between `[w1]` and `[w2]`.
* Integrated - a concatenation of both feature vectors detailed above.  

```
usage: classifier.py [-h] [--use_w1_w2_embeddings] [--use_paraphrase_vectors]
                     paraphrase_model_dir word_embeddings_for_model
                     word_embeddings_for_dist dataset_prefix model_dir

positional arguments:
  paraphrase_model_dir  the path to the trained paraphrasing model
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
