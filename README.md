# Natural Language Processing (NLP)
A collection of NLP projects/tutorials. 

## [Supervised Text Classification](https://github.com/karolisjan/Natural-Language-Processing/tree/master/supervised_classification)
Classification of the positive and negative sentiment in the [IMDB movie reviews](https://ai.stanford.edu/~amaas/data/sentiment/). 

- [Logistic Regression based model](https://github.com/karolisjan/Natural-Language-Processing/blob/master/supervised_classification/supervised_text_classification.ipynb)
- [Neural Network based model](https://github.com/karolisjan/Natural-Language-Processing/blob/master/supervised_classification/supervised_text_classification_w_Keras.ipynb) 

Neural Network based model was created with a [custom wrapper](https://github.com/karolisjan/Keras-Wrapper) written for [Keras](https://keras.io/) running on top of [TensorFlow](https://github.com/tensorflow/tensorflow).

A [Tf-Idf](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) matrix was used as input for both models. Both models achieved over 85% accuracy on a test set.  
