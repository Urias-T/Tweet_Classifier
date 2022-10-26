## Model2

This model is a Convolutional model and [GloVe](https://nlp.stanford.edu/projects/glove/) 100D vectors were used for the text embedding. The key difference between this model and Model1 is that in this model, I used the Natural Language Tookit ([NLTK](https://www.nltk.org/)) in preprocesing the data. 
This helped both in removing stopwords adn lemmatizing the words in the dataset. Also, I pickled the Tokenizer that was fit on the training set and used it in preparing the test set for classsification as well.

An f1score of 0.77627 was attained on the final submission.

![submission_score](submission_score.PNG)