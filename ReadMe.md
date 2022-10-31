# Tweet Classification 

## Natural Language Processing with Disaster Tweets.

**Data Source:** This data was sourced from [Kaggle](https://www.kaggle.com/competitions/nlp-getting-started/data)

**Description:** The [competition](https://www.kaggle.com/competitions/nlp-getting-started/overview) is for participants to build a model to predict which tweets are about real disasters and which ones are not.

**Results:**

- *Model1:* This model was averaging an f1score of about 0.5 on the test data despite extensive efforts with various model architectures and tuning to avoid overfitting. 

- *Model2:* With this model, I attained an f1 score of ~0.78 on the test data.

- *Model3:* Two model architectures were used. They are: 
   	1.) With BERT: An f1 score of ~0.76 was attained
	2.) with DistilBERT: An f1 score of ~0.73 was attained


**Possible Areas for Improvement:**

- One possible area worth exploring is to improve on fine-tuning the BERT and DistilBERT transformers.