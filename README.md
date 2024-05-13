# Introduction

The dataset usually consists of a large number of movie reviews. These
reviews can range from short sentences to long paragraphs, often longer
than 200 words. The reviews are typically divided into training,
validation, and testing sets, allowing participants to train and
fine-tune their models before making final predictions.

## Team

Please list your team members and describe their responsibilities:
**Artem Danyaev** found the baseline model paper and adopted it, added
models comparison. **Mikhail Bekusov** search for models, selection of
hyperparameters, training models, wrote the report, added models
comparison. **Ramiı Zay nullin**, search for models, training a model,
wrote the report, added models comparison.

# Related Work

-   [**Sentiment Analysis of Movie Reviews: A New Feature Selection
    Method**](https://hcis-journal.springeropen.com/articles/10.1186/s13673-018-0135-8):
    This paper presents a new feature selection method for sentiment
    analysis of movie reviews. The authors propose a method that
    combines statistical measures and semantic information, which
    improves the performance of sentiment classification.

-   [**Large-Scale Sentiment Analysis for News and
    Blogs**](https://www.icwsm.org/papers/3--Godbole-Srinivasaiah-Skiena.pdf):
    While not directly related to movie reviews, this work on
    large-scale sentiment analysis for news and blogs might provide
    useful methodologies and techniques that can be applied to movie
    review sentiment analysis.

-   [**Twitter as a Corpus for Sentiment Analysis and Opinion
    Mining**](http://www.lrec-conf.org/proceedings/lrec2010/pdf/385_Paper.pdf):
    This paper leverages Twitter data for sentiment analysis and opinion
    mining. The methodologies and findings could be adapted for movie
    review data.

-   [**Sentiment Analysis and
    Subjectivity**](https://www.cs.uic.edu/~liub/FBS/NLP-handbook-sentiment-analysis.pdf):
    This is a handbook chapter that provides a good overview of the
    field of sentiment analysis, including techniques for subjectivity
    analysis which could be very relevant for analyzing movie reviews.

-   [**Deep Learning for Sentiment Analysis: A
    Survey**](https://arxiv.org/abs/1801.07883): This work provides a
    comprehensive survey of various deep learning methods applied to
    sentiment analysis. It could provide insights into advanced
    techniques for classifying movie review sentiments.

# Model Description

The *AutoTokenizer* is used to convert the input text into a format that
the model can understand, and it also handles the truncation of the
input. The *AutoModelForSequenceClassification* is a type of model
designed for text classification tasks, such as sentiment analysis in
this case. The *DataCollatorWithPadding* is used to handle the padding
of the sequences to the same length, which is necessary when batching
sequences together. The *tokenizer* is passed to the
*DataCollatorWithPadding* to ensure the padding token used is consistent
with the tokens used by the model.

This approach provides more flexibility than traditional methods as it
allows for the use of different pre-trained models by simply changing
the 'checkpoint' variable. This is a significant improvement over
traditional methods, which often require extensive feature engineering
and model tuning.

# Base Model Description

The underlying formula for the model isn't a simple mathematical
equation like the area of a circle. The model is based on a complex
architecture called Transformers, which is used in natural language
processing tasks.

However, the core idea behind Transformers can be summarized by the
formula of the Scaled Dot-Product Attention mechanism, which is a key
component of the Transformer architecture:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Here: - $Q$, $K$, and $V$ are the query, key, and value vectors. These
are high-dimensional vectors that the model uses to understand the input
text. - The dot product $QK^T$ is used to calculate a score that
represents how much focus should a part of the sentence get. This score
is then scaled down by a factor of $\sqrt{d_k}$, where $d_k$ is the
dimension of the key vectors. - The softmax function is applied to
convert these scores to probabilities, ensuring that they are positive
and sum up to 1. - Finally, these probabilities are used to create a
weighted sum of the value vectors, which results in the output of the
attention mechanism.

This is a very high-level overview of the formula used in the
Transformer model. The actual implementation involves multiple layers of
these attention mechanisms, along with several other components like
position-wise feed-forward networks and normalization layers.

# Dataset

The dataset comprises two columns:

1.  Column *text* -- this column contains the review of the movie.

2.  Column *label* -- this column contains the sentiment label of the
    movie review.

The dataset is divided into three separate files:

1.  File *train.csv* -- this file is used for training the model.

2.  File *test.csv* -- this file is used for testing the model.

3.  File *valid.csv* -- this file is used for validating the model.

::: center
              Train   Valid   Test
  ---------- ------- ------- ------
  Articles    40000   5000    5000
  0 count     15019   2500    2500
  1 count     14981   2500    2500

  : Statistics of the Shai dataset.
:::

The target for this competition is the `label` column, where `0`
represents negative sentiment and `1` represents positive sentiment.

# Experiments

We compared the performance of six pre-trained models from HuggingFace
on the sentiment analysis task:

**mrm8488/camembert-base-finetuned-movie-review-sentiment-analysis:**
This model is based on the CamemBERT architecture, specifically
fine-tuned for movie review sentiment analysis.
**ashok2216/gpt2-amazon-sentiment-classifier:** This model is based on
the GPT-2 architecture, originally trained on Amazon reviews and
fine-tuned for sentiment classification.
**JamesH/Movie_review_sentiment_analysis_model:** This model is a
DeBERTa model, specifically designed for sentiment analysis tasks.
**google-bert/bert-base-uncased:** This is the base, uncased version of
the BERT model. **cardiffnlp/twitter-roberta-base-sentiment-latest:**
This model is a RoBERTa base model fine-tuned for sentiment analysis on
Twitter data. **siebert/sentiment-roberta-large-english:** This model is
a large, English-cased RoBERTa model fine-tuned for sentiment analysis.
All models, except the DeBERTa v2 model, were trained with optimizers
commonly used for fine-tuning pre-trained models for sentiment analysis
tasks. The following hyperparameters were used:

-   Learning rate: between 2e-5 and 1e-6

-   Weight decay: 5e-3

-   Epochs: 5

The DeBERTa v2 model employed a 5-folds bagging ensemble approach. It
used the Adam optimizer with a learning rate of 1e-6 and weight decay of
5e-3 for 5 epochs. Bagging is an ensemble technique that can improve
model robustness and potentially reduce variance.

# Metrics

The evaluation metric used for this task is **Binary Accuracy**. Binary
accuracy is a common metric used for binary classification tasks, like
sentiment analysis where the output is either positive or negative.

Binary accuracy is calculated as follows:

$$\text{Binary Accuracy} = \frac{\text{Number of Correct Predictions}}{\text{Total Number of Predictions}}$$

In the context of this dataset, a correct prediction means that the
sentiment predicted by the model matches the actual sentiment label in
the dataset. The binary accuracy ranges from 0 to 1, where 1 means the
model's predictions are 100% accurate.

It's important to note that while binary accuracy is a useful metric, it
might not provide a complete picture of the model's performance,
especially if the dataset is imbalanced (i.e., there are many more
reviews of one sentiment than the other). In such cases, other metrics
like Precision, Recall, or F1-score might also be useful.

## Baselines

-   **BERT**: While BERT is a widely used and well-understood model, it
    lacks the disentangled attention mechanism present in DeBERTa. This
    could potentially affect its performance on long sentences.

-   **RoBERTa**: RoBERTa, a variant of BERT, often outperforms BERT and
    could be a strong competitor to DeBERTa. It uses a different
    training approach and has larger training data.

-   **GPT-2**: GPT-2 is a generative model and is more commonly used for
    tasks different from classification. However, it can still serve as
    a baseline for comparison.

-   **XLNet**: XLNet uses a permutation-based training strategy and
    often has comparable performance to BERT. It could be another strong
    baseline model.

-   **DistilBERT**: DistilBERT, a smaller and faster version of BERT,
    could be a good baseline if there's interest in the trade-off
    between model size/performance and accuracy.

-   **SiEBERT**: SiEBERT enables reliable binary sentiment analysis for
    various types of English-language text. The model was fine-tuned and
    evaluated on 15 datasets from diverse text sources to enhance
    generalization across different types of texts (reviews, tweets,
    etc.). Consequently, it outperforms models trained on only one type
    of text (e.g., movie reviews from the popular SST-2 benchmark) when
    used on new data1.

# Results

##### This section now showcases the performance of each model.

The table below summarizes the performance of each model on the movie
review sentiment analysis task:

  | Model                                                              | \"Teacher\" |  Best Accuracy
  | ------------------------------------------------------------------ | ----------- |----------------
  | mrm8488/camembert-base-finetuned-movie-review-sentiment-analysis   |   Mikhail   |       0.905
  | google-bert/bert-base-uncased                                      |    Artem    |       0.932
  | JamesH/Movie_review_sentiment_analysis_model (DeBERTa)             |   Mikhail   |      0.9635
  | siebert/sentiment-roberta-large-english                            |    Ramil    |       0.953
  | cardiffnlp/twitter-roberta-base-sentiment-latest                   |    Artem    |       0.940
  | ashok2216/gpt2-amazon-sentiment-classifier                         |   Mikhail   |       0.924
  | JamesH/Movie_review_sentiment_analysis_model (DeBERTa_v2)          |    Artem    |    **0.9655**

  : Performance of pre-trained models on sentiment analysis task.

As shown in the table, the DeBERTa v2 model achieved the highest
accuracy of 0.9655, followed by the original DeBERTa model (0.9635), the
GPT-2 model (0.924), the sentiment-Roberta-large model (0.953), BERT
(0.932), RoBERTa (0.940), and CamemBERT (0.905). This suggests that the
DeBERTa architecture, particularly with the ensemble technique used in
DeBERTa v2, performs well for sentiment analysis of movie reviews. It's
possible that the bagging approach helps reduce variance and improve the
model's generalization ability.\
The sentiment-Roberta-large model achieved a competitive accuracy of
0.953, indicating the potential of using larger RoBERTa models for this
task. However, it fell short of the DeBERTa models. Further exploration
with hyperparameter tuning for this model and the other models could
potentially improve their performance.

##### More information about the competition, participants and results can be found at this link

[Shai
Competition](https://www.kaggle.com/competitions/shai-training-2024-a-level-2/leaderboard)\
Predictions: - bad - good

  Text                                                                                                            Prediction
  -------------------------------------------------------------------------------------------------------------- ------------
  I always wrote this series off as being a complete stink-fest because Jim Belushi was involved\...                  0
  1st watched 12/7/2002 - 3 out of 10(Dir-Steve Purcell): Typical Mary Kate & Ashley fare with a few more\...         0
  This movie was so poorly written and directed I fell asleep 30 minutes through the movie. The jokes\...             0
  The most interesting thing about Miryang (Secret Sunshine) is the actors. Jeon Do-yeon, as Lee Shin-ae\...          1
  when i first read about \"\"berlin am meer\"\" i didn't expect much. but i thought with the right people\...        0
  I saw this film on September 1st, 2005 in Indianapolis. I am one of the judges for the Heartland Film\...           1
  I saw a screening of this movie last night. I had high expectations going into it, but was definitely\...           0
  William Hurt may not be an American matinee idol anymore, but he still has pretty good taste\...                    1
  IT IS A PIECE OF CRAP! not funny at all. during the whole movie nothing ever happens. i almost\...                  0
  I'M BOUT IT(1997)\<br /\>\<br /\>Developed & published by No Limit Films\<br /\>\<br /\>\>\>Pros:\...               0

  : Predicts of our best pre-trained model on Shai Training task.

# Conclusion

The Kaggle competition involves sentiment analysis on a movie review
dataset. The reviews are long sentences, most of which are longer than
200 words. The evaluation metric for the competition is binary accuracy.

The DeBERTa model, with its disentangled attention mechanism, is a
strong candidate for this task. However, it's important to compare its
performance with other baseline models to ensure the best model is
chosen for the task.

In conclusion, while DeBERTa is a promising model for the task, it's
crucial to compare its performance with these baseline models to choose
the best model for the sentiment analysis task in this competition.
