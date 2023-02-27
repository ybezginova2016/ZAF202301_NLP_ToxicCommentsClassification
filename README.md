# 01_NLP_CommentClassification

## Project Description

The client for this project is an online shop selling different goods. It is onow launching a new service. Now the users can edit and add goods' description as they can in Wiki communities.

The client wants to get a tool, which tracks and finds different types of comments, labelling them accordingly to toxic and non-toxic and sending the latter ones to for review.

## Project Objective

The project goal is to build a machine learning algorithm for a binary text classification to label the clients' comments, using machine valid learning modelsa or BERT neural network.

**Quality Metric: F1-measure >= 0.75.**

## Project Methodology
- Data loading, data pre-processing, preparing subsamples for training and testing.
- Dealing to target features imbalance problem.
- Building different ML algorithms and measuring the quality score.
- Testing the best models using test data.

### Key Python Libraries for NLP
- from nltk.corpus import stopwords as nltk_stopwords
- from nltk.stem.wordnet import WordNetLemmatizer
- from sklearn.feature_extraction.text import CountVectorizer
- from sklearn.feature_extraction.text import TfidfVectorizer
