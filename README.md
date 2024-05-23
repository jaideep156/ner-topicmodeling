# Named Entity Recognition & Topic modeling of news articles

This project is aimed at performing [Named Entity Recognition](https://en.wikipedia.org/wiki/Named-entity_recognition) (using spaCy, BERT) & [Topic modeling](https://en.wikipedia.org/wiki/Topic_model) (using LDA, BERTopic) of actual news articles sourced from [Kaggle](https://www.kaggle.com/datasets/raqhea/medium-app-reviews-from-google-play-store/).

## To access the live version of the app, click [here](https://ner-topicmodeling.streamlit.app).

## Data
The data is fetched from [Kaggle](https://www.kaggle.com/datasets/raqhea/medium-app-reviews-from-google-play-store/) and first, we merge all the different datasets (like business, education, etc.) in to a master dataset called [`master_data.csv`](https://github.com/jaideep156/ner-topicmodeling/blob/main/data/master_data.csv).

Then, we remove irrelavant columns and save the resulting data as [`display_data.csv`](https://github.com/jaideep156/ner-topicmodeling/blob/main/data/display_data.csv). This is the data will be used everywhere for all the tasks here on. 

## Dependencies
- [pandas](https://pandas.pydata.org/docs/index.html) to manipulate the data.
- [scikit-learn](https://scikit-learn.org/stable/) for machine learning tasks.
- [matplotlib](https://matplotlib.org/) for visualizations.
- [NLTK](https://www.nltk.org/) for NLP tasks like tokenization, removing stop words, etc. 
- [spaCy](https://spacy.io/) which has pre-trained statistical models for common NLP tasks.
- [transformers]() which is a framework from [HuggingFace🤗](https://huggingface.co/) that has pre-trained models for NLP tasks.
- [BERTopic](https://maartengr.github.io/BERTopic/index.html) which is a topic modeling framework built on top of Transformers by HuggingFace🤗
- [streamlit cloud](https://streamlit.io/cloud) to deploy the app. 

## Methodology
First we do the exploratory data analysis on [`display_data.csv`](https://github.com/jaideep156/ner-topicmodeling/blob/main/data/display_data.csv) that includes wordclouds.

You can follow [notebook.ipynb](https://github.com/jaideep156/ner-topicmodeling/blob/main/notebook/notebook.ipynb) for a detailed walkthrough.


### Named Entity Recognition using spaCy
We are importing the [`display_data.csv`](https://github.com/jaideep156/ner-topicmodeling/blob/main/data/display_data.csv) & below are the steps carried out:
- Preprocessing (lowercasing & removing special characters)
- Using spaCy to find out different entity types (saving the output dataframe as [`entity_df.csv`](https://github.com/jaideep156/ner-topicmodeling/blob/main/data/entity_df.csv)) & visualizing them using a bar chart.
- Use [displacy](https://spacy.io/usage/visualizers) to visualize the linguistic structure of the sentences of the original data.

For the detailed steps you can look at [NER.ipynb](https://github.com/jaideep156/ner-topicmodeling/blob/main/notebook/NER.ipynb).

### Named Entity Recognition using BERT by HuggingFace🤗
We are importing the [`display_data.csv`](https://github.com/jaideep156/ner-topicmodeling/blob/main/data/display_data.csv) & below are the steps carried out:
- First we load the pre-trained NER model called [bert-large-cased-finetuned-conll03-english](https://huggingface.co/dbmdz/bert-large-cased-finetuned-conll03-english) & train this model on it.
- Next, I save the output dataframe as [`entity_bert.csv`](https://github.com/jaideep156/ner-topicmodeling/blob/main/data/entity_bert.csv)
- Finally, I print a random observation & its respective entities.

For the detailed steps you can look at [NER_BERT.ipynb](https://github.com/jaideep156/ner-topicmodeling/blob/main/notebook/NER_BERT.ipynb).

### Topic modeling using Latent Dirichlet Allocation 
We are importing the [`display_data.csv`](https://github.com/jaideep156/ner-topicmodeling/blob/main/data/display_data.csv) & below are the steps carried out:

- First, I apply the LDA model & limit the topics to 15. Now, the LDA model will classify the articles in 15 topics (numbered from 1 to 15). 
- Next, I saved the output as [`LDA_topics.csv`](https://github.com/jaideep156/ner-topicmodeling/blob/main/data/LDA_topics.csv)
- Finally, I visualize it as occurences of top 10 words, topic wise articles and its respective wordcloud.  

For the detailed steps you can look at [topicmodeling_LDA.ipynb](https://github.com/jaideep156/ner-topicmodeling/blob/main/notebook/topicmodeling_LDA.ipynb).

### Topic modeling using BERTopic
We are importing the [`display_data.csv`](https://github.com/jaideep156/ner-topicmodeling/blob/main/data/display_data.csv) & below are the steps carried out: 
- Preprocessing (lowercasing & removing special characters)
- Training BERTopic & saving the results as [`bertopic_results.csv`](https://github.com/jaideep156/ner-topicmodeling/blob/main/data/bertopic_results.csv)
- Finally, I visualize it as occurences of top 10 words, topic wise articles and its respective wordcloud.
For the detailed steps you can look at [topicmodeling_BERTopic.ipynb](https://github.com/jaideep156/ner-topicmodeling/blob/main/notebook/topicmodeling_BERTopic.ipynb).

## Run Locally

Clone the project

```bash
  git clone https://github.com/jaideep156/ner-topicmodeling
```

Go to the project directory

```bash
  cd ner-topicmodeling
```

Install dependencies

```bash
  pip install -r requirements.txt
```

Start the server

```bash
  streamlit run 1_🏠_Home.py
```

## Deployment
This code has been deployed using [Streamlit Community Cloud](https://streamlit.io/cloud) and the file is [`1_🏠_Home.py`](https://github.com/jaideep156/ner-topicmodeling/blob/main/1_%F0%9F%8F%A0_Home.py)