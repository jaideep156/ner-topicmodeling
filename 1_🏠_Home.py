import streamlit as st

st.set_page_config(page_title="Home", page_icon="🏠")
st.title("Named Entity Recognition & Topic modeling of news articles")

st.write("This project is aimed at performing [Named Entity Recognition](https://en.wikipedia.org/wiki/Named-entity_recognition) (using spaCy, BERT) & [Topic modeling](https://en.wikipedia.org/wiki/Topic_model) (using LDA, BERTopic) of news articles.")

st.write("### Head over to the respective pages on your left to see it in action!")

st.write("##### Trained on data which is available on [Kaggle](https://www.kaggle.com/datasets/banuprakashv/news-articles-classification-dataset-for-nlp-and-ml).")
st.write("##### You can find the full codebase & project specifics on my [GitHub](https://github.com/jaideep156/ner-topicmodeling).")