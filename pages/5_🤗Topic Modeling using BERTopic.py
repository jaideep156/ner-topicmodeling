import streamlit as st
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re

st.set_page_config(page_title="Topic modeling using BERTopic", page_icon="🤗")

df = pd.read_csv("data/bertopic_results.csv")
st.title("Topic modeling using BERTopic")
st.markdown("Here, we are using [BERTopic](https://maartengr.github.io/BERTopic/index.html) for topic modeling. For all the detailed steps on how we are doing it, you can checkout [topicmodeling_BERTopic.ipynb](https://github.com/jaideep156/ner-topicmodeling/blob/main/notebook/topicmodeling_BERTopic.ipynb).")

st.markdown("We are limiting number of topics to 15 & below are 10 random observations from output after after applying the BERTopic model:")
st.dataframe(df.sample(10))

st.subheader("Distribution of topics in the collection of articles.")
topic_distribution = df['topic'].value_counts().sort_index()
st.bar_chart(topic_distribution)

def contnet_topwords(topic):
    topic_df = df[df['topic'] == topic]
    text = ' '.join(topic_df['content'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)
    

st.subheader('Top Words and WordCloud by Topic')
topic_selected = st.selectbox('Select a Topic', sorted(df['topic'].unique()))

topic_df = df[df['topic'] == topic_selected]
all_content = ' '.join(topic_df['content'])

words = re.findall(r'\b\w+\b', all_content.lower())

word_counts = pd.Series(words).value_counts()
top_words = word_counts.head(10)
st.write(f"Top Words for topic {topic_selected}")
st.write(top_words)

top_articles = topic_df.head(10)  
st.write(f"Top 10 Articles for Topic {topic_selected}")
st.write(top_articles[['content', 'topic']])

st.write(f"Word Cloud for topic {topic_selected}")
contnet_topwords(topic_selected)