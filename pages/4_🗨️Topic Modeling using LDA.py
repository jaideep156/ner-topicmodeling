import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt

st.set_page_config(page_title="Topic modeling LDA", page_icon="🗨️")
st.title("Topic modeling using Latent Dirichlet Allocation")

lda_df = pd.read_csv('data/LDA_topics.csv')
st.markdown("We are using [Latent Dirichlet Allocation](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation) for topic modeling. For all the detailed steps on how we are doing it, you can checkout [topicmodeling_LDA.ipynb](https://github.com/jaideep156/news-articles/blob/main/notebook/topicmodeling_LDA.ipynb).")

st.markdown("We are limiting number of topics to 15 & below are 10 random observations from output after LDA:")
st.dataframe(lda_df.sample(10))

st.subheader("Distribution of topics in the collection of articles.")
topic_distribution = lda_df['topic'].value_counts().sort_index()
st.bar_chart(topic_distribution)

def topic_content_top_words_wordcloud(topic_num):
    topic_articles = lda_df[lda_df['topic'] == topic_num]['content']
    
    if topic_articles.empty:
        st.write("No content available for this topic.")
        return

    st.write(topic_articles.head(10))
    
    vectorizer = CountVectorizer(stop_words='english', token_pattern=r'\b\w+\b')
    X = vectorizer.fit_transform(topic_articles)
    
    feature_names = vectorizer.get_feature_names_out()
    top_words = pd.Series(X.sum(axis=0).A1, index=feature_names)
    top_words = top_words.sort_values(ascending=False).head(20)
    top_words = top_words.rename_axis('word').reset_index(name='frequency')
    
    st.write(f"Top Words for Topic number {topic_num}")
    st.write(top_words)
    
    st.write(f"Wordcloud for Topic number {topic_num}")
    text_for_wordcloud = ' '.join(topic_articles)
    wordcloud = WordCloud(width=1000, height=800, background_color='white').generate(text_for_wordcloud)
  
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Word Cloud for Topic {topic_num}')
    
    st.pyplot(plt)

st.subheader('Topic content, its top words, & their respective word clouds')

unique_topics = sorted(lda_df['topic'].unique())

selected_topic_num = st.selectbox("Select a Topic Number out of 15", unique_topics)

st.write(f"Displaying results of topic number {selected_topic_num} that you chose above.")
topic_content_top_words_wordcloud(selected_topic_num)