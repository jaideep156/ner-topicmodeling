import streamlit as st
import pandas as pd

st.markdown("# Exploratory Data Analysis📊")
st.sidebar.markdown(" ## Exploratory Data Analysis📊")


def load_data(file_name):
    return pd.read_csv(file_name)
file_name = "data/display_data.csv"

df = load_data(file_name)
st.markdown(f"Below are random 5 observations from [{file_name}](https://github.com/jaideep156/news-articles/blob/main/data/display_data.csv) dataset which is ready to be worked upon.")

st.write(df.sample(5))

st.markdown("#### Word cloud of the complete data.")
st.image('notebook/EDA_images/wordcloud_all.png', use_column_width=True)

st.markdown("#### Bigram Word cloud of the complete data.")
st.image('notebook/EDA_images/wordcloud_bigram_all.png', use_column_width=True)

st.markdown("#### Bar chart of the 15 most frequent words in the data.")
st.image('notebook/EDA_images/barchart_all.png', use_column_width=True)

st.markdown("#### Category wise bar chart of the 15 most frequent words in the data.")
st.image('notebook/EDA_images/top15_technology.png', use_column_width=True)
st.image('notebook/EDA_images/top15_business.png', use_column_width=True)
st.image('notebook/EDA_images/top15_sports.png', use_column_width=True)
st.image('notebook/EDA_images/top15_education.png', use_column_width=True)
st.image('notebook/EDA_images/top15_entertainment.png', use_column_width=True)