import streamlit as st
import pandas as pd
import random 
import spacy
from spacy import displacy 
nlp = spacy.load('en_core_web_sm')
st.set_page_config(page_title="NER using spaCy", page_icon="💬")
def load_data(file_name):
    return pd.read_csv(file_name)

entity_df = load_data("data/entity_df.csv")
df = load_data("data/display_data.csv")

st.markdown("# Named Entity Recognition using spaCy💬")
st.sidebar.markdown(" ## NER using spaCy💬")

st.markdown(f"For detailed steps on how we on how we are doing NER using spaCy, you can look at the [NER.ipynb](https://github.com/jaideep156/ner-topicmodeling/blob/main/notebook/NER.ipynb).")

st.write("We are importing the [display_data.csv](https://github.com/jaideep156/ner-topicmodeling/blob/main/data/display_data.csv) & below are the steps carried out:")
st.write(
    """
    - Preprocessing (lowercasing & removing special characters)
    - Finding out different entity types & visualizing them.
    - [`displacy`](https://spacy.io/usage/visualizers) to visualize the linguistic structure of the sentences of the original data. 
    """
)

st.subheader("10 random entity types from the dataset")
st.dataframe(entity_df.sample(10)) 
st.markdown("Following is the bar chart of the entity distribution of the whole data")
type_counts = entity_df['Type'].value_counts().reset_index()
st.bar_chart(type_counts.set_index('Type'))


st.subheader("Lingustic structure using `displacy` of a random observation")
random_index = random.randint(0, 10000)
original_text = df['content'][random_index]
st.write(f"Using the random observation number {random_index}.")

doc = nlp(original_text)

html = displacy.render(doc, style='ent', jupyter=False)
st.markdown(html, unsafe_allow_html=True)